import torch
import torch.nn as nn
import torch.nn.functional as F


class classifier_hateClipper_llm(nn.Module):
    """
    Pruned classifier_hateClipper_llm for the current training setup:
      - dataset: FB -> binary output only
      - batch_norm: False -> no BatchNorm branch
      - active LLM backbone: fixed FiLM + semantic residual + layer combination
      - loss compatibility: exposes get_aux_loss(labels)

    The constructor signature is intentionally kept compatible with the original
    training code. The `fusion_mode` argument is accepted but not used because
    this LLM variant already uses its own fixed FiLM-style fusion backbone.
    """

    def __init__(
        self,
        image_dim,
        text_dim,
        exp_dim,
        num_layers,
        proj_dim,
        map_dim,
        fusion_mode="align",
        dropout=None,
        batch_norm=False,
        args=None,
    ) -> None:
        super(classifier_hateClipper_llm, self).__init__()

        if dropout is None:
            dropout = [0.1, 0.1, 0.1]
        if len(dropout) != 3:
            raise ValueError("dropout must contain three values: [input_drop, fusion_drop, head_drop]")
        if batch_norm:
            raise ValueError("This pruned FB LLM classifier assumes --batch_norm False.")
        if args is not None and hasattr(args, "dataset") and args.dataset != "FB":
            raise ValueError("This pruned classifier is specialized for --dataset FB.")

        self.map_dim = map_dim
        self.output_dim = 1

        # ========= 1. Base projections =========
        self.img_proj = nn.Linear(image_dim, map_dim)
        self.text_proj = nn.Linear(text_dim, map_dim)

        # Keep the original module name `exp_proj` for checkpoint compatibility.
        # In the pruned loss, the third input is `context_feats`.
        self.exp_proj = nn.Linear(exp_dim, map_dim)

        self.img_norm = nn.LayerNorm(map_dim)
        self.text_norm = nn.LayerNorm(map_dim)
        self.exp_norm = nn.LayerNorm(map_dim)

        self.img_drop = nn.Dropout(dropout[0])
        self.text_drop = nn.Dropout(dropout[0])
        self.exp_drop = nn.Dropout(min(dropout[0], 0.05))

        # ========= 2. Context / explanation adapter =========
        self.exp_adapter = nn.Sequential(
            nn.Linear(map_dim, map_dim),
            nn.SiLU(),
            nn.Dropout(min(dropout[0], 0.05)),
        )

        # This active LLM variant uses a fixed concat-style image-text base.
        base_dim = map_dim * 2
        gate_in_dim = map_dim * 3

        self.base_ln = nn.LayerNorm(base_dim)
        self.exp_gate = nn.Sequential(
            nn.Linear(gate_in_dim, map_dim),
            nn.Sigmoid(),
        )

        # ========= 3. FiLM branch =========
        self.film_generator = nn.Linear(map_dim, base_dim * 2)
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)
        self.film_alpha = nn.Parameter(torch.tensor(0.0))

        # ========= 4. Semantic residual branch =========
        self.sem_res_proj = nn.Linear(map_dim, base_dim)
        nn.init.zeros_(self.sem_res_proj.weight)
        nn.init.zeros_(self.sem_res_proj.bias)

        self.sem_gate = nn.Sequential(
            nn.Linear(gate_in_dim, base_dim),
            nn.Sigmoid(),
        )
        self.sem_alpha = nn.Parameter(torch.tensor(0.0))

        # ========= 5. Layer combination =========
        self.fusion_ln = nn.LayerNorm(base_dim)
        self.layer_combine_logits = nn.Parameter(
            torch.tensor([-1.5, -1.0, 1.5], dtype=torch.float)
        )
        self.layer_combine_ln = nn.LayerNorm(base_dim)

        # ========= 6. Classification head =========
        if num_layers <= 0:
            self.mlp = nn.Identity()
            final_dim = base_dim
        else:
            layers = []
            cur_dim = base_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(cur_dim, proj_dim))
                layers.append(nn.LayerNorm(proj_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout[2]))
                cur_dim = proj_dim
            self.mlp = nn.Sequential(*layers)
            final_dim = proj_dim

        self.final_dim = final_dim
        self.head_drop = nn.Dropout(dropout[1])
        self.output_layer = nn.Linear(final_dim, self.output_dim)

        # ========= 7. Attribution-guided counterfactual module =========
        self.cf_topk_ratio = getattr(args, "cf_topk_ratio", 0.12) if args is not None else 0.12
        self.cf_mask_value = getattr(args, "cf_mask_value", 0.05) if args is not None else 0.05
        self.cf_fuse_alpha = nn.Parameter(torch.tensor(0.0))
        self.cf_margin = getattr(args, "cf_margin", 0.23) if args is not None else 0.23
        self.cf_neg_margin = getattr(args, "cf_neg_margin", 0.05) if args is not None else 0.05

        self._cached_linear_output = None
        self._cached_cf_output = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # These two are intentionally zero-initialized above so the model
                # starts close to the base branch.
                if m is self.film_generator or m is self.sem_res_proj:
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def _build_counterfactual_embed(self, embed, linear_output):
        """Build attribution-guided counterfactual embeddings for binary FB classification."""
        del linear_output  # binary path only uses the positive-class classifier weight

        with torch.no_grad():
            cls_w = self.output_layer.weight[0].abs().unsqueeze(0)  # [1, D]
            importance = embed.abs() * cls_w                         # [B, D]

            k = max(1, int(embed.size(1) * self.cf_topk_ratio))
            topk_idx = importance.topk(k=k, dim=1, largest=True, sorted=False).indices

            mask = torch.ones_like(embed)
            mask.scatter_(1, topk_idx, self.cf_mask_value)

        return embed * mask

    def get_aux_loss(self, labels):
        """
        Auxiliary counterfactual loss for binary FB classification.

        Positive samples: masking important evidence should lower the positive logit.
        Negative samples: masking important dimensions should not raise the positive logit too much.
        """
        if self._cached_linear_output is None or self._cached_cf_output is None:
            return torch.tensor(0.0, device=self.film_alpha.device)

        y = labels.float().view(-1, 1)
        orig = self._cached_linear_output
        cf = self._cached_cf_output

        pos_loss = F.relu(self.cf_margin - (orig - cf))
        neg_loss = F.relu((cf - orig) - self.cf_neg_margin)
        return (y * pos_loss + (1.0 - y) * neg_loss).mean()

    def forward(self, img_feats, text_feats, context_feats, return_embed=False):
        # ========= 1. Base features =========
        v_img = self.img_proj(img_feats)
        v_txt = self.text_proj(text_feats)
        v_ctx = self.exp_proj(context_feats)

        v_img = self.img_norm(v_img)
        v_txt = self.text_norm(v_txt)
        v_ctx = self.exp_norm(v_ctx)

        v_img = self.img_drop(v_img)
        v_txt = self.text_drop(v_txt)
        v_ctx = self.exp_drop(v_ctx)

        # ========= 2. Image-text base =========
        x_base = torch.cat((v_img, v_txt), dim=1)
        x_base = self.base_ln(x_base)

        # ========= 3. Context adapter and gate =========
        h_ctx = self.exp_adapter(v_ctx)
        ctx_gate_in = torch.cat((v_img, v_txt, h_ctx), dim=1)
        ctx_gate = self.exp_gate(ctx_gate_in)
        h_ctx = h_ctx * ctx_gate

        # ========= 4. FiLM =========
        film_params = self.film_generator(h_ctx)
        gamma, beta = torch.chunk(film_params, 2, dim=1)

        gamma = 0.05 * torch.tanh(gamma)
        beta = 0.05 * torch.tanh(beta)
        film_delta = gamma * x_base + beta

        x_film = x_base + torch.tanh(self.film_alpha) * film_delta

        # ========= 5. Semantic residual =========
        sem_res = 0.05 * torch.tanh(self.sem_res_proj(h_ctx))
        sem_gate_in = torch.cat((v_img, v_txt, h_ctx), dim=1)
        sem_gate = self.sem_gate(sem_gate_in)
        x_sem = x_film + torch.tanh(self.sem_alpha) * (sem_gate * sem_res)

        # ========= 6. Layer combination =========
        x_base_n = self.fusion_ln(x_base)
        x_film_n = self.fusion_ln(x_film)
        x_sem_n = self.fusion_ln(x_sem)

        layer_weights = F.softmax(self.layer_combine_logits, dim=0)
        x = (
            layer_weights[0] * x_base_n
            + layer_weights[1] * x_film_n
            + layer_weights[2] * x_sem_n
        )
        x = self.layer_combine_ln(x)

        # ========= 7. Main classifier =========
        x = self.head_drop(x)
        embed = self.mlp(x)
        linear_output = self.output_layer(embed)

        # ========= 8. Counterfactual branch =========
        embed_cf = self._build_counterfactual_embed(embed, linear_output)
        cf_output = self.output_layer(embed_cf)

        self._cached_linear_output = linear_output
        self._cached_cf_output = cf_output

        cf_gap = linear_output - cf_output
        output = linear_output + torch.tanh(self.cf_fuse_alpha) * cf_gap

        if return_embed:
            return output, embed
        return output
