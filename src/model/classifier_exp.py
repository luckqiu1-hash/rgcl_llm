import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceStyleVectorTokenizer(nn.Module):
    """
    Convert one pooled vector [B, D] into several modality tokens [B, T, D].

    MODA source operates on visual/text token sequences.
    Your current model receives pooled vectors, so this tokenizer creates
    lightweight learnable token sequences for modality-level attention.
    """

    def __init__(self, dim, num_tokens=4, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(dim, dim * num_tokens)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        bsz, dim = x.shape
        x = self.proj(x).view(bsz, self.num_tokens, dim)
        x = x + self.pos
        return self.drop(x)


class SourceStyleMODAAttention(nn.Module):
    """
    MODA-source-style key-state alignment attention.

    Main retained details:
      - visual / text-like token split through vflag / tflag
      - key normalization with +1e-6
      - Gram coordinate mapping: gram = key.T @ value
      - residual aligned key fusion with learnable ratio initialized as 0
      - pfuser_v / pfuser_t after aligned-key residual
      - lightweight modular modality bias initialized as 0

    This is adapted for non-causal classification, so RoPE and causal masks
    from LLM attention are intentionally omitted.
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        dropout=0.1,
        use_modality_bias=True,
        identity_init_pfuser=True,
    ):
        super().__init__()

        if dim % num_heads != 0:
            valid_heads = [h for h in [8, 4, 2, 1] if dim % h == 0]
            num_heads = valid_heads[0]

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.phi = nn.Identity()

        # MODA-style residual alignment ratios.
        # Initialized to 0 so this starts as almost standard attention.
        self.ratio_v = nn.Parameter(torch.zeros(1, 1, self.head_dim))
        self.ratio_t = nn.Parameter(torch.zeros(1, 1, self.head_dim))

        self.pfuser_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.pfuser_t = nn.Linear(self.head_dim, self.head_dim, bias=False)

        if identity_init_pfuser:
            nn.init.eye_(self.pfuser_v.weight)
            nn.init.eye_(self.pfuser_t.weight)

            # Prevent classifier_hateClipper._init_weights from overwriting this.
            self.pfuser_v._skip_reinit = True
            self.pfuser_t._skip_reinit = True

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.use_modality_bias = use_modality_bias
        if use_modality_bias:
            # 0 = visual, 1 = semantic/text-like.
            # This is a compact version of MODA's modular mask.
            self.modality_bias = nn.Parameter(torch.zeros(2, 2))

    def _shape(self, x):
        bsz, seq_len, dim = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()  # [B, H, N, Dh]

    @staticmethod
    def _norm_key(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _norm_gram(g):
        return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _apply_source_style_key_alignment(self, key_states, value_states, vflag, tflag):
        """
        key_states/value_states: [B, H, N, Dh]
        vflag/tflag:            [B, N], bool
        """
        bsz = key_states.size(0)

        # [B, H, N, Dh] -> [B, N, H, Dh]
        key_tok = key_states.transpose(1, 2).contiguous()
        value_tok = value_states.transpose(1, 2).contiguous()

        new_key_tok = key_tok.clone()

        for i in range(bsz):
            idx_v = torch.where(vflag[i])[0]
            idx_t = torch.where(tflag[i])[0]

            if idx_v.numel() == 0 or idx_t.numel() == 0:
                continue

            key_v = key_tok[i, idx_v].unsqueeze(0)       # [1, Nv, H, Dh]
            key_t = key_tok[i, idx_t].unsqueeze(0)       # [1, Nt, H, Dh]
            value_v = value_tok[i, idx_v].unsqueeze(0)   # [1, Nv, H, Dh]
            value_t = value_tok[i, idx_t].unsqueeze(0)   # [1, Nt, H, Dh]

            res_v = key_v
            res_t = key_t

            # Convert to [1, H, N, Dh], matching MODA-style attention internals.
            key_v_h = self.phi(key_v.permute(0, 2, 1, 3)) + 1e-6
            key_t_h = self.phi(key_t.permute(0, 2, 1, 3)) + 1e-6
            value_v_h = value_v.permute(0, 2, 1, 3)
            value_t_h = value_t.permute(0, 2, 1, 3)

            key_v_h = self._norm_key(key_v_h)
            key_t_h = self._norm_key(key_t_h)

            # Text/context coordinate basis guides visual keys.
            gram_t = key_t_h.transpose(-2, -1) @ value_t_h     # [1, H, Dh, Dh]
            gram_t = self._norm_gram(gram_t)

            aligned_v = key_v_h @ gram_t                       # [1, H, Nv, Dh]
            aligned_v = aligned_v.permute(0, 2, 1, 3)          # [1, Nv, H, Dh]

            fused_v = res_v + self.ratio_v * aligned_v
            fused_v = self.pfuser_v(fused_v)
            new_key_tok[i, idx_v] = fused_v.squeeze(0)

            # Visual coordinate basis guides text/context keys.
            # This keeps the "duplex" behavior instead of only text->visual.
            gram_v = key_v_h.transpose(-2, -1) @ value_v_h     # [1, H, Dh, Dh]
            gram_v = self._norm_gram(gram_v)

            aligned_t = key_t_h @ gram_v                       # [1, H, Nt, Dh]
            aligned_t = aligned_t.permute(0, 2, 1, 3)          # [1, Nt, H, Dh]

            fused_t = res_t + self.ratio_t * aligned_t
            fused_t = self.pfuser_t(fused_t)
            new_key_tok[i, idx_t] = fused_t.squeeze(0)

        return new_key_tok.transpose(1, 2).contiguous()        # [B, H, N, Dh]

    def _build_modality_bias(self, attn_scores, vflag, tflag):
        if not self.use_modality_bias:
            return attn_scores

        bsz, _, seq_len, _ = attn_scores.shape
        bias_list = []

        for i in range(bsz):
            mod = torch.zeros(seq_len, dtype=torch.long, device=attn_scores.device)
            mod[tflag[i]] = 1
            bias = self.modality_bias[mod[:, None], mod[None, :]]
            bias_list.append(bias)

        bias = torch.stack(bias_list, dim=0)  # [B, N, N]
        return attn_scores + bias.unsqueeze(1)

    def forward(self, x, vflag, tflag):
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        k = self._apply_source_style_key_alignment(k, v, vflag, tflag)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self._build_modality_bias(attn, vflag, tflag)

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.dim)

        out = self.o_proj(out)
        out = self.proj_drop(out)
        return out


class SourceStyleMODABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        dropout=0.1,
        ff_mult=4,
        identity_init_pfuser=True,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SourceStyleMODAAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            identity_init_pfuser=identity_init_pfuser,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

        self.attn_alpha = nn.Parameter(torch.tensor(0.1))
        self.ffn_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, vflag, tflag):
        x = x + torch.tanh(self.attn_alpha) * self.attn(self.norm1(x), vflag, tflag)
        x = x + torch.tanh(self.ffn_alpha) * self.ffn(self.norm2(x))
        return x


class SourceStyleSalM2CMA1D(nn.Module):
    """
    1D vector-token version of SalM2-style CrossModelAtt.

    Retained detail:
      perception = max(perception) - perception
      output = feature + gamma * perception_info

    gamma is initialized to 0, matching the residual-safe behavior.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.semantic_proj = nn.Linear(dim, dim)
        self.feature_proj = nn.Linear(dim, dim)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, semantic_tokens, feature_tokens):
        """
        semantic_tokens: [B, Ns, D]
        feature_tokens:  [B, Nf, D]
        """
        semantic_tokens = self.semantic_proj(semantic_tokens)
        feature_value = self.feature_proj(feature_tokens)

        q = semantic_tokens.transpose(1, 2)       # [B, D, Ns]
        k = semantic_tokens                       # [B, Ns, D]

        perception = torch.bmm(q.float(), k.float())  # [B, D, D]

        # SalM2-style inverse perception matrix.
        perception = perception.amax(dim=-1, keepdim=True).expand_as(perception) - perception

        v = feature_value.transpose(1, 2).float()     # [B, D, Nf]
        perception_info = torch.bmm(perception, v)    # [B, D, Nf]
        perception_info = perception_info.transpose(1, 2).to(feature_tokens.dtype)

        out = feature_tokens + self.gamma * self.drop(perception_info)
        return self.norm(out)


class CLIPAwareTinyTransformerFusion(nn.Module):
    """
    Fusion for CLIP-generated pooled embeddings.

    Token layout:
      image token
      text token
      context token
      |image-text| token
      image*text token

    This avoids pseudo-token expansion and preserves the CLIP embedding geometry.
    """

    def __init__(
        self,
        map_dim,
        base_dim,
        num_heads=4,
        dropout=0.1,
        ff_mult=2,
    ):
        super().__init__()

        if map_dim % num_heads != 0:
            valid_heads = [h for h in [8, 4, 2, 1] if map_dim % h == 0]
            num_heads = valid_heads[0]

        self.type_embed = nn.Parameter(torch.zeros(1, 5, map_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=map_dim,
            nhead=num_heads,
            dim_feedforward=map_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(map_dim * 5),
            nn.Linear(map_dim * 5, base_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_dim, base_dim),
            nn.LayerNorm(base_dim),
        )

        # Very small initial residual influence.
        # 0.1 * sigmoid(-4) ≈ 0.0018
        self.gate = nn.Parameter(torch.tensor(-4.0))

        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, v_img, v_txt, h_ctx):
        # Keep CLIP geometry more stable through feature normalization.
        v_img_n = F.normalize(v_img, dim=-1)
        v_txt_n = F.normalize(v_txt, dim=-1)
        h_ctx_n = F.normalize(h_ctx, dim=-1)

        diff = torch.abs(v_img_n - v_txt_n)
        prod = v_img_n * v_txt_n

        tokens = torch.stack(
            [
                v_img_n,
                v_txt_n,
                h_ctx_n,
                diff,
                prod,
            ],
            dim=1,
        )

        tokens = tokens + self.type_embed
        tokens = self.encoder(tokens)

        x = tokens.flatten(start_dim=1)
        x = self.out_proj(x)

        gate = 0.1 * torch.sigmoid(self.gate)
        return gate * x

class SourceStyleSalM2MODAFusion(nn.Module):
    """
    SalM2 + MODA source-style fusion module.

    Input:
      v_img: [B, map_dim]
      v_txt: [B, map_dim]
      h_ctx: [B, map_dim]

    Output:
      x_moda: [B, base_dim]
    """

    def __init__(
        self,
        map_dim,
        base_dim,
        num_layers=1,
        num_heads=4,
        num_tokens=4,
        dropout=0.1,
        identity_init_pfuser=True,
    ):
        super().__init__()

        self.map_dim = map_dim
        self.base_dim = base_dim
        self.num_tokens = num_tokens

        self.img_tok = SourceStyleVectorTokenizer(map_dim, num_tokens, dropout)
        self.txt_tok = SourceStyleVectorTokenizer(map_dim, num_tokens, dropout)
        self.ctx_tok = SourceStyleVectorTokenizer(map_dim, num_tokens, dropout)

        self.diff_proj = nn.Sequential(
            nn.Linear(map_dim, map_dim),
            nn.LayerNorm(map_dim),
            nn.GELU(),
        )
        self.prod_proj = nn.Sequential(
            nn.Linear(map_dim, map_dim),
            nn.LayerNorm(map_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [
                SourceStyleMODABlock(
                    dim=map_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    identity_init_pfuser=identity_init_pfuser,
                )
                for _ in range(num_layers)
            ]
        )

        self.salm2_cma = SourceStyleSalM2CMA1D(map_dim, dropout)
        self.final_norm = nn.LayerNorm(map_dim)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(map_dim * 4),
            nn.Linear(map_dim * 4, base_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_dim, base_dim),
            nn.LayerNorm(base_dim),
        )

    def forward(self, v_img, v_txt, h_ctx):
        bsz = v_img.size(0)
        t = self.num_tokens

        img_tokens = self.img_tok(v_img)
        txt_tokens = self.txt_tok(v_txt)
        ctx_tokens = self.ctx_tok(h_ctx)

        diff_token = self.diff_proj(torch.abs(v_img - v_txt)).unsqueeze(1)
        prod_token = self.prod_proj(v_img * v_txt).unsqueeze(1)
        rel_tokens = torch.cat([diff_token, prod_token], dim=1)

        tokens = torch.cat(
            [
                img_tokens,
                txt_tokens,
                ctx_tokens,
                rel_tokens,
            ],
            dim=1,
        )

        seq_len = tokens.size(1)

        # MODA-style token flags.
        # Visual: image tokens.
        # Text-like: text + context + relation tokens.
        vflag = torch.zeros(bsz, seq_len, dtype=torch.bool, device=tokens.device)
        tflag = torch.zeros(bsz, seq_len, dtype=torch.bool, device=tokens.device)

        vflag[:, :t] = True
        tflag[:, t:] = True

        for block in self.blocks:
            tokens = block(tokens, vflag, tflag)

        # SalM2-style semantic guidance.
        # Use original top-down tokens as semantic guidance.
        semantic_tokens = torch.cat([txt_tokens, ctx_tokens], dim=1)
        tokens = self.salm2_cma(
            semantic_tokens=semantic_tokens,
            feature_tokens=tokens,
        )

        tokens = self.final_norm(tokens)

        img_pool = tokens[:, :t, :].mean(dim=1)
        txt_pool = tokens[:, t: 2 * t, :].mean(dim=1)
        ctx_pool = tokens[:, 2 * t: 3 * t, :].mean(dim=1)
        rel_pool = tokens[:, 3 * t:, :].mean(dim=1)

        out = torch.cat([img_pool, txt_pool, ctx_pool, rel_pool], dim=1)
        out = self.out_proj(out)

        return out



class classifier_hateClipper(nn.Module):
    """
    Pruned classifier_hateClipper_llm with source-style SalM2 + MODA Transformer fusion.

    Kept compatibility:
      - same constructor signature
      - binary output
      - get_aux_loss(labels)
      - forward(img_feats, text_feats, context_feats, return_embed=False)
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
        super(classifier_hateClipper, self).__init__()

        if dropout is None:
            dropout = [0.1, 0.1, 0.1]
        if len(dropout) != 3:
            raise ValueError("dropout must contain three values: [input_drop, fusion_drop, head_drop]")
        if batch_norm:
            raise ValueError("This pruned classifier assumes --batch_norm False.")

        # Your original code checked Toxicn_mm while the comment says FB.
        # To avoid breaking existing training scripts, allow both.
        if args is not None and hasattr(args, "dataset"):
            if args.dataset not in {"FB", "Toxicn_mm"}:
                raise ValueError("This pruned classifier is specialized for binary FB/Toxicn_mm-style setup.")

        self.map_dim = map_dim
        self.output_dim = 1

        # ========= 1. Base projections =========
        self.img_proj = nn.Linear(image_dim, map_dim)
        self.text_proj = nn.Linear(text_dim, map_dim)
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

        base_dim = map_dim * 2
        gate_in_dim = map_dim * 3

        self.base_ln = nn.LayerNorm(base_dim)

        self.exp_gate = nn.Sequential(
            nn.Linear(gate_in_dim, map_dim),
            nn.Sigmoid(),
        )

        # ========= 3. Source-style SalM2 + MODA Transformer fusion =========
        tf_layers = getattr(args, "tf_layers", 1) if args is not None else 1
        tf_heads = getattr(args, "tf_heads", 4) if args is not None else 4
        tf_tokens = getattr(args, "tf_tokens", 4) if args is not None else 4
        tf_dropout = getattr(args, "tf_dropout", dropout[1]) if args is not None else dropout[1]

        self.source_style_moda_fusion = SourceStyleSalM2MODAFusion(
            map_dim=map_dim,
            base_dim=base_dim,
            num_layers=tf_layers,
            num_heads=tf_heads,
            num_tokens=tf_tokens,
            dropout=tf_dropout,
            identity_init_pfuser=True,
        )

        self.moda_alpha = nn.Parameter(torch.tensor(0.05))
        self.moda_ln = nn.LayerNorm(base_dim)

        # tf_heads = getattr(args, "tf_heads", 4) if args is not None else 4
        # tf_dropout = getattr(args, "tf_dropout", dropout[1]) if args is not None else dropout[1]
        #
        # self.clip_fusion = CLIPAwareTinyTransformerFusion(
        #     map_dim=map_dim,
        #     base_dim=base_dim,
        #     num_heads=tf_heads,
        #     dropout=tf_dropout,
        # )

        # self.clip_fusion_ln = nn.LayerNorm(base_dim)

        # ========= 4. FiLM branch =========
        self.film_generator = nn.Linear(map_dim, base_dim * 2)
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

        # Do not initialize to 0; otherwise this branch can learn too slowly.
        self.film_alpha = nn.Parameter(torch.tensor(0.1))

        # ========= 5. Semantic residual branch =========
        self.sem_res_proj = nn.Linear(map_dim, base_dim)
        nn.init.zeros_(self.sem_res_proj.weight)
        nn.init.zeros_(self.sem_res_proj.bias)

        self.sem_gate = nn.Sequential(
            nn.Linear(gate_in_dim, base_dim),
            nn.Sigmoid(),
        )

        # Do not initialize to 0; otherwise this branch can learn too slowly.
        self.sem_alpha = nn.Parameter(torch.tensor(0.1))

        # ========= 6. Layer combination =========
        self.fusion_ln = nn.LayerNorm(base_dim)

        # After adding MODA fusion, x_base is already strong.
        # Start by trusting x_base more.
        self.layer_combine_logits = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float)
        )

        self.layer_combine_ln = nn.LayerNorm(base_dim)

        # ========= 7. Classification head =========
        if num_layers <= 0:
            self.mlp = nn.Identity()
            final_dim = base_dim
        else:
            layers = []
            cur_dim = base_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(cur_dim, proj_dim))
                layers.append(nn.LayerNorm(proj_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout[2]))
                cur_dim = proj_dim
            self.mlp = nn.Sequential(*layers)
            final_dim = proj_dim

        self.final_dim = final_dim
        self.head_drop = nn.Dropout(dropout[1])
        self.output_layer = nn.Linear(final_dim, self.output_dim)

        # ========= 8. Attribution-guided counterfactual module =========
        self.cf_topk_ratio = getattr(args, "cf_topk_ratio", 0.12) if args is not None else 0.12
        self.cf_mask_value = getattr(args, "cf_mask_value", 0.05) if args is not None else 0.05
        self.cf_fuse_alpha = nn.Parameter(torch.tensor(0.0))
        self.cf_margin = getattr(args, "cf_margin", 0.23) if args is not None else 0.23
        self.cf_neg_margin = getattr(args, "cf_neg_margin", 0.05) if args is not None else 0.05

        # Important:
        # Counterfactual aux_loss and counterfactual logit fusion are decoupled.
        # Default False is usually more stable for ACC.
        self.use_cf_fusion = getattr(args, "use_cf_fusion", False) if args is not None else False
        self.cf_fuse_scale = getattr(args, "cf_fuse_scale", 0.1) if args is not None else 0.1

        self._cached_linear_output = None
        self._cached_cf_output = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "_skip_reinit", False):
                    continue

                # These two are intentionally zero-initialized above.
                if m is self.film_generator or m is self.sem_res_proj:
                    continue

                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def _build_counterfactual_embed(self, embed, linear_output):
        """Build attribution-guided counterfactual embeddings for binary classification."""
        del linear_output

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
        Auxiliary counterfactual loss.

        Positive samples:
            masking important evidence should lower the positive logit.
        Negative samples:
            masking important dimensions should not raise the positive logit too much.
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

        # ========= 2. Context adapter and gate =========
        h_ctx = self.exp_adapter(v_ctx)

        ctx_gate_in = torch.cat((v_img, v_txt, h_ctx), dim=1)
        ctx_gate = self.exp_gate(ctx_gate_in)
        h_ctx = h_ctx * ctx_gate

        # ========= 3. Image-text base =========
        x_base = torch.cat((v_img, v_txt), dim=1)
        x_base = self.base_ln(x_base)

        # ========= 4. Source-style SalM2 + MODA Transformer fusion =========
        x_moda = self.source_style_moda_fusion(v_img, v_txt, h_ctx)

        x_base = self.moda_ln(
            x_base + torch.tanh(self.moda_alpha) * x_moda
        )

        # x_clip = self.clip_fusion(v_img, v_txt, h_ctx)
        #
        # x_base = self.clip_fusion_ln(
        #     x_base + x_clip
        # )

        # ========= 5. FiLM =========
        film_params = self.film_generator(h_ctx)
        gamma, beta = torch.chunk(film_params, 2, dim=1)

        gamma = 0.05 * torch.tanh(gamma)
        beta = 0.05 * torch.tanh(beta)
        film_delta = gamma * x_base + beta

        x_film = x_base + torch.tanh(self.film_alpha) * film_delta

        # ========= 6. Semantic residual =========
        sem_res = 0.05 * torch.tanh(self.sem_res_proj(h_ctx))

        sem_gate_in = torch.cat((v_img, v_txt, h_ctx), dim=1)
        sem_gate = self.sem_gate(sem_gate_in)

        x_sem = x_film + torch.tanh(self.sem_alpha) * (sem_gate * sem_res)

        # ========= 7. Layer combination =========
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

        # ========= 8. Main classifier =========
        x = self.head_drop(x)
        embed = self.mlp(x)
        linear_output = self.output_layer(embed)

        # ========= 9. Counterfactual branch =========
        embed_cf = self._build_counterfactual_embed(embed, linear_output)
        cf_output = self.output_layer(embed_cf)

        self._cached_linear_output = linear_output
        self._cached_cf_output = cf_output

        if self.use_cf_fusion:
            cf_gap = linear_output - cf_output
            output = linear_output + self.cf_fuse_scale * torch.tanh(self.cf_fuse_alpha) * cf_gap
        else:
            output = linear_output

        if return_embed:
            return output, embed

        return output