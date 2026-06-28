import torch
import torch.nn as nn
import torch.nn.functional as F

from model.classifier import (
    CosineBinaryClassifier,
    ResidualClassificationHead,
    SourceStyleSalM2MODAFusion,
)


class classifier_hateClipper(nn.Module):
    """
    Ablation without the counterfactual module.

    Keeps the same public interface as classifier.py:
      - same constructor signature
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
        del fusion_mode

        if dropout is None:
            dropout = [0.1, 0.1, 0.1]
        if len(dropout) != 3:
            raise ValueError("dropout must contain three values: [input_drop, fusion_drop, head_drop]")
        if batch_norm:
            raise ValueError("This pruned classifier assumes --batch_norm False.")

        if args is not None and hasattr(args, "dataset"):
            if args.dataset not in {"FB", "Toxicn_mm"}:
                raise ValueError("This pruned classifier is specialized for binary FB/Toxicn_mm-style setup.")

        self.map_dim = map_dim
        self.output_dim = 1

        self.img_proj = nn.Linear(image_dim, map_dim)
        self.text_proj = nn.Linear(text_dim, map_dim)
        self.exp_proj = nn.Linear(exp_dim, map_dim)

        self.img_norm = nn.LayerNorm(map_dim)
        self.text_norm = nn.LayerNorm(map_dim)
        self.exp_norm = nn.LayerNorm(map_dim)

        self.img_drop = nn.Dropout(dropout[0])
        self.text_drop = nn.Dropout(dropout[0])
        self.exp_drop = nn.Dropout(min(dropout[0], 0.05))

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

        self.film_generator = nn.Linear(map_dim, base_dim * 2)
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)
        self.film_alpha = nn.Parameter(torch.tensor(0.1))

        self.sem_res_proj = nn.Linear(map_dim, base_dim)
        nn.init.zeros_(self.sem_res_proj.weight)
        nn.init.zeros_(self.sem_res_proj.bias)

        self.sem_gate = nn.Sequential(
            nn.Linear(gate_in_dim, base_dim),
            nn.Sigmoid(),
        )
        self.sem_alpha = nn.Parameter(torch.tensor(0.1))

        self.fusion_ln = nn.LayerNorm(base_dim)
        self.layer_combine_logits = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float)
        )
        self.layer_combine_ln = nn.LayerNorm(base_dim)

        self.mlp = ResidualClassificationHead(
            in_dim=base_dim,
            proj_dim=proj_dim,
            num_layers=num_layers,
            dropout=dropout[2],
        )

        self.final_dim = self.mlp.out_dim
        self.head_drop = nn.Dropout(dropout[1])

        head_scale = getattr(args, "head_scale", 16.0) if args is not None else 16.0
        self.output_layer = CosineBinaryClassifier(
            dim=self.final_dim,
            scale=head_scale,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "_skip_reinit", False):
                    continue
                if m is self.film_generator or m is self.sem_res_proj:
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if hasattr(self.output_layer, "weight"):
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        if hasattr(self.output_layer, "bias") and self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def get_aux_loss(self, labels):
        del labels
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, img_feats, text_feats, context_feats, return_embed=False):
        v_img = self.img_proj(img_feats)
        v_txt = self.text_proj(text_feats)
        v_ctx = self.exp_proj(context_feats)

        v_img = self.img_norm(v_img)
        v_txt = self.text_norm(v_txt)
        v_ctx = self.exp_norm(v_ctx)

        v_img = self.img_drop(v_img)
        v_txt = self.text_drop(v_txt)
        v_ctx = self.exp_drop(v_ctx)

        h_ctx = self.exp_adapter(v_ctx)
        ctx_gate = self.exp_gate(torch.cat((v_img, v_txt, h_ctx), dim=1))
        h_ctx = h_ctx * ctx_gate

        x_base = torch.cat((v_img, v_txt), dim=1)
        x_base = self.base_ln(x_base)

        x_moda = self.source_style_moda_fusion(v_img, v_txt, h_ctx)
        x_base = self.moda_ln(x_base + torch.tanh(self.moda_alpha) * x_moda)

        film_params = self.film_generator(h_ctx)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = 0.05 * torch.tanh(gamma)
        beta = 0.05 * torch.tanh(beta)
        x_film = x_base + torch.tanh(self.film_alpha) * (gamma * x_base + beta)

        sem_res = 0.05 * torch.tanh(self.sem_res_proj(h_ctx))
        sem_gate = self.sem_gate(torch.cat((v_img, v_txt, h_ctx), dim=1))
        x_sem = x_film + torch.tanh(self.sem_alpha) * (sem_gate * sem_res)

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

        x = self.head_drop(x)
        embed = self.mlp(x)
        output = self.output_layer(embed)

        if return_embed:
            return output, embed
        return output
