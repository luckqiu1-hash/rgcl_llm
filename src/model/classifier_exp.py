import torch
import torch.nn as nn
import torch.nn.functional as F

from model.classifier import CosineBinaryClassifier, ResidualClassificationHead


class classifier_hateClipper(nn.Module):
    """
    Ablation without LLM explanation fusion.

    The model keeps the same public interface as classifier.py, but ignores
    context_feats in forward so run_rac.py can switch imports directly.
    Counterfactual training is kept for the isolated explanation-fusion ablation.
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
        del exp_dim, fusion_mode

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

        self.img_norm = nn.LayerNorm(map_dim)
        self.text_norm = nn.LayerNorm(map_dim)

        self.img_drop = nn.Dropout(dropout[0])
        self.text_drop = nn.Dropout(dropout[0])

        base_dim = map_dim * 2
        self.base_ln = nn.LayerNorm(base_dim)

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

        self.cf_topk_ratio = getattr(args, "cf_topk_ratio", 0.12) if args is not None else 0.12
        self.cf_mask_value = getattr(args, "cf_mask_value", 0.05) if args is not None else 0.05
        self.cf_fuse_alpha = nn.Parameter(torch.tensor(0.0))
        self.cf_margin = getattr(args, "cf_margin", 0.23) if args is not None else 0.23
        self.cf_neg_margin = getattr(args, "cf_neg_margin", 0.05) if args is not None else 0.05
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if hasattr(self.output_layer, "weight"):
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        if hasattr(self.output_layer, "bias") and self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def _build_counterfactual_embed(self, embed, linear_output):
        del linear_output

        with torch.no_grad():
            cls_w = self.output_layer.weight[0].abs().unsqueeze(0)
            importance = embed.abs() * cls_w

            k = max(1, int(embed.size(1) * self.cf_topk_ratio))
            topk_idx = importance.topk(k=k, dim=1, largest=True, sorted=False).indices

            mask = torch.ones_like(embed)
            mask.scatter_(1, topk_idx, self.cf_mask_value)

        return embed * mask

    def get_aux_loss(self, labels):
        if self._cached_linear_output is None or self._cached_cf_output is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        y = labels.float().view(-1, 1)
        orig = self._cached_linear_output
        cf = self._cached_cf_output

        pos_loss = F.relu(self.cf_margin - (orig - cf))
        neg_loss = F.relu((cf - orig) - self.cf_neg_margin)

        return (y * pos_loss + (1.0 - y) * neg_loss).mean()

    def forward(self, img_feats, text_feats, context_feats, return_embed=False):
        del context_feats

        v_img = self.img_proj(img_feats)
        v_txt = self.text_proj(text_feats)

        v_img = self.img_norm(v_img)
        v_txt = self.text_norm(v_txt)

        v_img = self.img_drop(v_img)
        v_txt = self.text_drop(v_txt)

        x = torch.cat((v_img, v_txt), dim=1)
        x = self.base_ln(x)

        x = self.head_drop(x)
        embed = self.mlp(x)
        linear_output = self.output_layer(embed)

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
