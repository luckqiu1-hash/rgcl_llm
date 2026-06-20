import torch
import torch.nn as nn
from utils.retrieval import dense_retrieve_hard_negatives_pseudo_positive


EXPECTED_ARGS = {
    "metric": "cos",
    "loss": "triplet",
    "hard_negatives_loss": True,
    "no_hard_negatives": 1,
    "no_pseudo_gold_positives": 1,
    "hybrid_loss": True,
}


def _coerce_like(value, expected):
    """Make validation tolerant to argparse configs that keep bool/int values as strings."""
    if isinstance(expected, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)
    if isinstance(expected, int) and not isinstance(expected, bool):
        return int(value)
    return value


def _validate_args(args):
    """Fail fast if this pruned loss is accidentally used with another setup."""
    mismatches = []
    for name, expected in EXPECTED_ARGS.items():
        actual = getattr(args, name, None)
        try:
            actual_cmp = _coerce_like(actual, expected)
        except (TypeError, ValueError):
            actual_cmp = actual
        if actual_cmp != expected:
            mismatches.append(f"{name}={actual!r}, expected {expected!r}")

    sparse_dictionary = getattr(args, "sparse_dictionary", None)
    if sparse_dictionary not in (None, "None", ""):
        mismatches.append("sparse_dictionary is not None; this pruned file only keeps dense retrieval")

    if mismatches:
        raise ValueError(
            "loss_v2_pruned_for_FB_triplet.py was specialized for your current training args. "
            "Unsupported args: " + "; ".join(mismatches)
        )


def _row_mean_nonzero(masked_values):
    """
    Match the original V4 behavior: average only non-zero entries in each row.
    Rows with no non-zero entries return 0 instead of producing NaN.
    """
    nonzero_mask = masked_values != 0
    row_count = nonzero_mask.sum(dim=1)
    row_sum = masked_values.sum(dim=1)

    out = torch.zeros(masked_values.size(0), device=masked_values.device, dtype=masked_values.dtype)
    valid_rows = row_count > 0
    out[valid_rows] = row_sum[valid_rows] / row_count[valid_rows]
    return out


def compute_loss(batch,
                 train_dl,
                 model,
                 args,
                 train_set=None,
                 sparse_retrieval_dictionary=None,
                 train_feats=None,
                 train_labels=None):
    """
    Pruned loss for this fixed training configuration:
      --metric cos
      --loss triplet
      --hard_negatives_loss True --no_hard_negatives 1
      --no_pseudo_gold_positives 1
      --hybrid_loss True
      dense retrieval only, no sparse retrieval branch.
    """
    _validate_args(args)

    batch_size = len(batch["ids"])
    image_feats = batch["image_feats"].to(args.device)
    text_feats = batch["text_feats"].to(args.device)
    context_feats = batch["exp_feats"].to(args.device)
    labels_raw = batch["labels"].to(args.device)
    labels = labels_raw.bool()

    model.train()
    output, feats = model(image_feats, text_feats, context_feats, return_embed=True)

    cf_aux_loss = torch.tensor(0.0, device=args.device)
    if hasattr(model, "get_aux_loss"):
        cf_aux_loss = model.get_aux_loss(labels_raw)

    # ------------------------------------------------------------------
    # 1) In-batch negative loss, cosine only.
    #    Because no_pseudo_gold_positives == 1, in-batch positives are not used.
    # ------------------------------------------------------------------
    feats_i = feats.unsqueeze(2).expand(batch_size, -1, batch_size)
    sim_matrix = nn.CosineSimilarity(dim=1, eps=1e-8)(feats_i, feats_i.transpose(0, 2))
    sim_matrix.fill_diagonal_(0)

    same_label = labels.unsqueeze(1) == labels.unsqueeze(0)
    negative_mask = ~same_label
    in_batch_negative_loss = sim_matrix * negative_mask.int()
    in_batch_loss = _row_mean_nonzero(in_batch_negative_loss)

    # ------------------------------------------------------------------
    # 2) Dense retrieval: hard negative + pseudo positive.
    #    This is the branch reached when no_pseudo_gold_positives > 0.
    # ------------------------------------------------------------------
    (
        hard_negative_features,
        hard_negative_scores,
        pseudo_positive_features,
        pseudo_positive_scores,
        train_feats,
        train_labels,
    ) = dense_retrieve_hard_negatives_pseudo_positive(
        train_dl,
        feats,
        labels,
        model,
        largest_retrieval=args.no_pseudo_gold_positives,
        args=args,
        train_feats=train_feats,
        train_labels=train_labels,
    )

    # ------------------------------------------------------------------
    # 3) Hard negative loss, cosine only.
    # ------------------------------------------------------------------
    feats_for_hard = feats.unsqueeze(1).expand(batch_size, args.no_hard_negatives, -1)
    hard_valid_mask = torch.sum(hard_negative_features, dim=2) != 0
    hard_loss = hard_valid_mask * nn.CosineSimilarity(dim=2, eps=1e-8)(
        feats_for_hard,
        hard_negative_features,
    )
    hard_loss = torch.sum(hard_loss, dim=1)

    # ------------------------------------------------------------------
    # 4) Pseudo-positive loss, cosine only.
    # ------------------------------------------------------------------
    feats_for_pseudo = feats.unsqueeze(1).expand(batch_size, args.no_pseudo_gold_positives, -1)
    pseudo_gold_loss = nn.CosineSimilarity(dim=2, eps=1e-8)(
        feats_for_pseudo,
        pseudo_positive_features,
    )
    pseudo_gold_loss = torch.mean(pseudo_gold_loss, dim=1)

    # ------------------------------------------------------------------
    # 5) Triplet objective only.
    # ------------------------------------------------------------------
    total_loss = torch.mean(torch.relu(
        in_batch_loss + hard_loss - pseudo_gold_loss + args.triplet_margin
    ))

    # ------------------------------------------------------------------
    # 6) Hybrid BCE classification loss only.
    # ------------------------------------------------------------------
    if args.pos_weight_value is not None:
        loss_fn_classifier = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([args.pos_weight_value], device=args.device)
        )
    else:
        loss_fn_classifier = nn.BCEWithLogitsLoss()

    loss_classifier = loss_fn_classifier(output, labels.float().reshape(-1, 1))
    total_loss = total_loss * (1 - args.ce_weight) + loss_classifier * args.ce_weight

    # Same auxiliary term as the aligned first-version loss.
    total_loss = total_loss + 0.01 * cf_aux_loss

    return (
        total_loss,
        torch.mean(in_batch_loss),
        torch.mean(hard_loss),
        torch.mean(pseudo_gold_loss),
        loss_classifier,
        train_feats,
        train_labels,
    )
