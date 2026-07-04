import argparse
import csv
import importlib
import os
import sys
from collections import OrderedDict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
for module_path in (CURRENT_DIR, ROOT_DIR):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)


def str2bool(value):
    return str(value).lower() == "true"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate how models react when top-k toxic evidence dimensions are weakened."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "Model specs in name:module:checkpoint format. Example: "
            "full:model.classifier:E:/.../best_model.pt no_cf:model.classifier_cf:E:/.../best_model.pt"
        ),
    )
    parser.add_argument("--path", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="Toxicn_mm")
    parser.add_argument("--model", type=str, default="clip-vit-large-patch14-336_HF")
    parser.add_argument("--split", type=str, default="test_seen", choices=["dev", "test_seen", "test_unseen"])
    parser.add_argument("--output_dir", type=str, default="./src/cf_evidence_behavior")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--cf_topk_ratio", type=float, default=0.12)
    parser.add_argument("--cf_mask_value", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--fusion_mode", type=str, default="concat")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--map_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, nargs=3, default=[0.1, 0.4, 0.2])
    parser.add_argument("--batch_norm", type=str2bool, default=False)
    parser.add_argument("--tf_layers", type=int, default=1)
    parser.add_argument("--tf_heads", type=int, default=4)
    parser.add_argument("--tf_tokens", type=int, default=4)
    parser.add_argument("--tf_dropout", type=float, default=None)
    parser.add_argument("--head_scale", type=float, default=16.0)
    parser.add_argument("--strict_load", type=str2bool, default=True)
    return parser.parse_args()


def parse_model_spec(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid model spec: {spec}. Expected name:module:checkpoint"
        )
    name, module_name, checkpoint = parts
    return name, module_name, checkpoint


def load_checkpoint_state(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        state_dict[key] = value
    return state_dict


def safe_device(requested_device):
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, use CPU instead.")
        return torch.device("cpu")
    return torch.device(requested_device)


def load_eval_dataloader(args):
    if args.dataset == "FB":
        train, dev, test_seen, test_unseen = load_feats_from_CLIP(
            os.path.join(args.path, "CLIP_Embedding"), "FB", args.model
        )
        datasets = {"dev": dev, "test_seen": test_seen, "test_unseen": test_unseen}
    else:
        if args.split == "test_unseen":
            raise ValueError("--split test_unseen is only available for FB dataset.")
        train, dev, test_seen = load_feats_from_CLIP(
            os.path.join(args.path, "CLIP_Embedding"), args.dataset, args.model
        )
        datasets = {"dev": dev, "test_seen": test_seen}

    eval_dl = CLIP2Dataloader(
        datasets[args.split],
        batch_size=args.batch_size,
        return_dataset=False,
        normalize=False,
    )[0]
    return eval_dl


def build_model(args, model_cls, eval_dl, state_dict):
    first_batch = next(iter(eval_dl))
    image_feat_dim = first_batch["image_feats"].shape[1]
    text_feat_dim = first_batch["text_feats"].shape[1]
    exp_feat_dim = first_batch["exp_feats"].shape[1]

    if "img_proj.weight" in state_dict:
        image_feat_dim = state_dict["img_proj.weight"].shape[1]
        args.map_dim = state_dict["img_proj.weight"].shape[0]
    if "text_proj.weight" in state_dict:
        text_feat_dim = state_dict["text_proj.weight"].shape[1]
    if "exp_proj.weight" in state_dict:
        exp_feat_dim = state_dict["exp_proj.weight"].shape[1]

    return model_cls(
        image_dim=image_feat_dim,
        text_dim=text_feat_dim,
        exp_dim=exp_feat_dim,
        num_layers=args.num_layers,
        proj_dim=args.proj_dim,
        map_dim=args.map_dim,
        fusion_mode=args.fusion_mode,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        args=args,
    )


def sigmoid_tensor(logits):
    return torch.sigmoid(logits.view(-1))


def counterfactual_logits(model, embed, topk_ratio, mask_value):
    output_layer = model.output_layer
    if not hasattr(output_layer, "weight"):
        raise AttributeError("The model output_layer must expose a weight parameter.")

    with torch.no_grad():
        cls_w = output_layer.weight[0].abs().unsqueeze(0)
        importance = embed.abs() * cls_w
        k = max(1, int(embed.size(1) * topk_ratio))
        topk_idx = importance.topk(k=k, dim=1, largest=True, sorted=False).indices

        mask = torch.ones_like(embed)
        mask.scatter_(1, topk_idx, mask_value)
        embed_cf = embed * mask
        cf_logits = output_layer(embed_cf)

    return cf_logits, topk_idx


def evaluate_model(model_name, module_name, checkpoint, args, eval_dl, device):
    module = importlib.import_module(module_name)
    model_cls = getattr(module, "classifier_hateClipper")
    state_dict = load_checkpoint_state(checkpoint, device)
    model = build_model(args, model_cls, eval_dl, state_dict)
    load_result = model.load_state_dict(state_dict, strict=args.strict_load)
    if not args.strict_load:
        print(f"[{model_name}] load_state_dict result: {load_result}")
    model.to(device)
    model.eval()

    sample_rows = []
    with torch.no_grad():
        for batch in tqdm(eval_dl, desc=f"Evaluate {model_name}"):
            ids = [str(item) for item in batch["ids"]]
            labels = batch["labels"].to(device).long().view(-1)
            logits, embed = model(
                batch["image_feats"].to(device),
                batch["text_feats"].to(device),
                batch["exp_feats"].to(device),
                return_embed=True,
            )
            cf_logits, topk_idx = counterfactual_logits(
                model,
                embed,
                args.cf_topk_ratio,
                args.cf_mask_value,
            )

            orig_prob = sigmoid_tensor(logits)
            cf_prob = sigmoid_tensor(cf_logits)
            orig_pred = (orig_prob >= args.decision_threshold).long()
            cf_pred = (cf_prob >= args.decision_threshold).long()

            for i, sample_id in enumerate(ids):
                label = int(labels[i].item())
                op = float(orig_prob[i].detach().cpu().item())
                cp = float(cf_prob[i].detach().cpu().item())
                sample_rows.append(
                    {
                        "model": model_name,
                        "id": sample_id,
                        "label": label,
                        "orig_prob": op,
                        "cf_prob": cp,
                        "delta_orig_minus_cf": op - cp,
                        "orig_pred": int(orig_pred[i].item()),
                        "cf_pred": int(cf_pred[i].item()),
                        "flipped": int(orig_pred[i].item() != cf_pred[i].item()),
                        "topk_dims": " ".join(str(int(x)) for x in topk_idx[i].detach().cpu().tolist()),
                    }
                )

    return sample_rows


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def summarize_model(rows):
    labels = [row["label"] for row in rows]
    orig_preds = [row["orig_pred"] for row in rows]
    cf_preds = [row["cf_pred"] for row in rows]
    toxic_rows = [row for row in rows if row["label"] == 1]
    nontoxic_rows = [row for row in rows if row["label"] == 0]

    toxic_drops = [row["delta_orig_minus_cf"] for row in toxic_rows]
    nontoxic_increases = [row["cf_prob"] - row["orig_prob"] for row in nontoxic_rows]

    toxic_flip_down = [
        row for row in toxic_rows
        if row["orig_pred"] == 1 and row["cf_pred"] == 0
    ]
    nontoxic_flip_up = [
        row for row in nontoxic_rows
        if row["orig_pred"] == 0 and row["cf_pred"] == 1
    ]

    return {
        "model": rows[0]["model"] if rows else "",
        "total": len(rows),
        "toxic_total": len(toxic_rows),
        "nontoxic_total": len(nontoxic_rows),
        "orig_acc": mean_or_zero([int(y == p) for y, p in zip(labels, orig_preds)]),
        "cf_acc": mean_or_zero([int(y == p) for y, p in zip(labels, cf_preds)]),
        "prediction_consistency": mean_or_zero([int(o == c) for o, c in zip(orig_preds, cf_preds)]),
        "toxic_prob_drop_mean": mean_or_zero(toxic_drops),
        "toxic_prob_drop_rate": mean_or_zero([int(value > 0) for value in toxic_drops]),
        "toxic_to_nontoxic_flip_rate": len(toxic_flip_down) / len(toxic_rows) if toxic_rows else 0.0,
        "nontoxic_prob_increase_mean": mean_or_zero(nontoxic_increases),
        "nontoxic_prob_increase_rate": mean_or_zero([int(value > 0) for value in nontoxic_increases]),
        "nontoxic_to_toxic_flip_rate": len(nontoxic_flip_up) / len(nontoxic_rows) if nontoxic_rows else 0.0,
    }


def save_sample_rows(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "model",
        "id",
        "label",
        "orig_prob",
        "cf_prob",
        "delta_orig_minus_cf",
        "orig_pred",
        "cf_pred",
        "flipped",
        "topk_dims",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary_rows(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "model",
        "checkpoint",
        "total",
        "toxic_total",
        "nontoxic_total",
        "orig_acc",
        "cf_acc",
        "prediction_consistency",
        "toxic_prob_drop_mean",
        "toxic_prob_drop_rate",
        "toxic_to_nontoxic_flip_rate",
        "nontoxic_prob_increase_mean",
        "nontoxic_prob_increase_rate",
        "nontoxic_to_toxic_flip_rate",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    global torch
    global tqdm
    global load_feats_from_CLIP, CLIP2Dataloader

    import torch
    from tqdm import tqdm

    from data_loader.dataset import load_feats_from_CLIP
    from data_loader.rac_dataloader import CLIP2Dataloader

    if args.tf_dropout is None:
        args.tf_dropout = args.dropout[1]

    device = safe_device(args.device)
    args.device = str(device)
    eval_dl = load_eval_dataloader(args)

    all_sample_rows = []
    summary_rows = []
    for spec in args.models:
        model_name, module_name, checkpoint = parse_model_spec(spec)
        rows = evaluate_model(model_name, module_name, checkpoint, args, eval_dl, device)
        all_sample_rows.extend(rows)
        summary = summarize_model(rows)
        summary["checkpoint"] = os.path.abspath(checkpoint)
        summary_rows.append(summary)

    sample_path = os.path.join(args.output_dir, "cf_evidence_per_sample.csv")
    summary_path = os.path.join(args.output_dir, "cf_evidence_summary.csv")
    save_sample_rows(all_sample_rows, sample_path)
    save_summary_rows(summary_rows, summary_path)

    print("Counterfactual evidence behavior evaluation finished.")
    for row in summary_rows:
        print(
            "{model}: orig_acc={orig_acc:.6f}, cf_acc={cf_acc:.6f}, "
            "toxic_drop={toxic_prob_drop_mean:.6f}, nontoxic_increase={nontoxic_prob_increase_mean:.6f}, "
            "nontoxic_flip_up={nontoxic_to_toxic_flip_rate:.6f}".format(**row)
        )
    print("Saved summary to:", os.path.abspath(summary_path))
    print("Saved per-sample details to:", os.path.abspath(sample_path))


if __name__ == "__main__":
    main()
