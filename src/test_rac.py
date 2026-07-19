import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for module_path in (str(CURRENT_DIR), str(ROOT_DIR)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)


def str2bool(value):
    return str(value).lower() in {"true", "1", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a RAC checkpoint with Precision, Recall, F1, ACC and AUC."
    )

    # Data / checkpoint configs. Keep names close to run_rac.py for easy reuse.
    parser.add_argument("--path", type=str, default="./data/", help="Dataset root. CLIP_Embedding should be under it.")
    parser.add_argument("--dataset", type=str, default="Toxicn_mm")
    parser.add_argument("--model", type=str, default="clip-vit-large-patch14-336_HF")
    parser.add_argument("--ckpt_path", "--pth_path", dest="ckpt_path", type=str, default='E:\qxy\code\\rgcl_llm\src\log_toxicn_mm\Retrieval\Toxicn_mm\\best_model.pt',
                        help="Path to trained .pt/.pth checkpoint.")
    parser.add_argument("--split", type=str, default="test_seen", choices=["dev", "test_seen", "test_unseen"],
                        help="Split to evaluate.")
    parser.add_argument("--output_dir", type=str, default="./test_outputs",
                        help="Directory to save metrics and prediction details.")
    parser.add_argument("--save_prefix", type=str, default=None,
                        help="Output filename prefix. Default uses dataset, split and checkpoint name.")

    # RAC retrieval configs.
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--similarity_threshold", type=float, default=-1.0)
    parser.add_argument("--majority_voting", type=str, default="arithmetic", choices=["mean", "arithmetic"])
    parser.add_argument("--decision_threshold", type=float, default=None,
                        help="Classification threshold on retrieval probability. Default: best threshold on eval split.")
    parser.add_argument("--metric", type=str, default="cos", choices=["cos", "ip", "l2"])

    # Model configs copied from run_rac.py.
    parser.add_argument("--fusion_mode", type=str, default="align")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--map_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, nargs=3, default=[0.2, 0.4, 0.1])
    parser.add_argument("--batch_norm", type=str2bool, default=False)

    # Extra model options used by the current classifier.py.
    parser.add_argument("--tf_layers", type=int, default=1)
    parser.add_argument("--tf_heads", type=int, default=4)
    parser.add_argument("--tf_tokens", type=int, default=4)
    parser.add_argument("--tf_dropout", type=float, default=None)
    parser.add_argument("--head_scale", type=float, default=16.0)
    parser.add_argument("--cf_topk_ratio", type=float, default=0.12)
    parser.add_argument("--cf_mask_value", type=float, default=0.05)
    parser.add_argument("--cf_margin", type=float, default=0.23)
    parser.add_argument("--cf_neg_margin", type=float, default=0.05)
    parser.add_argument("--use_cf_fusion", type=str2bool, default=False)
    parser.add_argument("--cf_fuse_scale", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--Faiss_GPU", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    # Training-only arguments accepted for compatibility with run_rac.py command lines.
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--loss", type=str, default="triplet")
    parser.add_argument("--triplet_margin", type=float, default=0.1)
    parser.add_argument("--norm_feats_loss", type=str2bool, default=False)
    parser.add_argument("--l2_sqrt", type=str2bool, default=False)
    parser.add_argument("--hybrid_loss", type=str2bool, default=True)
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--pos_weight_value", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--lr_scheduler", type=str2bool, default=False)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--no_pseudo_gold_positives", type=int, default=1)
    parser.add_argument("--in_batch_loss", type=str2bool, default=True)
    parser.add_argument("--hard_negatives_loss", type=str2bool, default=True)
    parser.add_argument("--no_hard_negatives", type=int, default=1)
    parser.add_argument("--no_hard_positives", type=int, default=0)
    parser.add_argument("--hard_negatives_multiple", type=int, default=12)
    parser.add_argument("--reindex_every_step", type=str2bool, default=False)
    parser.add_argument("--sparse_dictionary", type=str, default=None)
    parser.add_argument("--use_attribute", type=str2bool, default=True)
    parser.add_argument("--sparse_topk", type=int, default=None)
    parser.add_argument("--eval_retrieval", type=str2bool, default=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--final_eval", type=str2bool, default=False)
    parser.add_argument("--exp_comment", type=str, default="")
    parser.add_argument("--group_name", type=str, default="RAC")
    parser.add_argument("--force", type=str2bool, default=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--output_log", type=str, default=None)

    return parser.parse_args()


def safe_device(requested_device):
    import torch

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        return torch.device("cpu")
    return torch.device(requested_device)


def load_checkpoint_state(ckpt_path, device):
    import torch

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        clean_key = key[len("module."):] if key.startswith("module.") else key
        state_dict[clean_key] = value
    return state_dict


def infer_feature_dim(state_dict, dataloader, key, batch_key):
    import torch

    first_batch = next(iter(dataloader))
    fallback_dim = first_batch[batch_key].shape[1]
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return weight.shape[1]
    return fallback_dim


def build_model(args, train_dl, state_dict):
    from model.classifier import classifier_hateClipper

    image_dim = infer_feature_dim(state_dict, train_dl, "img_proj.weight", "image_feats")
    text_dim = infer_feature_dim(state_dict, train_dl, "text_proj.weight", "text_feats")
    exp_dim = infer_feature_dim(state_dict, train_dl, "exp_proj.weight", "exp_feats")

    print(f"Image feature dimension: {image_dim}")
    print(f"Text feature dimension: {text_dim}")
    print(f"Explanation feature dimension: {exp_dim}")

    return classifier_hateClipper(
        image_dim=image_dim,
        text_dim=text_dim,
        exp_dim=exp_dim,
        num_layers=args.num_layers,
        proj_dim=args.proj_dim,
        map_dim=args.map_dim,
        fusion_mode=args.fusion_mode,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        args=args,
    )


def collect_outputs(dataloader, model, device, desc):
    import torch
    from tqdm import tqdm

    ids, labels, logits, embeds = [], [], [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            ids.extend([str(item) for item in batch["ids"]])
            output, embed = model(
                batch["image_feats"].to(device),
                batch["text_feats"].to(device),
                batch["exp_feats"].to(device),
                return_embed=True,
            )
            labels.append(batch["labels"].detach().cpu().view(-1))
            logits.append(output.detach().cpu().view(-1))
            embeds.append(embed.detach().cpu())

    return ids, torch.cat(labels), torch.cat(logits), torch.cat(embeds)


def build_faiss_index(train_embeds, args):
    import faiss
    import torch

    dim = train_embeds.shape[1]
    if args.metric == "l2":
        index = faiss.IndexFlatL2(dim)
        vectors = train_embeds.numpy().astype("float32")
    elif args.metric == "ip":
        index = faiss.IndexFlatIP(dim)
        vectors = train_embeds.numpy().astype("float32")
    else:
        index = faiss.IndexFlatIP(dim)
        vectors = torch.nn.functional.normalize(train_embeds, p=2, dim=1).numpy().astype("float32")

    return index, vectors


def search_neighbors(train_embeds, eval_embeds, args):
    import faiss
    import torch

    index, train_vectors = build_faiss_index(train_embeds, args)
    if args.metric == "cos":
        eval_vectors = torch.nn.functional.normalize(eval_embeds, p=2, dim=1).numpy().astype("float32")
    else:
        eval_vectors = eval_embeds.numpy().astype("float32")

    use_gpu = args.Faiss_GPU and hasattr(faiss, "StandardGpuResources")
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(train_vectors)
    return index.search(eval_vectors, args.topk)


def keep_retrieved_item(rank, score, threshold):
    # Matches the threshold behavior in model/evaluate_rac.py.
    return rank == 0 or threshold == -1 or score < threshold


def make_records(train_ids, train_labels, train_logits, eval_ids, eval_labels, eval_logits, distances, indices, args):
    records = []
    for row_idx, row_scores in enumerate(distances):
        retrieved = []
        for rank, score in enumerate(row_scores):
            if not keep_retrieved_item(rank, float(score), args.similarity_threshold):
                break

            train_idx = int(indices[row_idx, rank])
            retrieved.append({
                "rank": rank + 1,
                "id": train_ids[train_idx],
                "label": int(train_labels[train_idx].item()),
                "similarity": float(score),
                "logit": float(train_logits[train_idx].item()),
            })

        records.append({
            "id": eval_ids[row_idx],
            "label": int(eval_labels[row_idx].item()),
            "direct_logit": float(eval_logits[row_idx].item()),
            "retrieved": retrieved,
        })
    return records


def retrieval_scores(records, voting, topk):
    import numpy as np

    scores = []
    weights = np.arange(1, topk + 1)[::-1]
    for record in records:
        labels = np.array([item["label"] for item in record["retrieved"]], dtype=float)
        sims = np.array([item["similarity"] for item in record["retrieved"]], dtype=float)
        signed_sims = (labels * 2.0 - 1.0) * sims

        if voting == "mean":
            score = float(np.mean(signed_sims))
        elif voting == "arithmetic":
            length = len(signed_sims)
            score = float(np.sum(signed_sims * weights[:length]) / np.sum(weights[:length]))
        else:
            raise ValueError(f"Unsupported majority voting method: {voting}")

        scores.append(score)
    return np.asarray(scores, dtype=float)


def sigmoid(values):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-values))


def best_accuracy_threshold(probs, labels):
    import numpy as np
    from sklearn.metrics import accuracy_score

    thresholds = np.unique(np.asarray(probs, dtype=float))
    if thresholds.size == 0:
        return 0.5

    best_thr, best_acc = 0.5, -1.0
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_thr, best_acc = float(threshold), float(acc)
    return best_thr


def compute_metric_dict(labels, probs, preds, score_for_auc, threshold):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    metrics = {
        "threshold": round(float(threshold), 6),
        "total": int(len(labels)),
        "positive": int(np.sum(labels == 1)),
        "negative": int(np.sum(labels == 0)),
        "correct": int(np.sum(preds == labels)),
        "wrong": int(np.sum(preds != labels)),
        "ACC": round(float(accuracy_score(labels, preds)), 6),
        "Precision": round(float(precision_score(labels, preds, zero_division=0)), 6),
        "Recall": round(float(recall_score(labels, preds, zero_division=0)), 6),
        "F1": round(float(f1_score(labels, preds, zero_division=0)), 6),
    }

    try:
        metrics["AUC"] = round(float(roc_auc_score(labels, score_for_auc)), 6)
    except ValueError:
        metrics["AUC"] = None

    return metrics


def evaluate_retrieval(records, labels, args):
    scores = retrieval_scores(records, args.majority_voting, args.topk)
    probs = sigmoid(scores)
    threshold = args.decision_threshold
    if threshold is None:
        threshold = best_accuracy_threshold(probs, labels)
    preds = (probs >= threshold).astype(int)

    for record, score, prob, pred in zip(records, scores, probs, preds):
        record["retrieval_score"] = float(score)
        record["retrieval_prob"] = float(prob)
        record["pred"] = int(pred)

    return compute_metric_dict(labels, probs, preds, scores, threshold)


def evaluate_direct(logits, labels, decision_threshold):
    import numpy as np

    probs = sigmoid(logits)
    threshold = decision_threshold
    if threshold is None:
        threshold = best_accuracy_threshold(probs, labels)
    preds = (probs >= threshold).astype(int)
    return compute_metric_dict(labels, probs, preds, logits, threshold)


def save_outputs(args, retrieval_metrics, direct_metrics, records):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_stem = Path(args.ckpt_path).stem
    prefix = args.save_prefix or f"{args.dataset}_{args.split}_{ckpt_stem}"
    metrics_path = output_dir / f"{prefix}_metrics.json"
    csv_path = output_dir / f"{prefix}_predictions.csv"
    txt_path = output_dir / f"{prefix}_details.txt"

    payload = {
        "checkpoint": str(Path(args.ckpt_path).resolve()),
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "topk": args.topk,
        "similarity_threshold": args.similarity_threshold,
        "majority_voting": args.majority_voting,
        "retrieval_metrics": retrieval_metrics,
        "direct_classifier_metrics": direct_metrics,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "label", "pred", "retrieval_prob", "retrieval_score",
                "direct_logit", "no_retrieved", "retrieved_ids",
                "retrieved_labels", "retrieved_similarities",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "id": record["id"],
                "label": record["label"],
                "pred": record["pred"],
                "retrieval_prob": record["retrieval_prob"],
                "retrieval_score": record["retrieval_score"],
                "direct_logit": record["direct_logit"],
                "no_retrieved": len(record["retrieved"]),
                "retrieved_ids": "|".join(item["id"] for item in record["retrieved"]),
                "retrieved_labels": "|".join(str(item["label"]) for item in record["retrieved"]),
                "retrieved_similarities": "|".join(f"{item['similarity']:.6f}" for item in record["retrieved"]),
            })

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("RAC test results\n")
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))
        f.write("\n\nWrong predictions\n")
        wrong_records = [record for record in records if record["pred"] != record["label"]]
        for record in wrong_records:
            f.write(
                f"id={record['id']}\tlabel={record['label']}\tpred={record['pred']}\t"
                f"prob={record['retrieval_prob']:.6f}\tscore={record['retrieval_score']:.6f}\n"
            )
            retrieved = "; ".join(
                f"rank{item['rank']}:{item['id']}|label={item['label']}|sim={item['similarity']:.6f}"
                for item in record["retrieved"]
            )
            f.write(f"retrieved={retrieved}\n\n")

    return metrics_path, csv_path, txt_path


def main():
    args = parse_args()

    import numpy as np
    import torch

    from data_loader.dataset import load_feats_from_CLIP
    from data_loader.rac_dataloader import CLIP2Dataloader

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = safe_device(args.device)
    args.device = str(device)
    if args.tf_dropout is None:
        args.tf_dropout = args.dropout[1]

    if args.dataset == "FB":
        train, dev, test_seen, test_unseen = load_feats_from_CLIP(
            os.path.join(args.path, "CLIP_Embedding"), args.dataset, args.model
        )
        split_map = {"dev": dev, "test_seen": test_seen, "test_unseen": test_unseen}
    else:
        if args.split == "test_unseen":
            raise ValueError("--split test_unseen is only supported for FB dataset.")
        train, dev, test_seen = load_feats_from_CLIP(
            os.path.join(args.path, "CLIP_Embedding"), args.dataset, args.model
        )
        split_map = {"dev": dev, "test_seen": test_seen}

    train_dl, eval_dl = CLIP2Dataloader(
        train,
        split_map[args.split],
        batch_size=args.batch_size,
        return_dataset=False,
        normalize=False,
    )

    state_dict = load_checkpoint_state(args.ckpt_path, device)
    model = build_model(args, train_dl, state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    train_ids, train_labels, train_logits, train_embeds = collect_outputs(train_dl, model, device, "Encode train")
    eval_ids, eval_labels, eval_logits, eval_embeds = collect_outputs(eval_dl, model, device, f"Encode {args.split}")

    distances, indices = search_neighbors(train_embeds, eval_embeds, args)
    records = make_records(
        train_ids, train_labels, train_logits,
        eval_ids, eval_labels, eval_logits,
        distances, indices, args,
    )

    labels_np = eval_labels.numpy().astype(int)
    retrieval_metrics = evaluate_retrieval(records, labels_np, args)
    direct_metrics = evaluate_direct(eval_logits.numpy(), labels_np, args.decision_threshold)
    metrics_path, csv_path, txt_path = save_outputs(args, retrieval_metrics, direct_metrics, records)

    print("\nRetrieval metrics")
    for key, value in retrieval_metrics.items():
        print(f"{key}: {value}")

    print("\nDirect classifier metrics")
    for key, value in direct_metrics.items():
        print(f"{key}: {value}")

    print("\nSaved files")
    print(f"metrics: {metrics_path.resolve()}")
    print(f"predictions: {csv_path.resolve()}")
    print(f"details: {txt_path.resolve()}")


if __name__ == "__main__":
    main()
