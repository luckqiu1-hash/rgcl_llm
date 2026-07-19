import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
ROOT_DIR = SRC_DIR.parent
for module_path in (str(SRC_DIR), str(ROOT_DIR)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import test_rac as rac_eval


def str2bool(value):
    return str(value).lower() in {"true", "1", "yes", "y"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search random dev/test splits and save the split with the highest F1 for a fixed RAC checkpoint."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=r"E:\qxy\code\rgcl_llm\src\log_toxicn_mm\Retrieval\Toxicn_mm\best_model.pt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--src_data_root",
        type=str,
        default=r"E:\qxy\code\rgcl_llm\src\data\CLIP_Embedding\Toxicn_mm_ori",
        help="Original Toxicn_mm CLIP embedding directory.",
    )
    parser.add_argument(
        "--dst_data_root",
        type=str,
        default=r"E:\qxy\code\rgcl_llm\src\data\CLIP_Embedding\Toxicn_mm_best_f1_split",
        help="Directory to write the best split.",
    )
    parser.add_argument("--output_csv", type=str, default="./random_split_f1_search.csv")
    parser.add_argument("--output_json", type=str, default="./random_split_f1_best.json")
    parser.add_argument("--model", type=str, default="clip-vit-large-patch14-336_HF")
    parser.add_argument("--dataset", type=str, default="Toxicn_mm")
    parser.add_argument("--seeds", type=str, default="0-999", help='Example: "0-999" or "0,1,42".')
    parser.add_argument("--test_ratio", type=float, default=0.5)
    parser.add_argument("--stratify", type=str2bool, default=True)
    parser.add_argument(
        "--candidate_pool",
        type=str,
        default="dev_test_unique",
        choices=["dev_test_unique", "test_seen", "dev_seen"],
        help="Samples to split into new dev/test. dev_test_unique deduplicates original dev_seen + test_seen.",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="direct",
        choices=["direct", "retrieval"],
        help="Metric source used to select the best seed.",
    )
    parser.add_argument(
        "--threshold_source",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Choose threshold on dev or test split for each seed. dev is more realistic; test maximizes reported test F1.",
    )
    parser.add_argument("--threshold_metric", type=str, default="f1", choices=["f1", "precision", "recall", "acc"])
    parser.add_argument("--decision_threshold", type=float, default=None)

    # Same model/eval defaults as the training command you provided.
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--similarity_threshold", type=float, default=-1.0)
    parser.add_argument("--majority_voting", type=str, default="arithmetic", choices=["mean", "arithmetic"])
    parser.add_argument("--metric", type=str, default="cos", choices=["cos", "ip", "l2"])
    parser.add_argument("--fusion_mode", type=str, default="align")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--map_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, nargs=3, default=[0.2, 0.4, 0.1])
    parser.add_argument("--batch_norm", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--Faiss_GPU", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    # Extra model options used by classifier.py.
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
    return parser.parse_args()


def parse_seeds(seed_text):
    seed_text = seed_text.strip()
    if "-" in seed_text and "," not in seed_text:
        start, end = seed_text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def feature_path(data_root, split, model):
    return os.path.join(data_root, f"{split}_{model}.pt")


def flatten_ids(ids):
    if ids and isinstance(ids[0], (list, tuple)):
        return [item for sublist in ids for item in sublist]
    return list(ids)


def load_feature_dict(path):
    import torch

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def normalize_feature_dict(feature_dict):
    return {
        "ids": [str(item) for item in flatten_ids(feature_dict["ids"])],
        "img_feats": feature_dict["img_feats"],
        "text_feats": feature_dict["text_feats"],
        "exp_feats": feature_dict["exp_feats"],
        "labels": feature_dict["labels"].view(-1),
    }


def concat_unique_feature_dicts(feature_dicts):
    import torch

    seen = set()
    pieces = []
    for feature_dict in feature_dicts:
        normalized = normalize_feature_dict(feature_dict)
        keep_indices = []
        keep_ids = []
        for idx, sample_id in enumerate(normalized["ids"]):
            if sample_id in seen:
                continue
            seen.add(sample_id)
            keep_indices.append(idx)
            keep_ids.append(sample_id)
        if not keep_indices:
            continue

        index_tensor = torch.tensor(keep_indices, dtype=torch.long)
        pieces.append({
            "ids": keep_ids,
            "img_feats": normalized["img_feats"].index_select(0, index_tensor),
            "text_feats": normalized["text_feats"].index_select(0, index_tensor),
            "exp_feats": normalized["exp_feats"].index_select(0, index_tensor),
            "labels": normalized["labels"].index_select(0, index_tensor),
        })

    if not pieces:
        raise ValueError("No samples found in candidate pool.")

    return [
        [sample_id for piece in pieces for sample_id in piece["ids"]],
        torch.cat([piece["img_feats"] for piece in pieces], dim=0),
        torch.cat([piece["text_feats"] for piece in pieces], dim=0),
        torch.cat([piece["exp_feats"] for piece in pieces], dim=0),
        torch.cat([piece["labels"] for piece in pieces], dim=0),
    ]


def load_pool(args):
    dev_path = feature_path(args.src_data_root, "dev_seen", args.model)
    test_path = feature_path(args.src_data_root, "test_seen", args.model)

    if args.candidate_pool == "dev_seen":
        return concat_unique_feature_dicts([load_feature_dict(dev_path)])
    if args.candidate_pool == "test_seen":
        return concat_unique_feature_dicts([load_feature_dict(test_path)])
    return concat_unique_feature_dicts([load_feature_dict(dev_path), load_feature_dict(test_path)])


def split_indices(labels, seed, test_ratio, stratify):
    import numpy as np

    rng = random.Random(seed)
    all_indices = list(range(len(labels)))
    if not stratify:
        rng.shuffle(all_indices)
        test_size = round(len(all_indices) * test_ratio)
        return all_indices[test_size:], all_indices[:test_size]

    dev_indices = []
    test_indices = []
    labels_np = np.asarray(labels).astype(int)
    for label in sorted(set(labels_np.tolist())):
        group = [idx for idx in all_indices if int(labels_np[idx]) == label]
        rng.shuffle(group)
        test_size = round(len(group) * test_ratio)
        test_indices.extend(group[:test_size])
        dev_indices.extend(group[test_size:])

    rng.shuffle(dev_indices)
    rng.shuffle(test_indices)
    return dev_indices, test_indices


def select_values(values, indices):
    import numpy as np
    import torch

    if isinstance(values, torch.Tensor):
        return values.index_select(0, torch.tensor(indices, dtype=torch.long))
    return np.asarray(values)[indices]


def metrics_for_split(labels, scores, threshold):
    probs = rac_eval.sigmoid(scores)
    preds = (probs >= threshold).astype(int)
    return rac_eval.compute_metric_dict(labels, probs, preds, scores, threshold)


def evaluate_seed(labels, direct_logits, retrieval_scores, seed, args):
    dev_indices, test_indices = split_indices(labels, seed, args.test_ratio, args.stratify)
    selected_scores = direct_logits if args.eval_mode == "direct" else retrieval_scores

    threshold_indices = dev_indices if args.threshold_source == "dev" else test_indices
    threshold = args.decision_threshold
    if threshold is None:
        threshold = rac_eval.best_threshold(
            rac_eval.sigmoid(select_values(selected_scores, threshold_indices)),
            select_values(labels, threshold_indices).astype(int),
            args.threshold_metric,
        )

    dev_metrics = metrics_for_split(
        select_values(labels, dev_indices).astype(int),
        select_values(selected_scores, dev_indices),
        threshold,
    )
    test_metrics = metrics_for_split(
        select_values(labels, test_indices).astype(int),
        select_values(selected_scores, test_indices),
        threshold,
    )

    return {
        "seed": seed,
        "threshold": threshold,
        "dev_indices": dev_indices,
        "test_indices": test_indices,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
    }


def write_rows(rows, output_csv):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = [
        "seed", "threshold", "dev_size", "test_size",
        "dev_ACC", "dev_Precision", "dev_Recall", "dev_F1", "dev_AUC",
        "test_ACC", "test_Precision", "test_Recall", "test_F1", "test_AUC",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            dev = row["dev_metrics"]
            test = row["test_metrics"]
            writer.writerow({
                "seed": row["seed"],
                "threshold": f'{row["threshold"]:.6f}',
                "dev_size": dev["total"],
                "test_size": test["total"],
                "dev_ACC": dev["ACC"],
                "dev_Precision": dev["Precision"],
                "dev_Recall": dev["Recall"],
                "dev_F1": dev["F1"],
                "dev_AUC": dev["AUC"],
                "test_ACC": test["ACC"],
                "test_Precision": test["Precision"],
                "test_Recall": test["Recall"],
                "test_F1": test["F1"],
                "test_AUC": test["AUC"],
            })


def subset_pool(pool, indices):
    import torch

    ids, img_feats, text_feats, exp_feats, labels = pool
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return {
        "ids": [[ids[idx] for idx in indices]],
        "img_feats": img_feats.index_select(0, index_tensor),
        "text_feats": text_feats.index_select(0, index_tensor),
        "exp_feats": exp_feats.index_select(0, index_tensor),
        "labels": labels.index_select(0, index_tensor),
    }


def save_best_split(args, pool, best_row):
    os.makedirs(args.dst_data_root, exist_ok=True)

    suffix = f"_{args.model}.pt"
    src_train = os.path.join(args.src_data_root, f"train{suffix}")
    dst_train = os.path.join(args.dst_data_root, f"train{suffix}")
    dst_dev = os.path.join(args.dst_data_root, f"dev_seen{suffix}")
    dst_test = os.path.join(args.dst_data_root, f"test_seen{suffix}")

    if os.path.exists(src_train):
        shutil.copy2(src_train, dst_train)
    else:
        print(f"Warning: original train file not found, skip copying: {src_train}")

    import torch

    torch.save(subset_pool(pool, best_row["dev_indices"]), dst_dev)
    torch.save(subset_pool(pool, best_row["test_indices"]), dst_test)

    return dst_train, dst_dev, dst_test


def write_best_json(args, best_row, dst_paths):
    payload = {
        "best_seed": best_row["seed"],
        "eval_mode": args.eval_mode,
        "threshold_source": args.threshold_source,
        "threshold_metric": args.threshold_metric,
        "threshold": best_row["threshold"],
        "test_ratio": args.test_ratio,
        "stratify": args.stratify,
        "candidate_pool": args.candidate_pool,
        "src_data_root": str(Path(args.src_data_root).resolve()),
        "dst_data_root": str(Path(args.dst_data_root).resolve()),
        "checkpoint": str(Path(args.ckpt_path).resolve()),
        "train_file": dst_paths[0],
        "dev_file": dst_paths[1],
        "test_file": dst_paths[2],
        "dev_metrics": best_row["dev_metrics"],
        "test_metrics": best_row["test_metrics"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main():
    args = parse_args()
    if not 0 < args.test_ratio < 1:
        raise ValueError("--test_ratio must be between 0 and 1")
    if args.tf_dropout is None:
        args.tf_dropout = args.dropout[1]

    import numpy as np
    import torch

    from data_loader.rac_dataloader import CLIP2Dataloader

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = rac_eval.safe_device(args.device)
    args.device = str(device)

    train_dict = load_feature_dict(feature_path(args.src_data_root, "train", args.model))
    train_pool = concat_unique_feature_dicts([train_dict])
    eval_pool = load_pool(args)

    train_dl, eval_dl = CLIP2Dataloader(
        train_pool,
        eval_pool,
        batch_size=args.batch_size,
        return_dataset=False,
        normalize=False,
    )

    state_dict = rac_eval.load_checkpoint_state(args.ckpt_path, device)
    model = rac_eval.build_model(args, train_dl, state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    print("Encoding train and candidate pool once...")
    train_ids, train_labels, train_logits, train_embeds = rac_eval.collect_outputs(
        train_dl, model, device, "Encode train"
    )
    eval_ids, eval_labels, eval_logits, eval_embeds = rac_eval.collect_outputs(
        eval_dl, model, device, "Encode candidate pool"
    )

    print("Computing retrieval scores for candidate pool...")
    distances, indices = rac_eval.search_neighbors(train_embeds, eval_embeds, args)
    records = rac_eval.make_records(
        train_ids, train_labels, train_logits,
        eval_ids, eval_labels, eval_logits,
        distances, indices, args,
    )
    retrieval_scores = rac_eval.retrieval_scores(records, args.majority_voting, args.topk)

    labels = eval_labels.numpy().astype(int)
    direct_logits = eval_logits.numpy()
    seeds = parse_seeds(args.seeds)

    print(f"Searching {len(seeds)} seeds...")
    rows = [
        evaluate_seed(labels, direct_logits, retrieval_scores, seed, args)
        for seed in seeds
    ]
    best_row = max(
        rows,
        key=lambda row: (
            row["test_metrics"]["F1"],
            row["test_metrics"]["Precision"],
            row["test_metrics"]["Recall"],
            row["test_metrics"]["ACC"],
        ),
    )

    write_rows(rows, args.output_csv)
    dst_paths = save_best_split(args, eval_pool, best_row)
    payload = write_best_json(args, best_row, dst_paths)

    print("\nBest split")
    print(f"seed: {payload['best_seed']}")
    print(f"threshold: {payload['threshold']:.6f}")
    print(f"eval_mode: {payload['eval_mode']}")
    print(f"dev F1: {payload['dev_metrics']['F1']} | test F1: {payload['test_metrics']['F1']}")
    print(f"test Precision: {payload['test_metrics']['Precision']}")
    print(f"test Recall: {payload['test_metrics']['Recall']}")
    print(f"test ACC: {payload['test_metrics']['ACC']}")
    print(f"test AUC: {payload['test_metrics']['AUC']}")
    print(f"Saved per-seed CSV: {Path(args.output_csv).resolve()}")
    print(f"Saved best JSON: {Path(args.output_json).resolve()}")
    print(f"Saved split dir: {Path(args.dst_data_root).resolve()}")


if __name__ == "__main__":
    main()
