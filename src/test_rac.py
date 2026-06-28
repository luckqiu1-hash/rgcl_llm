import argparse
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
        description="Load a RAC checkpoint and save correct / wrong test predictions."
    )

    parser.add_argument("--pth_path", type=str, required=True, help="Path to the .pth/.pt checkpoint.")
    parser.add_argument("--save_txt", type=str, required=True, help="Path to save prediction details txt.")
    parser.add_argument("--path", type=str, default="./data/", help="Dataset root. CLIP_Embedding should be under it.")
    parser.add_argument("--dataset", type=str, default="Toxicn_mm")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--split", type=str, default="test_seen", choices=["dev", "test_seen", "test_unseen"])

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--similarity_threshold", type=float, default=-1.0)
    parser.add_argument("--majority_voting", type=str, default="mean", choices=["mean", "arithmetic"])
    parser.add_argument("--decision_threshold", type=float, default=None,
                        help="Class threshold on sigmoid retrieval score. Default: choose best threshold on this split.")

    parser.add_argument("--metric", type=str, default="cos", choices=["cos", "ip", "l2"])
    parser.add_argument("--fusion_mode", type=str, default="concat")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--proj_dim", type=int, default=1024)
    parser.add_argument("--map_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, nargs=3, default=[0.1, 0.4, 0.2])
    parser.add_argument("--batch_norm", type=str2bool, default=False)
    parser.add_argument("--Faiss_GPU", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")

    # Optional attributes used by classifier_hateClipper if present in trained configs.
    parser.add_argument("--tf_layers", type=int, default=1)
    parser.add_argument("--tf_heads", type=int, default=4)
    parser.add_argument("--tf_tokens", type=int, default=4)
    parser.add_argument("--tf_dropout", type=float, default=None)
    parser.add_argument("--head_scale", type=float, default=16.0)

    return parser.parse_args()


def safe_device(requested_device):
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, use CPU instead.")
        return torch.device("cpu")
    return torch.device(requested_device)


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


def build_model(args, train_dl, state_dict):
    first_batch = next(iter(train_dl))
    image_feat_dim = first_batch["image_feats"].shape[1]
    text_feat_dim = first_batch["text_feats"].shape[1]
    exp_feat_dim = first_batch["exp_feats"].shape[1]

    # Prefer checkpoint shapes when available, so old checkpoints remain loadable.
    image_feat_dim = state_dict.get("img_proj.weight", torch.empty(0, image_feat_dim)).shape[1]
    text_feat_dim = state_dict.get("text_proj.weight", torch.empty(0, text_feat_dim)).shape[1]
    exp_feat_dim = state_dict.get("exp_proj.weight", torch.empty(0, exp_feat_dim)).shape[1]

    print("Image feature dimension:", image_feat_dim)
    print("Text feature dimension:", text_feat_dim)
    print("Explanation feature dimension:", exp_feat_dim)

    model = classifier_hateClipper(
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
    return model


def collect_embeddings(dl, model, device, desc):
    ids = []
    labels = []
    logits = []
    embeds = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc=desc):
            ids.extend(batch["ids"])
            out, embed = model(
                batch["image_feats"].to(device),
                batch["text_feats"].to(device),
                batch["exp_feats"].to(device),
                return_embed=True,
            )
            labels.append(batch["labels"].detach().cpu())
            logits.append(out.detach().cpu())
            embeds.append(embed.detach().cpu())

    return ids, torch.cat(labels), torch.cat(logits), torch.cat(embeds)


def retrieve(train_ids, train_labels, train_logits, train_embeds, eval_ids, eval_labels,
             eval_logits, eval_embeds, args):
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faiss is required for RAC retrieval testing. Please run this script in the same "
            "environment used by run_rac.py, or install faiss-cpu / faiss-gpu."
        ) from exc

    train_embeds = torch.nn.functional.normalize(train_embeds, p=2, dim=1)
    eval_embeds = torch.nn.functional.normalize(eval_embeds, p=2, dim=1)

    dim = train_embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    using_gpu = args.Faiss_GPU and torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources")
    if using_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(train_embeds.cuda())
        distances, indices = index.search(eval_embeds.cuda(), args.topk)
        distances = distances.cpu().numpy()
        indices = indices.cpu().numpy()
    else:
        index.add(train_embeds.numpy().astype("float32"))
        distances, indices = index.search(eval_embeds.numpy().astype("float32"), args.topk)

    records = []
    for row_idx, row_scores in enumerate(distances):
        retrieved = []
        for rank, score in enumerate(row_scores):
            train_idx = int(indices[row_idx, rank])
            if rank == 0 or score < args.similarity_threshold or args.similarity_threshold == -1:
                retrieved.append({
                    "id": str(train_ids[train_idx]),
                    "score": float(score),
                    "label": int(train_labels[train_idx].item()),
                    "logit": float(train_logits[train_idx].view(-1)[0].item()),
                })
            else:
                break

        records.append({
            "id": str(eval_ids[row_idx]),
            "label": int(eval_labels[row_idx].item()),
            "eval_logit": float(eval_logits[row_idx].view(-1)[0].item()),
            "retrieved": retrieved,
        })
    return records


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def score_records(records, majority_voting, topk):
    scores = []
    if majority_voting == "arithmetic":
        weights = np.arange(1, topk + 1)[::-1]

    for record in records:
        retrieved_labels = np.array([item["label"] for item in record["retrieved"]])
        retrieved_scores = np.array([item["score"] for item in record["retrieved"]])
        label_sign = retrieved_labels * 2 - 1
        weighted_values = label_sign * retrieved_scores

        if majority_voting == "mean":
            scores.append(float(np.mean(weighted_values)))
        elif majority_voting == "arithmetic":
            length = len(weighted_values)
            scores.append(float(np.sum(weighted_values * weights[:length]) / np.sum(weights[:length])))
        else:
            raise ValueError("Unsupported majority voting method.")

    return np.array(scores)


def best_threshold(scores, labels):
    thresholds = np.linspace(0, 1, 1001)
    best_thr = 0.5
    best_acc = -1.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = threshold
    return best_thr


def save_predictions(records, output_path, metrics):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    correct = [record for record in records if record["pred"] == record["label"]]
    wrong = [record for record in records if record["pred"] != record["label"]]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("RAC test prediction results\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write(f"===== CORRECT ({len(correct)}) =====\n")
        for record in correct:
            write_record(f, record)

        f.write(f"\n===== WRONG ({len(wrong)}) =====\n")
        for record in wrong:
            write_record(f, record)


def write_record(f, record):
    f.write(
        "id={id}\tlabel={label}\tpred={pred}\tprob={prob:.6f}\tscore={score:.6f}\t"
        "eval_logit={eval_logit:.6f}\tno_retrieved={no_retrieved}\n".format(
            id=record["id"],
            label=record["label"],
            pred=record["pred"],
            prob=record["prob"],
            score=record["score"],
            eval_logit=record["eval_logit"],
            no_retrieved=len(record["retrieved"]),
        )
    )
    retrieved_text = "; ".join(
        "rank{}:{}|label={}|sim={:.6f}|logit={:.6f}".format(
            rank + 1,
            item["id"],
            item["label"],
            item["score"],
            item["logit"],
        )
        for rank, item in enumerate(record["retrieved"])
    )
    f.write(f"retrieved={retrieved_text}\n\n")


def main():
    args = parse_args()

    global np
    global torch
    global accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    global tqdm
    global load_feats_from_CLIP, CLIP2Dataloader, classifier_hateClipper

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from tqdm import tqdm

    from data_loader.dataset import load_feats_from_CLIP
    from data_loader.rac_dataloader import CLIP2Dataloader
    from model.classifier import classifier_hateClipper

    device = safe_device(args.device)
    args.device = str(device)
    if args.tf_dropout is None:
        args.tf_dropout = args.dropout[1]

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

    train_dl, eval_dl = CLIP2Dataloader(
        train,
        datasets[args.split],
        batch_size=args.batch_size,
        return_dataset=False,
        normalize=False,
    )

    state_dict = load_checkpoint_state(args.pth_path, device)
    model = build_model(args, train_dl, state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    train_ids, train_labels, train_logits, train_embeds = collect_embeddings(
        train_dl, model, device, "Encode train"
    )
    eval_ids, eval_labels, eval_logits, eval_embeds = collect_embeddings(
        eval_dl, model, device, f"Encode {args.split}"
    )

    records = retrieve(
        train_ids, train_labels, train_logits, train_embeds,
        eval_ids, eval_labels, eval_logits, eval_embeds,
        args,
    )
    raw_scores = score_records(records, args.majority_voting, args.topk)
    probs = sigmoid(raw_scores)
    labels_np = eval_labels.numpy().astype(int)
    threshold = args.decision_threshold
    if threshold is None:
        threshold = best_threshold(probs, labels_np)
    preds = (probs >= threshold).astype(int)

    for record, score, prob, pred in zip(records, raw_scores, probs, preds):
        record["score"] = float(score)
        record["prob"] = float(prob)
        record["pred"] = int(pred)

    metrics = {
        "checkpoint": os.path.abspath(args.pth_path),
        "dataset": args.dataset,
        "split": args.split,
        "topk": args.topk,
        "similarity_threshold": args.similarity_threshold,
        "majority_voting": args.majority_voting,
        "decision_threshold": round(float(threshold), 6),
        "total": len(records),
        "correct": int(np.sum(preds == labels_np)),
        "wrong": int(np.sum(preds != labels_np)),
        "acc": round(float(accuracy_score(labels_np, preds)), 6),
        "roc": round(float(roc_auc_score(labels_np, raw_scores)), 6),
        "precision": round(float(precision_score(labels_np, preds, zero_division=0)), 6),
        "recall": round(float(recall_score(labels_np, preds, zero_division=0)), 6),
        "f1": round(float(f1_score(labels_np, preds, zero_division=0)), 6),
    }
    save_predictions(records, args.save_txt, metrics)

    print("Test finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("Saved prediction details to:", os.path.abspath(args.save_txt))


if __name__ == "__main__":
    main()
