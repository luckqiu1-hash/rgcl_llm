import argparse
import os
import random
import re
import shutil


RECORD_PATTERN = re.compile(
    r"^id=(?P<id>[^\t]+)\tlabel=(?P<label>[01])\tpred=(?P<pred>[01])\tprob=(?P<prob>[-+0-9.eE]+)\t"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create dev/test feature files from the original test set using the same seed split as eval_random_split_acc.py."
    )
    parser.add_argument(
        "--src_data_root",
        type=str,
        default=r"E:\qxy\code\rgcl_llm\src\data\CLIP_Embedding\Toxicn_mm",
        help="Original CLIP embedding dataset directory, e.g. .../CLIP_Embedding/Toxicn_mm.",
    )
    parser.add_argument(
        "--dst_data_root",
        type=str,
        default=r"E:\qxy\code\rgcl_llm\src\data\CLIP_Embedding\Toxicn_mm_split",
        help="New CLIP embedding dataset directory to write the seed split.",
    )
    parser.add_argument("--res_txt", type=str, default="./src/toixic_mm_res.txt",
                        help="Prediction result txt used to define the split ids.")
    parser.add_argument("--dataset", type=str, default="Toxicn_mm")
    parser.add_argument("--model", type=str, default="clip-vit-large-patch14-336_HF")
    parser.add_argument("--seed", type=int, default=638)
    parser.add_argument("--test_ratio", type=float, default=0.5)
    parser.add_argument("--stratify", type=lambda x: str(x).lower() == "true", default=True)
    return parser.parse_args()


def flatten_ids(ids):
    if ids and isinstance(ids[0], (list, tuple)):
        return [item for sublist in ids for item in sublist]
    return list(ids)


def load_result_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = RECORD_PATTERN.match(line.strip())
            if match:
                records.append({
                    "id": str(match.group("id")),
                    "label": int(match.group("label")),
                })
    if not records:
        raise ValueError(f"No prediction records found in {path}")
    return records


def split_records(records, seed, test_ratio, stratify):
    rng = random.Random(seed)
    if not stratify:
        shuffled = list(records)
        rng.shuffle(shuffled)
        test_size = round(len(shuffled) * test_ratio)
        return shuffled[test_size:], shuffled[:test_size]

    val_records = []
    test_records = []
    for label in sorted({record["label"] for record in records}):
        group = [record for record in records if record["label"] == label]
        rng.shuffle(group)
        test_size = round(len(group) * test_ratio)
        test_records.extend(group[:test_size])
        val_records.extend(group[test_size:])

    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return val_records, test_records


def subset_feature_dict(feature_dict, selected_ids):
    import torch

    ids = flatten_ids(feature_dict["ids"])
    id_to_index = {str(item): idx for idx, item in enumerate(ids)}
    missing = [item for item in selected_ids if str(item) not in id_to_index]
    if missing:
        raise ValueError(f"{len(missing)} selected ids are missing from feature file. First missing id: {missing[0]}")

    indices = [id_to_index[str(item)] for item in selected_ids]
    index_tensor = torch.tensor(indices, dtype=torch.long)

    return {
        "ids": [[ids[idx] for idx in indices]],
        "img_feats": feature_dict["img_feats"].index_select(0, index_tensor),
        "text_feats": feature_dict["text_feats"].index_select(0, index_tensor),
        "exp_feats": feature_dict["exp_feats"].index_select(0, index_tensor),
        "labels": feature_dict["labels"].index_select(0, index_tensor),
    }


def copy_if_exists(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    if not 0 < args.test_ratio < 1:
        raise ValueError("--test_ratio must be between 0 and 1")

    import torch

    src_embedding_dir = args.src_data_root
    dst_embedding_dir = args.dst_data_root
    os.makedirs(dst_embedding_dir, exist_ok=True)

    suffix = f"_{args.model}.pt"
    src_train = os.path.join(src_embedding_dir, f"train{suffix}")
    src_dev = os.path.join(src_embedding_dir, f"dev_seen{suffix}")
    src_test = os.path.join(src_embedding_dir, f"test_seen{suffix}")

    dst_train = os.path.join(dst_embedding_dir, f"train{suffix}")
    dst_dev = os.path.join(dst_embedding_dir, f"dev_seen{suffix}")
    dst_test = os.path.join(dst_embedding_dir, f"test_seen{suffix}")

    if not os.path.exists(src_test):
        raise FileNotFoundError(src_test)

    copy_if_exists(src_train, dst_train)

    records = load_result_records(args.res_txt)
    val_records, test_records = split_records(records, args.seed, args.test_ratio, args.stratify)
    val_ids = [record["id"] for record in val_records]
    test_ids = [record["id"] for record in test_records]

    test_feature_dict = torch.load(src_test, map_location="cpu")
    dev_split = subset_feature_dict(test_feature_dict, val_ids)
    test_split = subset_feature_dict(test_feature_dict, test_ids)

    torch.save(dev_split, dst_dev)
    torch.save(test_split, dst_test)

    print("Seed split created.")
    print(f"seed: {args.seed}")
    print(f"val/dev size: {len(val_ids)} -> {dst_dev}")
    print(f"test size: {len(test_ids)} -> {dst_test}")
    print(f"train copied: {os.path.exists(dst_train)} -> {dst_train}")
    if os.path.exists(src_dev):
        print(f"original dev was not used for this split: {src_dev}")


if __name__ == "__main__":
    main()
