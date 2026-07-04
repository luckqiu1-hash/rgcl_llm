import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
for module_path in (CURRENT_DIR, ROOT_DIR):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import run_rac


_original_load_feats_from_clip = run_rac.load_feats_from_CLIP


def _replace_exp_feats_with_text_feats(split):
    """
    Keep the original split structure but replace exp_feats with text_feats.

    Split format:
      [ids, img_feats, text_feats, exp_feats, labels]
    """
    ids, img_feats, text_feats, _exp_feats, labels = split
    return [ids, img_feats, text_feats, text_feats, labels]


def load_feats_text_as_exp(path, dataset, model, *args, **kwargs):
    splits = _original_load_feats_from_clip(path, dataset, model, *args, **kwargs)
    return tuple(_replace_exp_feats_with_text_feats(split) for split in splits)


def main():
    args = run_rac.parse_args()

    # Avoid accidentally sharing the same output directory with the normal
    # explanation-feature experiment.
    if "text_as_exp" not in args.exp_comment:
        args.exp_comment = (args.exp_comment or "") + "_text_as_exp"

    run_rac.load_feats_from_CLIP = load_feats_text_as_exp
    run_rac.np.random.seed(args.seed)
    run_rac.torch.manual_seed(args.seed)
    run_rac.main(args)


if __name__ == "__main__":
    main()
