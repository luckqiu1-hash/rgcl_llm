import run_rac
from model.classifier_ori import classifier_hateClipper as OriClassifier
from model.loss_ori import compute_loss as compute_loss_ori


class classifier_hateClipper(OriClassifier):
    """
    Original RAC baseline wrapper.

    classifier_ori.py does not use explanation features or counterfactual loss.
    The wrapper keeps the same call interface as the current run_rac.py model so
    retrieval/evaluation utilities can still pass exp_feats without changes.
    """

    def __init__(
        self,
        image_dim,
        text_dim,
        exp_dim=None,
        num_layers=3,
        proj_dim=1024,
        map_dim=1024,
        fusion_mode="align",
        dropout=None,
        batch_norm=False,
        args=None,
    ):
        super().__init__(
            image_dim=image_dim,
            text_dim=text_dim,
            num_layers=num_layers,
            proj_dim=proj_dim,
            map_dim=map_dim,
            fusion_mode=fusion_mode,
            dropout=dropout,
            batch_norm=batch_norm,
            args=args,
        )

    def forward(self, img_feats, text_feats, exp_feats=None, return_embed=False):
        return super().forward(img_feats, text_feats, return_embed=return_embed)


def main():
    import torch
    import numpy as np

    args = run_rac.parse_args()

    # Keep user-provided values untouched, but make the default output clearly
    # separate from the explanation/CF experiments.
    args.output_path = "E:\qxy\code\\rgcl_llm\src\log_toxicn_mm_baseline/"
    args.output_log = "E:\qxy\code\\rgcl_llm\src\log_toxicn_mm_baseline.txt"

    if "_ori_baseline" not in args.exp_comment:
        args.exp_comment = f"{args.exp_comment}_ori_baseline"

    # Swap only the model and loss used by run_rac. Everything else stays
    # identical: dataloader, retrieval evaluation, classifier-head evaluation,
    # checkpoint naming, and logging.
    run_rac.classifier_hateClipper = classifier_hateClipper
    run_rac.compute_loss = compute_loss_ori

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_rac.main(args)


if __name__ == "__main__":
    main()
