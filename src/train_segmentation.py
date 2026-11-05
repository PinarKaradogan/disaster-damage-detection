"""Phase 5 U-Net building localization (train/val only): python -m src.train_segmentation --help."""
import argparse
import json
from dataclasses import replace

from src.models.unet import UNET_ENCODERS
from src.train import nonnegative_int, positive_int
from src.training.segmentation import SegmentationConfig, run_segmentation_experiment


def build_parser():
    parser = argparse.ArgumentParser(description="U-Net building segmentation; selection by val pixel Dice")
    parser.add_argument("--model", choices=list(UNET_ENCODERS), default="unet_resnet34")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--num-workers", type=nonnegative_int, default=4)
    parser.add_argument("--tiles-per-image", type=positive_int, default=4)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--resume", action="store_true", help="resume this experiment from its last.pth")
    parser.add_argument("--session-epochs", type=positive_int, default=None)
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 4 train / 2 val images (pipeline check)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = replace(SegmentationConfig(), model_name=args.model, epochs=args.epochs, batch_size=args.batch_size,
                     num_workers=args.num_workers, tiles_per_image=args.tiles_per_image, seed=args.seed)
    result = run_segmentation_experiment(config, experiment_name=args.experiment_name, smoke=args.smoke,
                                         resume=args.resume, session_epochs=args.session_epochs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
