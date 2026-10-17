"""Run Phase 4 train/validation experiments: python -m src.train --help."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.data.dataset import ROOT
from src.training.config import TrainingConfig
from src.training.experiment import run_experiment


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description="Phase 4 train/val only; selection by val macro F1")
    parser.add_argument("--model", required=True, choices=["simple_cnn", "resnet34_early_fusion"])
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--sampler", choices=["none", "weighted"], default="none")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--batch-size", type=positive_int, default=None,
                        help="default 32; in smoke mode default 8")
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/patches_full/patches.csv")
    parser.add_argument("--smoke", action="store_true", help="up to 2 epochs, 2 train and 2 val batches")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sampler == "weighted" and args.loss != "ce":
        parser.error("--sampler weighted requires --loss ce")
    config = TrainingConfig(model_name=args.model, epochs=args.epochs,
                            num_workers=args.num_workers, seed=args.seed)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size, smoke_batch_size=args.batch_size)
    result = run_experiment(config, manifest_path=args.manifest,
                            experiment_name=args.experiment_name, loss=args.loss,
                            sampler=args.sampler, smoke=args.smoke)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
