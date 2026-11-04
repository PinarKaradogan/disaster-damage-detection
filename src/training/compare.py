"""Compare completed FULL validation runs; never consume smoke/test scores.

Stage 1: --kind imbalance selects a strategy from the three Simple CNN runs.
Stage 2: --kind models writes the final CNN/ResNet validation comparison.
"""
import argparse
import csv
import json
from pathlib import Path

from src.training.artifacts import write_json

POLICY_KEYS = (
    "manifest_sha256", "split_counts", "seed", "input_size", "epochs", "optimizer",
    "learning_rate", "weight_decay", "scheduler", "scheduler_eta_min", "scheduler_t_max",
    "early_stopping_patience", "augmentation", "normalization", "amp",
    "label_map", "class_weight_formula", "train_class_counts",
)
# Loader-only settings that may differ between runs. They do not change the model, loss,
# data, split or metric, but can change load order of augmentation RNG draws across
# workers, so every value is recorded in the CSV and the JSON summary.
ALLOWED_DIFFERENCES = ("num_workers",)
STRATEGIES = {("weighted_ce", "none"), ("focal", "none"), ("ce", "weighted")}
COLUMNS = ["model", "val_macro_f1", "no_damage_f1", "minor_damage_f1", "major_damage_f1",
           "destroyed_f1", "best_epoch", "parameter_count", "training_time_seconds",
           "loss_strategy", "batch_size", "seed", "num_workers"]


def compare_runs(run_dirs, *, kind, output_path):
    if kind not in {"imbalance", "models"}:
        raise ValueError("kind must be imbalance or models")
    runs = []
    for run_dir in map(Path, run_dirs):
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "completed" or config["smoke"] or metrics["smoke"]:
            raise ValueError("Only completed non-smoke runs may be compared")
        if config["evaluation_split"] != "val" or metrics["evaluation_split"] != "val":
            raise ValueError("Only VALIDATION results are allowed")
        runs.append((run_dir, config, metrics))
    expected = 3 if kind == "imbalance" else 2
    if len(runs) != expected:
        raise ValueError(f"Expected {expected} runs")
    baseline = runs[0][1]
    for _, config, _ in runs[1:]:
        differences = [key for key in POLICY_KEYS if config[key] != baseline[key]]
        if kind == "imbalance" and config["batch_size"] != baseline["batch_size"]:
            differences.append("batch_size")
        if differences:
            raise ValueError(f"Unfair comparison: different policy fields {differences}")
    if kind == "imbalance":
        if any(config["model_name"] != "simple_cnn" for _, config, _ in runs):
            raise ValueError("Imbalance comparison must use Simple CNN only")
        if {(config["loss"], config["sampler"]) for _, config, _ in runs} != STRATEGIES:
            raise ValueError("Expected weighted CE, focal, and sampler + CE")
    else:
        if {config["model_name"] for _, config, _ in runs} != {"simple_cnn", "resnet34_early_fusion"}:
            raise ValueError("Expected Simple CNN and ResNet34")
        for _, config, _ in runs:
            if (config["loss"], config["sampler"]) != (baseline["loss"], baseline["sampler"]):
                raise ValueError("Final models must use the same imbalance strategy")
            if config["model_name"] == "resnet34_early_fusion" and not config["pretrained"]:
                raise ValueError("Final ResNet run must use pretrained=True")
    output_path = Path(output_path)
    if output_path.exists() or output_path.with_suffix(".json").exists():
        raise FileExistsError("Comparison output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for _, config, metrics in runs:
            writer.writerow(dict(
                model=config["model_name"], val_macro_f1=metrics["best_val_macro_f1"],
                **dict(zip(COLUMNS[2:6], metrics["class_f1"], strict=True)),
                best_epoch=metrics["best_epoch"], parameter_count=config["parameter_count"],
                training_time_seconds=metrics["training_time_seconds"],
                loss_strategy=f"{config['loss']}+sampler:{config['sampler']}",
                batch_size=config["batch_size"], seed=config["seed"],
                num_workers=config["num_workers"],
            ))
    winner = max(runs, key=lambda run: run[2]["best_val_macro_f1"])
    summary = {
        "schema_version": 1, "evaluation_split": "val", "kind": kind,
        "selection_rule": "Maximum best validation macro F1; ties retain CLI input order",
        "selected_run": str(winner[0]), "loss": winner[1]["loss"], "sampler": winner[1]["sampler"],
        "best_val_macro_f1": winner[2]["best_val_macro_f1"],
        "runs": [str(path) for path, _, _ in runs],
        "batch_sizes": {config["model_name"]: config["batch_size"] for _, config, _ in runs},
        "allowed_policy_differences": {
            key: {str(path): config[key] for path, config, _ in runs}
            for key in ALLOWED_DIFFERENCES
            if len({json.dumps(config[key]) for _, config, _ in runs}) > 1
        },
        "test_used": False,
    }
    write_json(output_path.with_suffix(".json"), summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["imbalance", "models"])
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(compare_runs(args.runs, kind=args.kind, output_path=args.output), indent=2))


if __name__ == "__main__":
    main()
