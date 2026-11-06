"""Phase 6 per-disaster analysis with the pre-registered support rule (results/phase6_protocol.md, section 5).

python -m src.evaluation.disaster_analysis predict-val --experiment <classifier run> --output <csv>
python -m src.evaluation.disaster_analysis classifier --predictions <csv> --output <json>
python -m src.evaluation.disaster_analysis segmentation --per-image <csv> --output <json>

Rules: class F1 is reported only for classes with >= MIN_SUPPORT buildings in that group, otherwise
"n < 30"; the group's "present-class Macro-F1" averages only those classes and states how many were
included; groups with fewer than MIN_SUPPORT buildings report accuracy only. Every row carries its counts.
The same code runs on validation predictions (development) and on the one-shot test predictions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.constants import LABEL_MAP
from src.data.dataset import ROOT

MIN_SUPPORT = 30
CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
DISASTER_TYPE = {
    "hurricane-florence": "flooding", "hurricane-harvey": "flooding", "midwest-flooding": "flooding",
    "hurricane-matthew": "wind", "hurricane-michael": "wind",
    "santa-rosa-wildfire": "fire", "socal-fire": "fire",
    "mexico-earthquake": "earthquake", "palu-tsunami": "tsunami", "guatemala-volcano": "volcano",
}


def disaster_of(image_id):
    return str(image_id).rsplit("_", 1)[0]


def group_metrics(labels, predictions, *, min_support=MIN_SUPPORT):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    support = [int((labels == c).sum()) for c in range(4)]
    row = dict(buildings=int(len(labels)), support=dict(zip(CLASS_NAMES, support)),
               accuracy=float((labels == predictions).mean()) if len(labels) else None)
    if len(labels) < min_support:
        row.update(class_f1=None, present_class_macro_f1=None, classes_included=0,
                   note=f"fewer than {min_support} buildings: accuracy only")
        return row
    f1 = f1_score(labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0)
    included = [c for c in range(4) if support[c] >= min_support]
    row.update(class_f1={CLASS_NAMES[c]: (float(f1[c]) if c in included else f"n < {min_support}") for c in range(4)},
               present_class_macro_f1=float(np.mean([f1[c] for c in included])) if included else None,
               classes_included=len(included))
    return row


def analyse_classifier(predictions_csv, *, min_support=MIN_SUPPORT):
    frame = pd.read_csv(predictions_csv)
    if "disaster" not in frame or frame["disaster"].isna().any():
        frame["disaster"] = frame["image_id"].map(disaster_of)
    unknown = sorted(set(frame["disaster"]) - set(DISASTER_TYPE))
    if unknown:
        raise ValueError(f"Disasters without a type mapping: {unknown}")
    frame["disaster_type"] = frame["disaster"].map(DISASTER_TYPE)
    report = dict(min_support=min_support, overall=group_metrics(frame["label"], frame["pred"], min_support=min_support),
                  by_disaster={}, by_type={})
    for key, column in (("by_disaster", "disaster"), ("by_type", "disaster_type")):
        for name, group in sorted(frame.groupby(column)):
            report[key][name] = group_metrics(group["label"], group["pred"], min_support=min_support)
    if "holdout_source" in frame and frame["holdout_source"].notna().any():
        report["by_holdout_source"] = {name: group_metrics(g["label"], g["pred"], min_support=min_support)
                                       for name, g in sorted(frame.groupby("holdout_source"))}
    return report


def analyse_segmentation(per_image_csv):
    frame = pd.read_csv(per_image_csv)
    frame["disaster"] = frame["image_id"].map(disaster_of)
    frame["disaster_type"] = frame["disaster"].map(DISASTER_TYPE)

    def summary(group):
        tp, fp, fn = (int(group[c].sum()) for c in ("tp", "fp", "fn"))
        denominator = 2 * tp + fp + fn
        return dict(images=int(len(group)), images_with_buildings=int((group["gt_pixels"] > 0).sum()),
                    dice=float(2 * tp / denominator) if denominator else 1.0,
                    iou=float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0)

    return dict(overall=summary(frame),
                by_disaster={n: summary(g) for n, g in sorted(frame.groupby("disaster"))},
                by_type={n: summary(g) for n, g in sorted(frame.groupby("disaster_type"))})


def markdown_table(report):
    """Compact per-disaster table for the report (class F1 or 'n < 30')."""
    lines = ["| Afet | Tür | Bina | no | minor | major | destroyed | Mevcut sınıf Macro-F1 (sınıf) | Acc. |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["by_disaster"].items():
        def cell(c):
            if row["class_f1"] is None:
                return "–"
            value = row["class_f1"][c]
            return f"{value:.3f}".replace(".", ",") if isinstance(value, float) else value
        macro = row["present_class_macro_f1"]
        macro = f"{macro:.3f}".replace(".", ",") + f" ({row['classes_included']})" if macro is not None else "–"
        lines.append(f"| {name} | {DISASTER_TYPE[name]} | {row['buildings']} | " + " | ".join(cell(c) for c in CLASS_NAMES)
                     + f" | {macro} | {row['accuracy']:.3f}".replace(".", ",") + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict-val", help="validation predictions (development only; never the test split)")
    p.add_argument("--experiment", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=4)
    for name, flag in (("classifier", "--predictions"), ("segmentation", "--per-image")):
        q = sub.add_parser(name)
        q.add_argument(flag, type=Path, required=True)
        q.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "predict-val":
        from src.evaluation.evaluate_test import load_classifier, predict_split, write_predictions
        model, config, _, manifest, device = load_classifier(ROOT / "results" / args.experiment)
        frame, labels, preds, probs, _, _ = predict_split(model, config, manifest, "val", device,
                                                          num_workers=args.num_workers)
        write_predictions(args.output, frame, labels, preds, probs)
        print(f"wrote {len(frame)} validation predictions to {args.output}")
        return
    report = (analyse_classifier(args.predictions) if args.command == "classifier"
              else analyse_segmentation(args.per_image))
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.command == "classifier":
        print(markdown_table(report))
    else:
        print(json.dumps(report["by_disaster"], indent=2))


if __name__ == "__main__":
    main()
