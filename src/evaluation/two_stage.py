"""Phase 5.4: two-stage inference (U-Net localization -> damage classifier) and its evaluation.

Stage 1: U-Net on the full 1024x1024 PRE image -> building probability -> threshold ->
connected components (4-connectivity) -> components smaller than ``min_area`` dropped.
Full-image inference means component coordinates are already original image coordinates.
Stage 2: every component's bbox goes through the SAME Phase 3 crop rule
(``patch_geometry`` + ``paired_crops``: square side max(64, 2 x bbox), black padding) on
PRE and POST, then the Phase 4/5 deterministic 128x128 transform, then the classifier.

Evaluation (validation only) keeps two numbers apart:
- oracle: the classifier on crops from the GROUND-TRUTH building boxes (reproduces the
  patch-based classification result);
- end-to-end: the classifier on crops from U-NET components, matched to ground-truth
  buildings. Missed buildings are reported separately.
"""
import argparse
import json
import multiprocessing
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely import wkt

from src.data.constants import LABEL_MAP
from src.data.dataset import ROOT
from src.data.segmentation_dataset import normalize, read_split_ids
from src.models import create_model
from src.models.unet import create_unet
from src.preprocessing.extract_patches import paired_crops, patch_geometry
from src.training.data import deterministic_transform
from src.training.metrics import classification_metrics

CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
DAMAGE_COLORS = {0: (46, 204, 64), 1: (255, 220, 0), 2: (255, 133, 27), 3: (255, 30, 30)}
DEFAULT_UNET = "results/phase5_unet_resnet34_seed42_w4/best.pth"
DEFAULT_CLASSIFIER = "results/phase4_final_resnet34_focal_seed42_w4/best.pth"
MIN_AREA = 10  # pixels; removes speckle only (smallest xBD footprints are larger), not tuned


# ---------------------------------------------------------------- stage 1: localization
@torch.no_grad()
def building_probability(unet, pre_uint8, device, *, amp=True):
    unet.eval()
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        logits = unet(normalize(pre_uint8)[None].to(device))
    return torch.sigmoid(logits.float())[0, 0].cpu().numpy()


def split_touching(binary, erosion):
    """Separate buildings joined by thin bridges.

    Erode the mask ``erosion`` times; each remaining core is one building seed. Every mask
    pixel takes the label of its nearest seed (Euclidean). Connected regions that lose all
    their pixels in the erosion (small buildings) keep their own label instead.
    """
    labels, count = ndimage.label(binary)
    if erosion <= 0 or count == 0:
        return labels, count
    core = ndimage.binary_erosion(binary, iterations=erosion)
    seeds, n_seeds = ndimage.label(core)
    if n_seeds == 0:
        return labels, count
    _, (rows, cols) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    split = np.where(binary, seeds[rows, cols], 0)
    seeded = np.zeros(count + 1, dtype=bool)
    seeded[np.unique(labels[core])] = True
    seeded[0] = False
    # A pixel must never take a seed from another connected region (that would cross
    # background): unseeded regions and such pixels keep their region's own label.
    foreign = binary & (labels[rows, cols] != labels)
    keep_region = (binary & ~seeded[labels]) | foreign
    split[keep_region] = n_seeds + labels[keep_region]
    unique = np.unique(split)
    remap = np.zeros(int(unique.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(len(unique))
    return remap[split], len(unique) - 1


def components(probability, *, threshold=0.5, min_area=MIN_AREA, split_erosion=0):
    """Binary mask -> labeled components (0 = background, 1..N) and their pixel bboxes.

    ``split_erosion`` > 0 separates touching buildings (see ``split_touching``).
    bbox uses pixel-edge coordinates: x_max = last column + 1, so width = x_max - x_min.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    labels, count = split_touching(probability > threshold, split_erosion)  # 4-connectivity
    if count:
        areas = np.bincount(labels.ravel(), minlength=count + 1)
        keep = np.flatnonzero(areas >= min_area)
        keep = keep[keep > 0]
        remap = np.zeros(count + 1, dtype=np.int32)
        remap[keep] = np.arange(1, len(keep) + 1)
        labels = remap[labels]
    boxes = []
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        rows, cols = slices
        boxes.append(dict(component=index, area=int((labels[slices] == index).sum()),
                          x_min=float(cols.start), y_min=float(rows.start),
                          x_max=float(cols.stop), y_max=float(rows.stop)))
    return labels, boxes


# ---------------------------------------------------------------- stage 2: classification
def crop_pairs(pre, post, boxes):
    """Same Phase 3 crop for PRE and POST; returns PIL patch pairs and their crops."""
    pairs = []
    for box in boxes:
        crop = patch_geometry(box["x_min"], box["y_min"], box["x_max"], box["y_max"],
                              box["x_max"] - box["x_min"], box["y_max"] - box["y_min"])
        pairs.append((*paired_crops(pre, post, crop), crop))
    return pairs


@torch.no_grad()
def classify(classifier, pairs, device, *, input_size=128, batch_size=64, amp=True):
    if not pairs:
        return np.zeros((0, 4), dtype=np.float32)
    classifier.eval()
    transform = deterministic_transform(input_size)
    outputs = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        pre = torch.stack([transform(p[0]) for p in chunk]).to(device)
        post = torch.stack([transform(p[1]) for p in chunk]).to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = classifier(pre, post)
        outputs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(outputs)


def load_models(device, *, unet_path=DEFAULT_UNET, classifier_path=DEFAULT_CLASSIFIER, project_root=ROOT):
    unet_ckpt = torch.load(Path(project_root) / unet_path, map_location="cpu", weights_only=True)
    unet = create_unet(unet_ckpt["config"]["model_name"], pretrained=False)
    unet.load_state_dict(unet_ckpt["model_state_dict"])
    cls_ckpt = torch.load(Path(project_root) / classifier_path, map_location="cpu", weights_only=True)
    cfg = cls_ckpt["config"]
    classifier = create_model(cfg["model_name"], num_classes=cfg["num_classes"], pretrained=False,
                              training_mode=cfg.get("training_mode", "full"))
    classifier.load_state_dict(cls_ckpt["model_state_dict"])
    meta = dict(unet=str(unet_path), unet_epoch=unet_ckpt["epoch"], classifier=str(classifier_path),
                classifier_model=cfg["model_name"], classifier_epoch=cls_ckpt["epoch"],
                classifier_input_size=cfg["input_size"])
    return unet.to(device), classifier.to(device), meta


def run_two_stage(pre, post, unet, classifier, device, *, threshold=0.5, min_area=MIN_AREA, input_size=128,
                  split_erosion=0):
    """pre/post: PIL RGB images of the same size. Returns component labels and detections."""
    if pre.size != post.size:
        raise ValueError("PRE/POST dimensions differ")
    probability = building_probability(unet, np.array(pre.convert("RGB")), device)
    labels, boxes = components(probability, threshold=threshold, min_area=min_area, split_erosion=split_erosion)
    probabilities = classify(classifier, crop_pairs(pre, post, boxes), device, input_size=input_size)
    for box, probs in zip(boxes, probabilities):
        box.update(predicted_class=int(probs.argmax()), predicted_damage=CLASS_NAMES[int(probs.argmax())],
                   probabilities=[round(float(p), 4) for p in probs])
    return labels, boxes


def draw_damage(image, labels, detections, *, alpha=0.45, boxes=True):
    """Color every detected building by its predicted damage on top of an RGB image."""
    base = np.array(image.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(base)
    mask = np.zeros(labels.shape, dtype=bool)
    for detection in detections:
        region = labels == detection["component"]
        color[region] = DAMAGE_COLORS[detection["predicted_class"]]
        mask |= region
    base[mask] = (1 - alpha) * base[mask] + alpha * color[mask]
    out = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for detection in detections if boxes else []:
        draw.rectangle((detection["x_min"], detection["y_min"], detection["x_max"] - 1, detection["y_max"] - 1),
                       outline=DAMAGE_COLORS[detection["predicted_class"]], width=1)
    return out


def save_example_figure(path, image_id, pre, post, labels, detections, *, project_root=ROOT):
    """PRE | POST with ground-truth damage | POST with two-stage predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    gt, buildings = gt_instances(image_id, pre.size, project_root=project_root)
    gt_detections = []
    for building in buildings:
        if building["damage"] in LABEL_MAP and (gt == building["index"]).any():
            gt_detections.append(dict(component=building["index"], predicted_class=LABEL_MAP[building["damage"]]))
    gt_image = draw_damage(post, gt, gt_detections, boxes=False)
    counts = Counter(d["predicted_damage"] for d in detections)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8.6))
    panels = [(pre, "PRE (U-Net girisi)"),
              (gt_image, f"POST + gercek hasar ({len(gt_detections)} etiketli bina)"),
              (draw_damage(post, labels, detections), f"POST + iki asamali tahmin ({len(detections)} bina adayi)")]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image); ax.set_title(title, fontsize=12); ax.axis("off")
    handles = [Patch(color=np.array(DAMAGE_COLORS[i]) / 255, label=f"{CLASS_NAMES[i]} ({counts.get(CLASS_NAMES[i], 0)} tahmin)")
               for i in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=12)
    fig.suptitle(f"Faz 5.4 iki asamali sistem - {image_id} (validation)", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- evaluation helpers
def gt_instances(image_id, size, *, project_root=ROOT):
    """Rasterize post-disaster polygons into an instance map (0 = background, i = building i)."""
    label = json.loads((Path(project_root) / "data/train/labels" / f"{image_id}_post_disaster.json")
                       .read_text(encoding="utf-8"))
    canvas = Image.new("I", size, 0)
    draw = ImageDraw.Draw(canvas)
    buildings = []
    for index, feature in enumerate(label["features"]["xy"], start=1):
        geometry = wkt.loads(feature["wkt"])
        polygons = getattr(geometry, "geoms", [geometry])
        for polygon in polygons:
            draw.polygon(list(polygon.exterior.coords), fill=index)
        buildings.append(dict(index=index, uid=feature["properties"]["uid"],
                              damage=feature["properties"].get("subtype")))
    return np.array(canvas, dtype=np.int32), buildings


def overlap_matrix(pred_labels, gt_labels):
    """Pixel intersections between every predicted component and every GT building."""
    n_pred, n_gt = int(pred_labels.max()), int(gt_labels.max())
    joint = np.bincount((pred_labels.astype(np.int64) * (n_gt + 1) + gt_labels).ravel(),
                        minlength=(n_pred + 1) * (n_gt + 1)).reshape(n_pred + 1, n_gt + 1)
    return joint, joint.sum(1), joint.sum(0)


def match_buildings(pred_labels, gt_labels, *, iou_threshold=0.5, coverage_threshold=0.5):
    """Object-level localization and assignment of GT buildings to components.

    - one-to-one greedy matching at IoU >= iou_threshold -> TP / FP / FN (localization F1);
    - a GT building is "detected" when one component covers >= coverage_threshold of its
      pixels; it then takes that component's damage prediction (merged components can
      therefore serve several buildings, which is counted as ``merged_components``).
    """
    joint, pred_area, gt_area = overlap_matrix(pred_labels, gt_labels)
    inter = joint[1:, 1:].astype(np.float64)
    union = pred_area[1:, None] + gt_area[None, 1:] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    pairs = sorted(((iou[p, g], p, g) for p, g in zip(*np.nonzero(iou >= iou_threshold))), reverse=True)
    used_p, used_g = set(), set()
    for _, p, g in pairs:
        if p not in used_p and g not in used_g:
            used_p.add(p); used_g.add(g)
    n_pred, n_gt = iou.shape
    present = gt_area[1:] > 0
    coverage = np.divide(inter, gt_area[None, 1:], out=np.zeros_like(inter), where=gt_area[None, 1:] > 0)
    best_component = coverage.argmax(0) if n_pred else np.zeros(n_gt, dtype=int)
    best_coverage = coverage.max(0) if n_pred else np.zeros(n_gt)
    assigned = {g + 1: int(best_component[g]) + 1 for g in range(n_gt)
                if present[g] and best_coverage[g] >= coverage_threshold}
    served = Counter(assigned.values())
    return dict(tp=len(used_g), fp=n_pred - len(used_p), fn=int(present.sum()) - len(used_g),
                assigned=assigned, merged_components=sum(1 for c in served.values() if c > 1),
                buildings_in_merged=sum(c for c in served.values() if c > 1))


def evaluate_val(device, *, project_root=ROOT, threshold=0.5, min_area=MIN_AREA, split_erosion=0, limit=None,
                 **model_paths):
    unet, classifier, meta = load_models(device, project_root=project_root, **model_paths)
    patches = pd.read_csv(Path(project_root) / "data/processed/patches_full/patches.csv",
                          usecols=["image_id", "uid", "damage", "split"])
    eligible = patches[patches["split"] == "val"].set_index("uid")["damage"].map(LABEL_MAP).to_dict()
    boxes_table = pd.read_csv(Path(project_root) / "data/processed/buildings_train.csv",
                              usecols=["uid", "x_min", "y_min", "x_max", "y_max"]).set_index("uid")
    image_ids = read_split_ids("val", project_root=project_root)[:limit]
    totals = Counter()
    rows = []
    for number, image_id in enumerate(image_ids, start=1):
        base = Path(project_root) / "data/train/images"
        pre = Image.open(base / f"{image_id}_pre_disaster.png").convert("RGB")
        post = Image.open(base / f"{image_id}_post_disaster.png").convert("RGB")
        labels, detections = run_two_stage(pre, post, unet, classifier, device, threshold=threshold,
                                           min_area=min_area, split_erosion=split_erosion)
        gt, buildings = gt_instances(image_id, pre.size, project_root=project_root)
        match = match_buildings(labels, gt)
        for key in ("tp", "fp", "fn", "merged_components", "buildings_in_merged"):
            totals[key] += match[key]
        totals["components"] += len(detections)
        by_component = {d["component"]: d["predicted_class"] for d in detections}
        target = [b for b in buildings if b["uid"] in eligible]
        oracle_boxes = [dict(boxes_table.loc[b["uid"]]) for b in target]
        oracle = classify(classifier, crop_pairs(pre, post, oracle_boxes), device).argmax(1) if target else []
        for building, oracle_pred in zip(target, oracle):
            component = match["assigned"].get(building["index"])
            rows.append(dict(image_id=image_id, uid=building["uid"], label=eligible[building["uid"]],
                             oracle_pred=int(oracle_pred), detected=component is not None,
                             pipeline_pred=by_component[component] if component is not None else -1))
        if number % 25 == 0:
            print(f"[two-stage val] {number}/{len(image_ids)} images", flush=True)
    frame = pd.DataFrame(rows)
    detected = frame[frame["detected"]]
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    oracle_all = classification_metrics(frame["label"], frame["oracle_pred"])
    report = dict(
        evaluation_split="val", test_used=False, models=meta, threshold=threshold, min_area=min_area,
        split_erosion=split_erosion,
        images=len(image_ids),
        localization=dict(iou_threshold=0.5, tp=totals["tp"], fp=totals["fp"], fn=totals["fn"],
                          precision=precision, recall=recall,
                          f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                          components=totals["components"], merged_components=totals["merged_components"],
                          buildings_in_merged_components=totals["buildings_in_merged"],
                          note="all GT polygons incl. un-classified count as buildings for localization"),
        eligible_buildings=len(frame),
        detection_rate=float(frame["detected"].mean()),
        detection_rate_by_class={CLASS_NAMES[c]: float(g["detected"].mean()) for c, g in frame.groupby("label")},
        oracle_gt_boxes=dict(macro_f1=oracle_all["macro_f1"], class_f1=oracle_all["class_f1"],
                             accuracy=oracle_all["accuracy"]),
        end_to_end_detected=dict(
            buildings=len(detected),
            pipeline=classification_metrics(detected["label"], detected["pipeline_pred"]),
            oracle_same_buildings=classification_metrics(detected["label"], detected["oracle_pred"])),
        end_to_end_missed_as_no_damage=classification_metrics(
            frame["label"], frame["pipeline_pred"].where(frame["detected"], 0)),
    )
    return report, frame


def sweep_val(device, *, thresholds, erosions, project_root=ROOT, min_area=MIN_AREA, limit=None,
              unet_path=DEFAULT_UNET):
    """Localization-only selection on validation: U-Net runs once per image, every
    (threshold, erosion) setting is evaluated on the same probability map."""
    checkpoint = torch.load(Path(project_root) / unet_path, map_location="cpu", weights_only=True)
    unet = create_unet(checkpoint["config"]["model_name"], pretrained=False)
    unet.load_state_dict(checkpoint["model_state_dict"])
    unet.to(device)
    settings = [(t, e) for t in thresholds for e in erosions]
    totals = {s: Counter() for s in settings}
    image_ids = read_split_ids("val", project_root=project_root)[:limit]
    for number, image_id in enumerate(image_ids, start=1):
        pre = np.array(Image.open(Path(project_root) / "data/train/images" / f"{image_id}_pre_disaster.png").convert("RGB"))
        probability = building_probability(unet, pre, device)
        gt, _ = gt_instances(image_id, (pre.shape[1], pre.shape[0]), project_root=project_root)
        for setting in settings:
            labels, boxes = components(probability, threshold=setting[0], min_area=min_area, split_erosion=setting[1])
            match = match_buildings(labels, gt)
            for key in ("tp", "fp", "fn", "merged_components", "buildings_in_merged"):
                totals[setting][key] += match[key]
            totals[setting]["components"] += len(boxes)
        if number % 50 == 0:
            print(f"[sweep val] {number}/{len(image_ids)} images", flush=True)
    rows = []
    for (threshold, erosion), c in totals.items():
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        rows.append(dict(threshold=threshold, split_erosion=erosion, precision=precision, recall=recall,
                         f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                         components=c["components"], merged_components=c["merged_components"],
                         buildings_in_merged=c["buildings_in_merged"], tp=c["tp"], fp=c["fp"], fn=c["fn"]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("predict", help="run both stages on one validation image pair and save a figure")
    one.add_argument("--image-id", required=True)
    one.add_argument("--output", type=Path, required=True)
    full = sub.add_parser("evaluate-val", help="building-level evaluation on all validation images")
    full.add_argument("--output", type=Path, required=True)
    full.add_argument("--limit", type=int, default=None)
    sweep = sub.add_parser("sweep-val", help="select threshold / split erosion on validation (localization F1)")
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--thresholds", type=float, nargs="+", default=[0.5])
    sweep.add_argument("--erosions", type=int, nargs="+", default=[0])
    sweep.add_argument("--min-area", type=int, default=MIN_AREA)
    sweep.add_argument("--limit", type=int, default=None)
    for p in (one, full):
        p.add_argument("--threshold", type=float, default=0.5)
        p.add_argument("--min-area", type=int, default=MIN_AREA)
        p.add_argument("--split-erosion", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.command == "predict":
        if args.image_id not in set(read_split_ids("val")):
            raise ValueError("Phase 5 two-stage examples use validation images only")
        unet, classifier, meta = load_models(device)
        base = ROOT / "data/train/images"
        pre = Image.open(base / f"{args.image_id}_pre_disaster.png").convert("RGB")
        post = Image.open(base / f"{args.image_id}_post_disaster.png").convert("RGB")
        labels, detections = run_two_stage(pre, post, unet, classifier, device, threshold=args.threshold,
                                           min_area=args.min_area, split_erosion=args.split_erosion)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_example_figure(args.output, args.image_id, pre, post, labels, detections)
        summary = dict(image_id=args.image_id, models=meta, buildings_found=len(detections),
                       damage_counts=dict(Counter(d["predicted_damage"] for d in detections)))
        print(json.dumps(summary, indent=2))
    elif args.command == "sweep-val":
        rows = sweep_val(device, thresholds=args.thresholds, erosions=args.erosions, min_area=args.min_area,
                         limit=args.limit)
        best = max(rows, key=lambda r: r["f1"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(evaluation_split="val", test_used=False,
                                               selection="max localization F1 (IoU>=0.5)", best=best,
                                               settings=rows), indent=2) + "\n", encoding="utf-8")
        for row in sorted(rows, key=lambda r: (r["threshold"], r["split_erosion"])):
            print(f"t={row['threshold']:.2f} e={row['split_erosion']}: F1 {row['f1']:.4f} P {row['precision']:.3f} "
                  f"R {row['recall']:.3f} components {row['components']} merged {row['merged_components']} "
                  f"({row['buildings_in_merged']} buildings)")
    else:
        report, frame = evaluate_val(device, threshold=args.threshold, min_area=args.min_area,
                                     split_erosion=args.split_erosion, limit=args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        frame.to_csv(args.output.with_name(args.output.stem + "_buildings.csv"), index=False)
        print(json.dumps({k: report[k] for k in ("localization", "detection_rate", "oracle_gt_boxes")}, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
