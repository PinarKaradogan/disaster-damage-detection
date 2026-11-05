"""Phase 6 Grad-CAM and error gallery for the early-fusion ResNet34 classifier.

python -m src.evaluation.gradcam --predictions <csv> --split val|test --output <png>

Sample selection is fixed and independent of the model's scores (protocol section 7): for every true
class, 4 correct and 4 incorrect predictions drawn with numpy RandomState(42) from the uid-sorted rows.
Grad-CAM uses ``backbone.layer4`` of the early-fusion model. PRE and POST enter as one 6-channel input,
so there is ONE map per building, shown at the same location on PRE and POST.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.data.dataset import ROOT, DamagePatchDataset
from src.evaluation.disaster_analysis import CLASS_NAMES
from src.training.data import IMAGENET_MEAN, IMAGENET_STD, deterministic_transform

DEFAULT_RUN = "phase4_final_resnet34_focal_seed42_w4"


def select_samples(predictions, *, per_group=4, seed=42):
    """Fixed rule: per true class, ``per_group`` correct and ``per_group`` incorrect rows (fewer if absent)."""
    frame = predictions.sort_values("uid").reset_index(drop=True)
    rng = np.random.RandomState(seed)
    chosen = []
    for label in range(4):
        for correct in (True, False):
            pool = frame[(frame["label"] == label) & ((frame["pred"] == label) == correct)]
            take = min(per_group, len(pool))
            if take:
                picked = pool.iloc[np.sort(rng.choice(len(pool), size=take, replace=False))]
                chosen.append(picked.assign(correct=correct))
    return pd.concat(chosen, ignore_index=True) if chosen else frame.iloc[:0].assign(correct=[])


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.activations, self.gradients = model, None, None
        layer.register_forward_hook(self._forward)
        layer.register_full_backward_hook(self._backward)

    def _forward(self, module, inputs, output):
        self.activations = output

    def _backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, pre, post, target=None):
        """Returns (cam [B, H, W] in [0, 1], predicted class, target class)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(pre, post)
        predicted = logits.argmax(1)
        target = predicted if target is None else target
        logits.gather(1, target[:, None]).sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=pre.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        low, high = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
        cam = ((flat - low) / (high - low).clamp_min(1e-8)).view_as(cam)
        return cam.detach(), predicted.detach(), target


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def render(samples, dataset, model, device, output, *, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam_tool = GradCAM(model, model.backbone.layer4)
    index_of = {(r.image_id, r.uid): i for i, r in dataset.df.reset_index(drop=True).iterrows()}
    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(7.5, 2.5 * rows))
    axes = np.atleast_2d(axes)
    for ax_row, (_, sample) in zip(axes, samples.iterrows()):
        item = dataset[index_of[(sample["image_id"], sample["uid"])]]
        pre, post = item["pre"][None].to(device), item["post"][None].to(device)
        cam, predicted, _ = cam_tool(pre, post)
        cam = cam[0].cpu().numpy()
        ax_row[0].imshow(denormalize(item["pre"])); ax_row[1].imshow(denormalize(item["post"]))
        ax_row[2].imshow(denormalize(item["post"])); ax_row[2].imshow(cam, cmap="jet", alpha=0.45)
        verdict = "DOGRU" if sample["correct"] else "YANLIS"
        ax_row[0].set_ylabel(f"gercek: {CLASS_NAMES[int(sample['label'])]}\ntahmin: "
                             f"{CLASS_NAMES[int(predicted[0])]} ({verdict})", fontsize=8)
        for ax, name in zip(ax_row, ("PRE", "POST", "POST + Grad-CAM")):
            ax.set_xticks([]); ax.set_yticks([])
            if ax_row is axes[0]:
                ax.set_title(name, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output, dpi=90)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="predictions CSV of the classifier run")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--experiment", default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True, help="PNG path; one file per true class is written")
    args = parser.parse_args(argv)
    from src.evaluation.evaluate_test import load_classifier
    model, config, _, manifest, device = load_classifier(ROOT / "results" / args.experiment)
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "layer4"):
        raise ValueError("Grad-CAM here supports the ResNet34 early-fusion classifier")
    samples = select_samples(pd.read_csv(args.predictions))
    dataset = DamagePatchDataset(manifest, split=args.split, project_root=ROOT,
                                 transform=deterministic_transform(config["input_size"]))
    outputs = []
    for label in range(4):
        path = args.output.with_name(f"{args.output.stem}_{CLASS_NAMES[label]}{args.output.suffix}")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        subset = samples[samples["label"] == label]
        if len(subset):
            render(subset, dataset, model, device, path,
                   title=f"Grad-CAM ({args.split}) - gercek sinif: {CLASS_NAMES[label]} | 4 dogru + 4 yanlis, seed 42")
            outputs.append(str(path))
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
