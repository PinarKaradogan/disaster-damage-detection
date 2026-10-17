"""Phase 4 loaders; the Phase 3 Dataset and persisted splits stay unchanged."""
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data.dataset import DamagePatchDataset, PairedRandomGeometric, ROOT
from src.training.sampling import class_weights, make_weighted_sampler, train_labels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainingData:
    train: DataLoader
    val: DataLoader
    class_weights: torch.Tensor
    train_class_counts: list[int]
    split_counts: dict[str, int]


def deterministic_transform(input_size=128):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loaders(config, manifest_path, *, sampler="none", smoke=False, project_root=ROOT):
    if sampler not in {"none", "weighted"}:
        raise ValueError("Unknown sampler")
    transform = deterministic_transform(config.input_size)
    train = DamagePatchDataset(
        manifest_path, split="train", paired_transform=PairedRandomGeometric(),
        transform=transform, project_root=project_root,
    )
    val = DamagePatchDataset(
        manifest_path, split="val", transform=transform, project_root=project_root,
    )
    labels = train_labels(train)
    weights = class_weights(labels, split="train")
    counts = torch.bincount(labels, minlength=4).tolist()
    split_counts = {"train": len(train), "val": len(val)}
    batch_size = config.smoke_batch_size if smoke else config.batch_size
    # Smoke uses seeded bounded subsets; never presents its scores as full validation.
    if smoke:
        generator = torch.Generator().manual_seed(config.seed)
        indices = torch.randperm(len(train), generator=generator)[:2 * batch_size]
        train = Subset(train, indices.tolist())
        # Retain full-train weights, even when a smoke subset omits a class.
        sample_weights = weights[labels[indices]]
        val_indices = torch.randperm(len(val), generator=generator)[:2 * batch_size]
        val = Subset(val, val_indices.tolist())
    else:
        sample_weights = weights[labels]
    weighted_sampler = None
    if sampler == "weighted":
        weighted_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(train), replacement=True,
            generator=torch.Generator().manual_seed(config.seed),
        ) if smoke else make_weighted_sampler(labels, split="train", seed=config.seed)
    common = dict(batch_size=batch_size, num_workers=config.num_workers)
    train_loader = DataLoader(
        train, shuffle=weighted_sampler is None, sampler=weighted_sampler,
        generator=torch.Generator().manual_seed(config.seed), **common,
    )
    val_loader = DataLoader(val, shuffle=False, **common)
    return TrainingData(train_loader, val_loader, weights, counts, split_counts)
