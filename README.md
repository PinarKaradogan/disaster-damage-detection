# Disaster Damage Detection

Deep learning based post-disaster building damage detection from pre- and post-disaster satellite imagery (xBD dataset). Course project for _Applied Neural Networks_.

Buildings are classified into four damage levels: no damage, minor damage, major damage, destroyed. Models compared: simple CNN, ResNet, Siamese ResNet, Vision Transformer and U-Net.

## Project structure

| Folder               | Content                                              |
| -------------------- | ---------------------------------------------------- |
| `data/`              | xBD dataset (local only, not committed)              |
| `notebooks/`         | Exploration and experiment notebooks                 |
| `src/preprocessing/` | Patch extraction, data splits, Dataset classes       |
| `src/models/`        | Model definitions                                    |
| `src/training/`      | Classifier and U-Net training engines, losses, comparison |
| `src/evaluation/`    | Run audits, segmentation metrics, two-stage pipeline |
| `results/`           | Figures and tables (model weights are not committed) |

## Setup

1. Create a Python 3.13 environment (venv or conda).
2. Install dependencies (requires an NVIDIA GPU with CUDA 13.2 support):

```bash
   pip install -r requirements.txt
```

3. Verify the environment:

```bash
   python src/check_environment.py
```

## Dataset

Download xBD from [xview2.org](https://xview2.org) (registration required) and extract the archives into `data/`. The dataset is licensed CC BY-NC-SA; see the xBD paper (Gupta et al., 2019).

## Phase 4 training

See [epoch-boundary resume training](docs/resume-training.md) for `--resume`,
`--session-epochs`, checkpoint recovery, and reproducibility limits.

For interrupted checkpoints, see [validation-only evaluation](docs/validation-evaluation.md).

## Phase 5: advanced models and building localization

Results and design decisions: [results/phase5_report.md](results/phase5_report.md). All commands use train/validation
only; the test split is reserved for Phase 6.

```bash
# Siamese damage classifiers (same training engine and policy as Phase 4)
python -m src.train --model siamese_resnet34 --loss focal --experiment-name <name> --num-workers 4
python -m src.train --model siamese_vit_small --loss focal --experiment-name <name> --num-workers 4

# U-Net building segmentation on raw 1024x1024 PRE images (data/train/images + targets)
python -m src.train_segmentation --experiment-name <name>          # add --resume to continue

# Audits (read-only; re-evaluate best.pth on validation)
python -m src.evaluation.audit_run --experiment <classifier run> --output results/phase5_audit/audit_<name>.json
python -m src.evaluation.audit_segmentation --experiment <unet run> --output results/phase5_audit/audit_<name>.json

# Two-stage system: U-Net localization -> damage classifier
python -m src.evaluation.two_stage predict --image-id <val image id> --threshold 0.9 --split-erosion 4 --output <figure.png>
python -m src.evaluation.two_stage evaluate-val --threshold 0.9 --split-erosion 4 --output <report.json>
```

## Demo (Phase 7)

A local web demo of the two-stage system (no extra packages; Python's standard-library HTTP server):

```bash
python -m src.demo_app
```

Open http://127.0.0.1:7860, pick a validation example or upload a PRE/POST image pair (same size, sides a
multiple of 32; xBD images are 1024×1024) and get the damage map. Settings are the ones frozen in the
Phase 6 protocol (U-Net threshold 0.9, split erosion 4, ResNet34 early fusion classifier). The server listens
on 127.0.0.1 only and needs the local checkpoints (`results/phase5_unet_resnet34_seed42_w4/best.pth`,
`results/phase4_final_resnet34_focal_seed42_w4/best.pth`).

## Team

- Pınar Karadoğan
- Bahar Toslak
