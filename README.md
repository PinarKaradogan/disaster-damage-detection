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
| `src/evaluation/`    | Metrics, Grad-CAM                                    |
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

## Team

- Pınar Karadoğan
- Bahar Toslak
