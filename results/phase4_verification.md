# Phase 4 implementation verification — 2026-10-17

Status: **READY FOR FULL EXPERIMENTS**. No full training or final validation
comparison has been run. No imbalance strategy has been selected.

## Code and environment

- Branch: `feature/faz4-training-baselines`; `712ca24` is an ancestor of HEAD.
- Foundation files were already correct and were preserved without rewriting.
- Final full pytest: **146 passed** (114 existing, 32 added); CPU synthetic
  model/training tests, with no pretrained downloads in unit tests.
- Targeted stages: foundation 5; imbalance/metrics 5; paired models 5;
  loaders 3; engine/CLI/artifacts 8; comparison 6. All PASS after corrections below.
- `python -m compileall src`: PASS.
- Environment: Python 3.13.7, PyTorch 2.14.1+cu132, CUDA 13.2,
  NVIDIA GeForce RTX 4050 Laptop GPU. timm 1.0.30.
- Existing CI already installs timm/tensorboard and runs compileall, pytest,
  environment check. Workflow unchanged. Hosted CI itself was not run here.
- Dataset implementation, manifests and persisted splits unchanged. Full train
  count 134028; val 25615. No test samples were loaded for Phase 4 training,
  validation, weighting, or selection. Manifest checksum reads the file bytes
  for provenance; it does not use the test distribution for any decision.

## Local GPU smoke (not performance evidence)

Both runs used seed 42, 128x128, batch size 8, workers 0, CUDA AMP,
weighted CE from complete train frequencies, two epochs, and seeded subsets
of 16 train / 16 val samples. Forward/backward, AdamW, cosine scheduler,
validation metrics, best checkpoint, JSON/CSV and TensorBoard completed.
No OOM occurred. A full batch size of 32 has not been validated by these smokes.

| Run | Best smoke epoch | Smoke val macro F1 | Parameters |
| --- | --- | --- | --- |
| `phase4_smoke_simple_cnn_seed42` | 1 | 0.0 | 94852 |
| `phase4_smoke_resnet34_seed42` | 1 | 0.19230769230769232 | 21296132 |

**These tiny-subset scores must not be used to choose a model or strategy.**
The comparison command rejects runs marked smoke. Full validation reports
remain pending approval and completion of full experiments.

Actual GPU checkpoints were reloaded on CPU with `weights_only=True`, and
TensorBoard scalar tags were checked for both runs. Pretrained ResNet loading
used `pretrained=True`, with no random-weight fallback. A separate offline
check compared its six-channel first conv against the downloaded three-channel
checkpoint: exact equality to RGB weights repeated twice and scaled by 0.5.
See `phase4_pretrained_verification.json`.

## Failures and corrections

1. The first focal gamma=0 equivalence test passed a gradient tensor to
   `pytest.approx`, triggering a NumPy conversion error. Changed the test to
   compare the scalar `.item()`. Targeted tests and final full suite PASS.
2. Initial loader tests could not access pytest's existing temporary directory
   under the sandbox (`PermissionError`, WinError 5). Permission-approved pytest
   reruns outside the sandbox PASS; no repository permission changes were made.
3. First ResNet GPU smoke failed with WinError 10013 / LocalEntryNotFoundError:
   sandbox networking was blocked and weights were not cached. Permission-approved
   rerun downloaded the public checkpoint and the actual pretrained smoke PASS.
4. Hugging Face emitted unauthenticated-download and Windows symlink-cache
   warnings. Loading/training succeeded; no token or system settings changed.
5. Sandboxed `git status` warns that `.pytest_cache/` cannot be opened. It still
   reports the new files successfully; cache permissions were left unchanged.

## Git safety

- `git diff --check`: PASS; untracked text files were separately checked for
  trailing whitespace because Git's default diff excludes them.
- Tracked files unchanged; staging area empty. No commits or pushes.
- New files: Phase 4 model/training/CLI code, six test modules, this report,
  experiment plan, pretrained verification JSON, and small smoke metadata.
- `git ls-files` found no raw/processed data, patch PNGs, `.pth`/`.pt` weights,
  or TensorBoard runs. `git check-ignore` confirms the actual smoke checkpoints,
  TensorBoard directories and processed manifest are ignored.
- No final `phase4_validation_summary.csv` was fabricated from smoke results.

See `phase4_experiment_plan.md` for the five proposed full runs, commands,
batch/epoch limits, policy and checkpoint paths. Approval is required before
starting those potentially long GPU experiments.
