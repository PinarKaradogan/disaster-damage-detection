# Phase 4: train/validation experiment plan

Status: implementation and short smoke verification only. Full experiments need
Pinar's approval. Smoke scores are pipeline checks, not model-selection evidence.
No imbalance strategy has been selected and no final validation result exists yet.

## Fixed policy

- Existing `data/processed/patches_full/patches.csv`; train/val only. Never use test
  scores or test label frequencies for weights, tuning, or model selection.
- Seed 42, input 128x128, four classes in `src/data/constants.py` order.
- Identical deterministic ImageNet RGB normalization for PRE and POST.
- Train-only `PairedRandomGeometric`: flips 0.5/0.5, rotation +/-15 degrees.
- AdamW: learning rate 3e-4, weight decay 1e-4.
- CosineAnnealingLR: T_max 30, eta_min 1e-6; at most 30 epochs.
- Early stopping: five consecutive epochs without strictly higher val macro F1.
- Best checkpoint: maximum val macro F1; ties retain the earlier checkpoint.
- CUDA AMP, initial batch size 32, num_workers 0. These are shared starting
  conditions, not optimized hyperparameters. If OOM occurs, lower batch size;
  retain input size and explicitly record any batch-size difference.
- Inverse-frequency class weights normalized to mean one, computed using all
  train labels only. Four train classes must be present. Focal gamma is 2.0
  without alpha. Oversampling uses replacement and standard unweighted CE.
- Simple CNN: three convolution blocks, early fusion. ResNet34: timm pretrained
  ImageNet backbone, four-class fresh head. The RGB first-conv weights are
  repeated twice and multiplied by 0.5 for six channels. Loading errors propagate;
  there is no silent random-weight fallback. CPU/CI tests use pretrained=False.

The AMP API follows the [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
The pretrained adaptation uses [timm's model loading](https://huggingface.co/docs/timm/main/reference/models);
the installed `adapt_input_conv` and its invocation in `load_pretrained` are also
checked locally. Seeding sets RNG states; bit-identical GPU runs are not promised.
`PYTHONHASHSEED` set at runtime applies to future processes, not existing hashes.

## Commands after full-training approval

Use the repo virtual environment's Python. Run these sequentially, not together
on the 6 GB GPU. Each experiment directory must be new; existing results are
never overwritten. Maximum epoch count does not predict wall-clock duration;
two-batch smoke timing is not a reliable full-dataset runtime estimate.

```powershell
.\.venv\Scripts\python.exe -m src.train --model simple_cnn --loss weighted_ce --experiment-name phase4_cnn_weighted_ce_seed42
.\.venv\Scripts\python.exe -m src.train --model simple_cnn --loss focal --experiment-name phase4_cnn_focal_seed42
.\.venv\Scripts\python.exe -m src.train --model simple_cnn --loss ce --sampler weighted --experiment-name phase4_cnn_sampler_ce_seed42

.\.venv\Scripts\python.exe -m src.training.compare --kind imbalance --runs results/phase4_cnn_weighted_ce_seed42 results/phase4_cnn_focal_seed42 results/phase4_cnn_sampler_ce_seed42 --output results/phase4_imbalance_validation.csv
```

Read the selected `loss` and `sampler` from `phase4_imbalance_validation.json`.
Select only by best validation macro F1; ties use the listed order. Retain that
selection rationale and the three runs. Then run the final two experiments with
the selected arguments (replace the placeholders; these are not literal commands):

```text
python -m src.train --model simple_cnn --loss <selected_loss> --sampler <selected_sampler> --experiment-name phase4_final_cnn_seed42
python -m src.train --model resnet34_early_fusion --loss <selected_loss> --sampler <selected_sampler> --experiment-name phase4_final_resnet34_seed42
python -m src.training.compare --kind models --runs results/phase4_final_cnn_seed42 results/phase4_final_resnet34_seed42 --output results/phase4_validation_summary.csv
```

Five full runs are planned, each at batch size 32 and at most 30 epochs:

| Experiment | Local best checkpoint |
| --- | --- |
| CNN weighted CE | `results/phase4_cnn_weighted_ce_seed42/best.pth` |
| CNN focal | `results/phase4_cnn_focal_seed42/best.pth` |
| CNN sampler + CE | `results/phase4_cnn_sampler_ce_seed42/best.pth` |
| Final CNN | `results/phase4_final_cnn_seed42/best.pth` |
| Final ResNet34 | `results/phase4_final_resnet34_seed42/best.pth` |

Each run writes versioned `config.json`, `metrics.json`, `status.json`,
`history.csv`, local `best.pth`, and `runs/<experiment_name>/` TensorBoard events.
Configuration records the manifest SHA256, full split sizes, train frequencies,
effective batch/epoch count, normalization, augmentation, device, versions,
pretrained source, parameter count and policy. Metrics describe the best
validation epoch, not the final epoch. Checkpoints support inference reload,
not exact interrupted-run resumption.

`--smoke` uses at most two epochs and seeded subsets of at most two batches per
split (default batch size 8). It still derives class weights from the complete
train split. It logs `smoke=true` and subset sizes; comparison refuses smoke runs.
`--batch-size` overrides the smoke batch size as well when explicitly supplied.

Checkpoint weights, TensorBoard runs, raw and processed data remain ignored.
Small metadata and validation summaries are reviewable artifacts; no artifacts
are automatically staged, committed, or pushed. Phase 3 code and persisted
split files are unchanged. Full validation summaries must not be produced from
the current smoke scores.
