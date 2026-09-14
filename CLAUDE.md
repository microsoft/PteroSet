# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository. For setup, commands, and pipeline overview see [README.md](README.md). This file only captures Claude-specific context.

## Project Overview

Bird vocalization detection and classification on the [PteroSet](https://zenodo.org/records/19137071) dataset (5 projects: MAP1, PPA1–PPA4). Spectrogram-based deep learning built on **PytorchWildlife** with **PyTorch Lightning**.

## Environment

- Python managed with **conda**. Activate env `bioacoustics` before running anything (`conda activate bioacoustics`).
- Detailed rules auto-loaded from `.claude/rules/`. Agents in `.claude/agents/`. Skills in `.claude/skills/`.

## Identity

Senior engineer and AI/ML researcher. Engineering rigor, scientific discipline, cite evidence for claims.

## Architecture Notes (not in README)

### Key Code Entry Points
| File | Purpose |
|------|---------|
| `train.py` | `SpectrogramDataModule`, `train_single_fold()` |
| `prepare_dataset.py` | Orchestrates windows, segmentation, spectrograms, splits |
| `data/data_reader.py` | Parses RAVEN labels into COCO annotations |
| `plot_cv_results.py` | Loads checkpoints, evaluates folds, plots PR curves |

### Dataset Versions
Each version has a paired `windows_mapping_*.json` and `folds_*/` directory. Results and trained weights live in matching `outputs_v{N}/` and `checkpoints_v{N}/`.

| Version | Windows JSON | Folds dir | Outputs | Checkpoints | Total windows | Positives |
|---------|--------------|-----------|---------|-------------|---------------|-----------|
| overlaped (original) | `windows_mapping_4.0overlap.json` | `data/folds_overlaped/` | — | — | — | — |
| segmented (v1) | `windows_mapping_4.0overlap_segmented.json` | `data/folds_segmented/` | `outputs_v1/` | `checkpoints_v1/` | 157 004 | 33 009 |
| segmented_v2 | `windows_mapping_4.0overlap_segmented_v2.json` | `data/folds_segmented_v2/` | `outputs_v2/` | `checkpoints_v2/` | 157 004 | 34 619 |
| segmented_v3 | `windows_mapping_4.0overlap_segmented_v3.json` | `data/folds_segmented_v3/` | `outputs_v3/` | `checkpoints_v3/` | 160 244 | 33 463 |
| segmented_v4 | `windows_mapping_4.0overlap_segmented_v4.json` | `data/folds_segmented_v4/` | `outputs_v4/` | `checkpoints_v4/` | 160 244 | 35 073 |
| **segmented_v5 (current corrected dataset)** | `windows_mapping_4.0overlap_segmented_v5.json` | `data/folds_segmented_v5/` | `outputs_v5/` | `checkpoints_v5/` | 162 066 | 35 103 |

**What changed at each step** (verified by diffing the JSONs on physical window identity `(sound_id, start, end)`):
- **overlaped → segmented (v1)**: drops windows that cross 10-second segment boundaries (PteroSet audio is 48 × 10 s concatenated time-lapse segments, so a window spanning two segments mixes acoustic contexts).
- **v1 → v2**: missing PPA4 annotations were added. Same 157 004 window set; 1 615 PPA4 windows flipped 0→1, 5 flipped 1→0 (net +1 610 positives). `dataset` field also added so each entry now carries its project (MAP1/PPA1–PPA4).
- **v2 → v3**: two intended changes plus one likely regression.
  1. **PPA1 segments recomputed**: PPA1 time-lapse files use a 1 s crossfade between 10 s segments (segment stride = 9 s, file duration = 433 s vs. 480 s for other projects; see paper §Data Records), so the fixed 10 s-multiple boundary assumption was wrong for PPA1. Re-segmenting removes 9 396 old PPA1 windows and adds 12 636 new ones (net +3 240). No label flips on PPA1 windows that survive re-segmentation.
  2. **Fold directory renamed** to publication codes (MAP1, PPA1–PPA4). Mapping: `PAREX` → `MAP1` (now fold_0), `GEOPARK_PUTUMAYO_2024` → `PPA1` (fold_1), `GeoPark_II_T2_2024` → `PPA2` (fold_2), `GeoPark_II_T3_2025` → `PPA3` (fold_3), `GeoPark_II_T4_2025` → `PPA4` (fold_4). Every project shifts +1 fold index except PAREX which moves from fold_4 to fold_0.
  3. **PPA4 annotation regression** (unintended — to be re-fixed in v4): the v2 PPA4 fix did not carry over. All 34 280 PPA4 windows in v3 match v1 labels exactly (v1 positives = v3 positives = 7 334 vs. v2 positives = 8 944). Specifically, the exact same 1 615 PPA4 windows that were flipped 0→1 in v1→v2 are flipped 1→0 in v2→v3. MAP1, PPA2, PPA3 labels are untouched. v3 was already trained (`checkpoints_v3/`, `outputs_v3/`) before the regression was detected, so those v3 results are on buggy PPA4 labels.
- **v3 → v4**: PPA4 annotation fix re-applied via a pipeline patch (Option B). Root cause was a cache shortcut in `run_windows()` that bypassed label-regeneration from current annotations; fix moved label derivation into `run_segment_windows()` so every new segmented version reflects the annotations JSON at time-of-derivation regardless of upstream cache state. Same 160 244 window set as v3; labels change in PPA4 only (1 615 flips 0→1, 5 flips 1→0). `prepare_dataset.py` gained a `--version` CLI flag; `train.py` gained `--ckpt_dir`. See `docs/implementation/v4-ppa4-fix/` for root cause analysis (`phase1_findings.md`), plan (`plan.md`), and training/evaluation results (`results.md`).
- **v4 → v5**: segmented windows are generated directly from current annotation sound durations and project-aware segment geometry instead of filtering the stale recording-level `windows_mapping_4.0overlap.json`. This recovers 1 822 valid tail windows. The total is 162 066 rather than 162 144 because five short PPA4 recordings provide 13 fewer complete segments (78 windows). The default `prepare_dataset.py --version` is `v5`; the CLI rejects v1-v4 as historical output destinations and writes `windows_mapping_4.0overlap_segmented_v5.json` plus `folds_segmented_v5/`. Annotation-driven segmented generation supports only the `sliding` window strategy. See `docs/implementation/segment-manifest/README.md`.

**Current dataset target**: v5. Preserve v1-v4 artifacts, but rebuild the
v5 mapping, required spectrograms, `folds_segmented_v5/`, and v5
training/evaluation artifacts as one consistent revision. Existing
`outputs_v4/` and `checkpoints_v4/` are historical results trained on 160 244
windows and exclude the 1 822 recovered windows; do not report them as
corrected v5 results. v3 contains buggy PPA4 labels, and v2 predates the PPA1
crossfade correction.

### Model Options (train.py flags)
- `--backbone`: `resnet18` (default), `resnet34`, `resnet50`
- `--num_classes`: `2` for binary (Birds/No Birds), or N for multiclass species
- `--use_specaug` for SpecAugment; MixUp enabled automatically for binary
- `--freeze_backbone early` with `--backbone_lr_ratio` for transfer learning

### Cross-Validation
Leave-one-project-out: each fold holds out 1 of 5 projects; remaining 4 split into train/val. Prevents leakage across recording projects.

## Data Artifacts (gitignored)
Audio files, spectrograms, annotation JSONs, fold CSVs, and model checkpoints are not tracked in git. The `spectrograms/` directory stores precomputed `.npy` files used during training.
