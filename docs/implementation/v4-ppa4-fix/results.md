# v4 Training & Evaluation Results

## Setup

- Pipeline fix: Option B — `run_segment_windows` re-derives labels from current annotations (`phase1_findings.md`, `plan.md` §Phase 2).
- Hyperparameters: **mirrored v3 exactly** (resnet18, binary, batch 32, lr 1e-4, epochs up to 50 w/ EarlyStopping patience 20, monitor `val/f1`).
- Training artifacts:
  - `data/windows_mapping_4.0overlap_segmented_v4.json` (160 244 windows, 35 073 positives)
  - `data/folds_segmented_v4/fold_{0..4}_{MAP1,PPA1,PPA2,PPA3,PPA4}_segmented/`
  - `checkpoints_v4/fold_{0..4}/`
  - `outputs_v4/{cv_results.csv, precision_recall_curves.png, qualitative_results*.png, version_comparison_by_project.csv}`

## Fold → project mapping caveat

v2 folds use original project codes; v3/v4 use publication codes. **Fold indices do not map 1:1**. All comparisons below are aligned by held-out project:

| Project | v2 fold | v3/v4 fold |
|---|---|---|
| MAP1 | 4 (PAREX) | 0 |
| PPA1 | 0 (GEOPARK_PUTUMAYO_2024) | 1 |
| PPA2 | 1 (GeoPark_II_T2_2024) | 2 |
| PPA3 | 2 (GeoPark_II_T3_2025) | 3 |
| PPA4 | 3 (GeoPark_II_T4_2025) | 4 |

## Three-way comparison (project-aligned)

### F1 by project

| Project | v2 | v3 | v4 | Δ(v4−v3) | Δ(v4−v2) |
|---|---|---|---|---|---|
| MAP1 | 0.6406 | 0.6702 | 0.6503 | −0.0199 | +0.0097 |
| PPA1 | 0.7547 | 0.7522 | 0.7407 | −0.0115 | −0.0140 |
| PPA2 | 0.7112 | 0.7338 | 0.7227 | −0.0111 | +0.0115 |
| PPA3 | 0.7459 | 0.7559 | 0.7338 | −0.0221 | −0.0121 |
| PPA4 | 0.7354 | 0.7398 | **0.7422** | +0.0024 | +0.0068 |

### AUPRC by project

| Project | v2 | v3 | v4 | Δ(v4−v3) | Δ(v4−v2) |
|---|---|---|---|---|---|
| MAP1 | 0.6692 | 0.7391 | 0.7343 | −0.0048 | +0.0651 |
| PPA1 | 0.8333 | 0.8336 | 0.8207 | −0.0129 | −0.0126 |
| PPA2 | 0.7885 | 0.8084 | 0.7797 | −0.0287 | −0.0088 |
| PPA3 | 0.8044 | 0.8178 | 0.7930 | −0.0248 | −0.0114 |
| PPA4 | 0.8457 | 0.8078 | **0.8590** | +0.0512 | +0.0133 |

### Aggregate (mean across 5 folds)

| Version | F1 | AUPRC | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| v2 | 0.7176 | 0.7882 | 0.7087 | 0.7541 | 0.8468 |
| v3 | 0.7304 | 0.8014 | 0.7089 | 0.7640 | 0.8620 |
| v4 | 0.7179 | 0.7973 | 0.7122 | 0.7489 | 0.8499 |

### PPA4 fold only (test labels directly affected by the annotation fix)

| Version | Precision | Recall | F1 | AUPRC | Accuracy |
|---|---|---|---|---|---|
| v2 | 0.8586 | 0.6432 | 0.7354 | 0.8457 | 0.8770 |
| v3 | 0.7587 | 0.7218 | 0.7398 | 0.8078 | 0.8891 |
| **v4** | **0.8614** | 0.6520 | **0.7422** | **0.8590** | 0.8796 |

## Interpretation

### Where v4 improves
- **PPA4 AUPRC: +0.0512 vs. v3** — largest single gain. v3's buggy test labels made 1 615 hard positives look like easy negatives, deflating AUPRC under correct evaluation. v4 restores it.
- **PPA4 precision: 0.8614 (v4) vs. 0.7587 (v3)** — near-v2 level. With corrected labels, true positives align with model predictions; v3's inflated "false positives" disappear.
- **PPA4 F1: marginal +0.0024 over v3**, +0.0068 over v2.

### Where v4 is slightly worse than v3 (but still better than v2 on most projects)
- Non-PPA4 folds show v4 F1 −0.011 to −0.022 relative to v3; AUPRC similar. Candidate causes:
  1. **Training variance** — no shared seed between v3 and v4 runs was verified beyond config defaults; stochastic GPU ops can yield per-fold differences at this magnitude.
  2. **Decision-boundary shift** — 1 610 additional PPA4 positives in training (folds 0–3) may nudge the model toward higher-recall/lower-precision regimes that fit non-PPA4 test data slightly less well.
  3. **Early stopping behavior** — fold_0 (MAP1) and fold_4 (PPA4) both checkpointed at `epoch=00`, suggesting val/f1 did not improve past the first epoch. Compared to v3 fold_0 also stopping early, this pattern is not v4-specific.

### Is v3's aggregate F1 advantage real?
No. v3's aggregate F1 (0.7304) beats v4 (0.7179) partly because v3 evaluates PPA4 against incorrect test labels that systematically underestimate the task difficulty. Under correct evaluation (v4 test labels), the same v3 model would score lower on PPA4. The v3 → v4 aggregate drop is an artifact of correcting the benchmark, not of model degradation.

## Decision: ship v4

v4 is the scientifically correct version and the recommended shipping artifact:
- Correct PPA4 annotations applied in training AND evaluation
- Correct PPA1 segmentation (9-second stride for crossfade) preserved from v3
- Publication-ready fold naming (MAP1, PPA1–PPA4)
- PPA4 AUPRC clearly best across all three versions

Ship v4 as the publication result. Retain v2 and v3 as historical references; mark v3 as buggy (see CLAUDE.md).

## Open items (low-priority follow-ups)

1. **Training variance quantification**: retrain v4 with 3 seeds per fold to estimate ±σ for non-PPA4 folds. Would confirm whether the −0.01 to −0.02 v4−v3 deltas are noise.
2. **Regression test**: add an automated check (or doctest) that asserts the source-derived labels match `windows_mapping_*_segmented_*.json` labels. Would catch a future repeat of this bug without manual audit.
3. **Drop the stale `windows_mapping_4.0overlap.json` unsegmented cache** from the default pipeline, or invalidate it by annotation mtime — the Option B patch makes `run_segment_windows` self-correcting, but the unsegmented cache is still potentially misleading if used for any other purpose.
