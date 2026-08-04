# Perch v2 Species Linear Probe: Technical Validation Results

**Date**: 2026-07-29  
**Status**: COMPLETE  
**Task**: multilabel species classification with true no-bird windows represented by all-zero targets

## Why

This experiment tests whether frozen Perch v2 embeddings contain linearly separable species
information for PteroSet. It is an in-distribution technical validation over the task-specific
sound-grouped split, not an unseen-site or unseen-project generalization result.

## What

| Component | Value |
|---|---|
| Encoder | Perch v2, frozen |
| Embedding | 1,536 dimensions |
| Classifier | One-vs-rest logistic regression |
| Class weighting | `balanced` |
| Solver | `lbfgs`, `max_iter=1000` |
| Parallelism | 16 species fits, BLAS threads limited to 1 |
| Classes | 68 |
| Train / val / test rows | 95,536 / 6,398 / 6,474 |
| C grid | 0.01, 0.1, 1.0, 10.0 |
| Selection metric | Validation macro average precision |
| Selected C | **0.1** |

Training uses the augmented train split; validation and test contain canonical non-overlapping
windows only. Embeddings are L2-normalized independently per split before fitting or prediction.

## C Selection

| C | Val macro-AP | Val macro-AP (`val_pos >= 5`) | Val macro-AP (`val_pos >= 10`) | Non-converged |
|---:|---:|---:|---:|---:|
| 0.01 | 0.3959 | 0.4889 | 0.5062 | 0 |
| **0.1** | **0.4379** | 0.5106 | 0.5470 | 0 |
| 1.0 | 0.4282 | 0.5114 | **0.5497** | 0 |
| 10.0 | 0.4302 | 0.5074 | 0.5435 | 0 |

The pre-registered selection rule uses all validation-evaluable species, so `C=0.1` is selected.
The support-restricted columns are sensitivity summaries, not alternative selection rules.

## Test Results

| Metric | Result |
|---|---:|
| Macro average precision, all 68 species | **0.4089** |
| Macro prevalence-only AP baseline | 0.0020 |
| Improvement over prevalence baseline | **+0.4070** |
| Macro AP, species with `test_pos >= 5` (44 species) | **0.4817** |
| Macro AP, species with `test_pos >= 10` (25 species) | **0.5738** |
| Any-bird vs no-bird AUROC | **0.9437** |
| Any-bird vs no-bird AP | **0.7133** |

The observed improvement over the prevalence-only baseline is 0.4070. This is direct evidence that
the frozen embedding carries species-discriminative signal on the approved technical-validation
split; no additional post-hoc threshold is needed to make that statement.

### Per-Species Examples

Highest AP:

| Species code | AP |
|---|---:|
| QUEPUR | 1.0000 |
| ATAPIL | 0.8817 |
| ORTGAR | 0.8667 |
| COLSQU | 0.7946 |
| TOLFLA | 0.7631 |

Lowest AP:

| Species code | AP |
|---|---:|
| MESCAY | 0.0007 |
| CROANI | 0.0025 |
| MELCRU | 0.0061 |
| CYCGUJ | 0.0130 |
| DRYLIN | 0.0152 |

The complete per-species table, including train/validation/test support, is stored in
`checkpoints/perch/species_v1/species_ap.csv`.

## Diagnostics

- All 68 species were trainable, validation-evaluable, and test-evaluable.
- All 68 final estimators produced finite coefficients.
- No estimator reached `max_iter`; no convergence failure was detected.
- The four-grid sweep required 66.6 seconds of aggregate fit time; the final refit required
  16.8 seconds with `n_jobs=16` in the final source-matched run.
- Test probabilities are finite and have shape `(6,474, 68)`.

## Artifacts

Derived artifacts are gitignored:

```text
data/embeddings/perch_v2/species_v1/
  embedding_manifest.json
  train_emb.npz
  val_emb.npz
  test_emb.npz

checkpoints/perch/species_v1/
  c_selection.csv
  logreg_species.joblib
  macro_ap_summary.csv
  no_bird_detection.csv
  species_ap.csv
  species_diagnostics.csv
  test_predictions.npz
  training_manifest.json
```

`test_predictions.npz` stores probabilities and any-bird scores as float64 so every published AP and
AUROC value can be reproduced exactly from the persisted prediction artifact.

## Limitations

1. This is an in-distribution technical validation. `sound_id` is disjoint across splits, but
   recording sites and projects are not held out.
2. Test support is thin for some species: 24 of 68 have fewer than five positive test windows, and
   43 have fewer than ten. A species with one positive window can obtain an unstable AP estimate,
   including an apparent AP of 1.0.
3. Species annotations are not exhaustive absence labels. Per-species negatives mean "not annotated
   as this species" within the approved label rules.
4. The training split contains overlapping train-only augmentation, whereas validation and test use
   canonical non-overlapping windows.
5. The producing repository is currently dirty/uncommitted, so manifests use content hashes and
   `git_dirty=true` rather than a commit SHA. The user owns the commit step.

The Phase 3 trainer passed 97 focused tests; the complete repository suite passed 263 tests.

## Conclusion

Frozen Perch v2 embeddings support a strong linear species-classification baseline on PteroSet. The
headline macro-AP is substantially above the prevalence baseline, and performance improves as the
minimum test support increases. This validates the dataset and embedding pipeline for species
classification, while leaving cross-site/project generalization and rare-species reliability as
separate follow-on questions.

The relationship between per-species AP and train/validation/test representation is analyzed in
[`ap_support_analysis.md`](ap_support_analysis.md).
