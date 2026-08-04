# Phase 3 Data Validation Report: Perch v2 Species Linear Probe (`checkpoints/perch/species_v1`)

**Validator**: independent data-validator agent (read-only; no artifacts edited)
**Reviewed against**: `docs/design/perch2_species_linear_probe_plan.md` (STATUS: FINAL) and
`docs/implementation/species-linear-probe-v1/results.md`
**Review window**: 2026-07-29 18:24–19:06 UTC
**Method**: independent recomputation from raw artifacts (embeddings, split CSVs, saved model),
not re-derivation of the reviewed script's own printed numbers.

**Final snapshot re-verified after the precision fix** (probability/`any_bird_score` storage moved
from `float32` to `float64`, and `training_manifest.json` gained an `embedding_npz_sha256` field
hashing the three embedding NPZs directly). Hashes below are confirmed stable across two checks
~110 seconds apart (19:04:01 and 19:05:51 UTC), and `training_manifest.json.script_source_sha256`
now matches the current `train_perch_logreg.py` on disk (`58b6b5ac...`) — the directory and the
script that produced it are mutually consistent at this snapshot.

---

## 1. Manifest hash chain — every recorded hash re-verified against its physical file

| Reference | Recorded | Actual | Result |
|---|---|---|---|
| `class_list_sha256` | `6640aada...53f86` | `class_list.json` | OK |
| `embedding_manifest_sha256` | `2fef6254...5762b94` | `embedding_manifest.json` | OK |
| `embedding_npz_sha256.{train,val,test}` | 3 hashes | `{train,val,test}_emb.npz` | OK (all 3) |
| `embedding_manifest.json.identity_hash_inputs.split_csv_sha256.{train,val,test}` | 3 hashes | `{train,val,test}_split.csv` | OK (all 3) |
| `embedding_manifest.json.identity_hash_inputs.split_manifest_sha256` | 1 hash | `split_manifest.json` | OK |
| `artifact_hashes.{macro_ap_summary,no_bird_detection,species_ap,species_diagnostics}.csv` | 4 hashes | physical files | OK (all 4) |
| `artifact_hashes.{c_selection.csv, test_predictions.npz, logreg_species.joblib}` | 3 hashes | physical files | OK (all 3) |

Every hash in the chain matches at this snapshot. **Verified.**

## 2. C selection — recomputed independently

Refit `OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced"))`
from scratch on L2-normalized `train_emb.npz`/`val_emb.npz` for the full grid `{0.01, 0.1, 1.0, 10.0}`:

| C | `c_selection.csv` | Independent refit | Match |
|---:|---:|---:|---|
| 0.01 | 0.3959136671 | 0.3959136671 | exact |
| **0.1** | **0.4378809928** | **0.4378809928** | exact |
| 1.0 | 0.4281892577 | 0.4281892577 | exact |
| 10.0 | 0.4301725914 | 0.4301719400 | matches to 5 decimals (BLAS-thread noise) |

`C*=0.1` is the unambiguous argmax; matches `selected_c` and `logreg_species.joblib.metadata.model_config.C`. **Verified.**

## 3. AP / prevalence baselines, macro and thin-support metrics

Recomputed from `test_predictions.npz` (now `float64` probabilities) and independently confirmed by a
second, fully-from-scratch refit (train on raw `train_emb.npz` at `C=0.1`, predict on raw `test_emb.npz`):

| Metric | Artifact | Recomputed (stored predictions) | Independent refit | Match |
|---|---:|---:|---:|---|
| Macro AP, all 68 species | 0.4089036643 | 0.4089036643 | 0.4089036643 | exact (all three) |
| Macro prevalence-only baseline | 0.0019535154 | 0.0019535154 | 0.0019535154 | exact |
| Delta vs. baseline | 0.4069501489 | 0.4069501489 | 0.4069501489 | exact |
| Macro AP, `test_pos>=5` (n=44) | 0.4817116150 | 0.4817116150 | — | exact |
| Macro AP, `test_pos>=10` (n=25) | 0.5738379434 | 0.5738379434 | — | exact |

Per-species AP (`species_ap.csv`) matches the raw-prediction recomputation to `9.5e-17` max abs
difference across all 68 species (float64 vs. float64 now — the earlier float32 storage round-trip
noise is gone). Thin-support disclosure confirmed: 24/68 species `test_pos<5`, 43/68 `test_pos<10`.
**Go/no-go criterion (macro-AP beats per-species prevalence-only baseline by >=0.05) is met by 8x
the required margin (+0.407).** **Verified.**

## 4. No-bird / any-bird detection metrics

Recomputed AUROC = 0.9436529213, AP = 0.7133337315 from `float64` `any_bird_score` against
`y_any = (target_vector.sum(axis=1) > 0)`; matches `no_bird_detection.csv` exactly.
`n_positive=674`, `n_negative=5800` — consistent with `species_distribution.csv`'s test-split
no-bird fraction and `split_manifest.json`'s global canonical no-bird prevalence (0.9096).
**Verified.**

## 5. Supports, eligibility, identities

- `species_diagnostics.csv`: all 68 species `converged=True` (max `n_iter=33` of `max_iter=1000`),
  zero `ConvergenceWarning`s, `coef_finite=True` for all 68; no arbitrary coefficient-norm exclusion
  gate applied, matching the FINAL plan's explicit rejection of such a gate.
- Vocabulary/support gate independently reconstructed from `split_manifest.json`'s
  `vocabulary_fixed_point_history`: 68 retained species all have `n_sound_ids>=7` (round-0 support
  counts checked directly); 88 correctly dropped for `<7`; fixed point reached in one no-op round.
- Taxonomy resolution independently re-derived from `data/species.csv` (168 rows): reproduces the
  exact 6 exclusions (`PICIDA_1, PSITTA, PSITTACIDAE, PSITTACIFORMES, RHACAR, TYRANN_SP1`) and 2
  duplicate-binomial merges (`ATAPIL`/`ATRPIL`→`ATAPIL`; `RAMTUC`/`RHATUC`→`RAMTUC`).
- Split/group integrity: zero `sound_id` and zero physical-window `(sound_id,start,end)` overlap
  between train/val/test; augmented `train_split.csv` (95,536 rows = 29,513 canonical + 66,023
  overlap-only, `is_canonical` flagged correctly); `val_split.csv`/`test_split.csv` are 100%
  canonical; no duplicate `window_id` in any split.
- Identity alignment: `{train,val,test}_emb.npz` row order/`window_id`/`sound_id`/`start`/`end`
  match their source split CSVs exactly; `test_predictions.npz` matches `test_emb.npz` row-for-row.
**All verified.**

## 6. Probability shapes / finiteness / model metadata

- `train/val/test_emb.npz`: `(95536,1536)/(6398,1536)/(6474,1536)`, `float32`, all finite — matches
  plan's Phase 2 acceptance criterion.
- `test_predictions.npz["probabilities"]`: shape `(6474,68)`, **now `float64`** (precision fix
  applied — previously `float32`), all finite, range `[1.18e-05, 0.99993]`, no saturated/degenerate
  values. `any_bird_score` now `float64`, range `[0.0743, 1.0]`.
- `logreg_species.joblib`: `OneVsRestClassifier` of 68 `LogisticRegression` estimators;
  `metadata.model_config = {C:0.1, class_weight:balanced, solver:lbfgs, max_iter:1000}` (no forbidden
  `multi_class` kwarg); coefficients reloaded and independently confirmed to reproduce
  `test_predictions.npz` to `3.2e-7` (float32-embedding-input rounding only) and the exact headline
  macro-AP (0.4089036643316387). The joblib's on-disk bytes changed since the prior snapshot
  (re-serialization during the precision-fix rerun) but the fitted coefficients and resulting
  predictions/metrics are numerically identical — **confirmed by direct reload and re-prediction**,
  not merely by re-reading a cached number.
**Verified.**

## 7. What changed since the pre-fix snapshot, and confirmation it didn't move any result

| Artifact | Changed? | Why | Effect on reported numbers |
|---|---|---|---|
| `test_predictions.npz` | Yes | `probabilities`/`any_bird_score` now stored `float64` instead of `float32` | None — recomputed macro-AP, prevalence baseline, ge5/ge10, AUROC/AP all identical to the pre-fix values to full double precision |
| `logreg_species.joblib` | Yes (bytes) | Re-serialized during rerun | None — coefficients/predictions reload-verified identical |
| `training_manifest.json` | Yes | New `embedding_npz_sha256` field added; `config_hash`/`script_source_sha256`/timings updated | Additive only — closes a prior gap where the embedding NPZs themselves weren't directly hashed in this manifest |
| `c_selection.csv` | Yes | Wall-clock timing columns only | None — all four `val_macro_ap` values unchanged |
| `macro_ap_summary.csv`, `no_bird_detection.csv`, `species_ap.csv`, `species_diagnostics.csv` | **No** | — | Byte-identical to the pre-fix snapshot |

The precision fix is a pure numerical-fidelity improvement (removing an unnecessary float64→float32
downcast round-trip on stored predictions) plus a manifest hash-coverage improvement. It changed no
scientific conclusion.

## 8. Documentation-drift finding (minor, non-blocking)

`results.md` states "89 focused tests / 255 total tests." Current suite: `tests/test_train_perch_logreg.py`
collects/passes **90**; full repository suite collects/passes **256**. Both off by exactly one
(consistent with a test added since `results.md` was written); all currently pass. **SUGGESTION**:
refresh these two figures in `results.md`.

---

## Summary

### Pipeline Correctness
- C selection: verified by independent refit — `C=0.1` is the true argmax.
- AP/prevalence baselines, macro and thin-support metrics: verified by stored-prediction
  recomputation **and** a fully independent train→test refit; exact match on both paths.
- No-bird/any-bird metrics: verified; exact match.
- Supports, eligibility, identities: verified — no leakage, no duplicate windows, exact alignment.
- Probability shapes/finiteness: verified; now `float64`, all finite, no degenerate values.
- Model metadata: verified — matches the FINAL plan's specified estimator/config.
- Manifest hash chain: every hash re-verified OK against its physical target at this snapshot.

### Leakage Check
- Split integrity: clean — zero `sound_id`/physical-window overlap across train/val/test.
- Statistics leakage: clean — L2 normalization stateless and split-independent; vocabulary/support
  gate computed on canonical windows before fitting; `C` selected on validation only, test touched
  once after `C*` fixed.
- Augmentation leakage: clean — only `train_split.csv` carries non-canonical overlap windows.

### Schema Validation
- Expected fields: all present in every artifact enumerated in the plan's "Files" section.
- Types: correct (`float32` embeddings, `uint8` targets, `float64` probabilities post-fix, integer
  identity fields, no seconds column anywhere).
- Completeness: complete — `embedding_manifest.json` reports 0 excluded / 0 padding across all splits.

### Edge Cases
- Thin test support (24/68 `test_pos<5`, 43/68 `test_pos<10`): disclosed, not hidden; sensitivity
  subsets recomputed and match.
- Non-finite coefficients: none occurred (all 68 finite).

### Recommendations
- [ ] SUGGESTION: Refresh "89/255 tests" in `results.md` to the current 90/256.
- [ ] SUGGESTION: Commit `train_perch_logreg.py`, `extract_perch_embeddings.py`,
  `prepare_species_splits.py` (currently untracked/`git_dirty=true`) so `training_manifest.json` can
  record a real `git_commit` instead of `null`.
- [ ] SUGGESTION: Now that the directory has settled and hash-verified, treat this exact snapshot
  (all hashes listed in §1) as the citable "final" state going forward; re-run this validation's hash
  checks if the trainer is ever re-run again with `--force`.

---

STATUS: PASS

Every recomputable quantity — C selection, macro-AP and prevalence baselines, macro and
thin-support metrics, no-bird/any-bird metrics, supports, split/group identities, probability
shapes and finiteness, model metadata, and the full manifest hash chain — was independently
reproduced from raw embeddings, split CSVs, and the saved model against the current, post-precision-fix
snapshot, confirmed stable across repeated hash checks, and matches both the artifacts under
`checkpoints/perch/species_v1/` and the claims in
`docs/implementation/species-linear-probe-v1/results.md` to floating-point precision. The precision
fix (float32→float64 stored predictions, plus an added embedding-NPZ hash-verification field) changed
no scientific result.
