# Perch v2 Species Linear Probe -- Final Implementation Plan

**STATUS: FINAL -- converged in design Round 8 with zero new findings.**

This document is self-contained: every artifact schema, gate rule, and threshold it relies on is
specified in full below, not by reference to other design-round documents. Rounds 1-3
(`docs/design/round_01/*`, `docs/design/round_02/*`, `docs/design/round_03/architect_pipeline_review.md`)
remain available for the exploratory reasoning and alternatives-considered discussion behind these
decisions, but nothing in this document requires reading them. See `docs/design/CHANGELOG.md` for a
summary of the full Round 1-8 review history and rationale, if that process context is useful.

---

## Why

PteroSet's existing pipeline (`train.py`, `prepare_dataset.py`) does bird/no-bird detection from
spectrograms. Species-level classification is a new objective. Rather than designing a bespoke
extraction/training framework, this plan reuses the already-running, production-validated Perch v2
embedding + linear-probe recipe from the sibling repo `orcas_dclde2026`
(`eval_perch_ecotype.py`, `extract_perch_3class.py`, `remap_perch_embeddings.py`), adapted for
PteroSet's two structural differences: (1) species labels are multilabel (co-occurring birds in one
window), not single-label; (2) evaluation uses 5 fixed leave-one-project-out (LOPO) folds, not one
seed-selected stratified split. This revision closes a set of specific correctness and traceability
gaps identified after the previous draft, without adding scope: every change below tightens an
existing mechanism (exclusion handling, eligibility definitions, identity/hash discipline, numerical
trust) rather than adding a new pipeline stage.

## What (scope)

**In scope**: frozen Perch v2 embeddings, extracted once into a global pool; a species multi-hot
target derived from `annotations_species.json`; per-fold embedding views materialized from the pool
by gather-by-`window_id` (no re-extraction, no inference); a multilabel
`OneVsRestClassifier(LogisticRegression)` head trained per fold; an optional single-label comparator
with disclosed retention statistics; CSV metrics; window-level evaluation only.

**Out of scope, explicitly deferred, not silently dropped**: fine-tuning Perch's backbone, temporal/
sequence modeling, active learning, changes to the existing binary bird-detector pipeline (`train.py`,
`checkpoints_v4`), any recording- or segment-level pooling, a `C`-grid/calibration sweep, a Perch
v1-vs-v2 ablation, Perch's own zero-shot classifier as a secondary baseline, and promotion of the
copy/adapted scripts to a shared cross-repo package (revisit only when a third consumer needs the
same Perch-loading logic, or a CUDA/TF upgrade requires a synchronized fix across 2+ repos).

## How -- architecture summary

```
windows_mapping_4.0overlap_segmented_v4.json ------------+
annotations_species.json / species.csv / taxonomy       |
crosswalk / class_list config                     ------+--> build_species_labels.py -->
                                                                species_labels_v1.json, class_list.json

windows_mapping_4.0overlap_segmented_v4.json --> build_embedding_pool_csv.py --> pool.csv
    (identity columns: window_id, sound_id, sound_filepath, project,
     start_sample int, end_sample int, sample_rate int -- SAMPLES, not seconds; see section
     "Identity discipline" below)

pool.csv --> extract_embeddings_pteroset.py --(GPU, Perch v2, one pass)-->
    pool_emb_v2.npz (successful rows only) + pool_manifest.json (records both
    successes and every excluded row with a reason -- see "Failure handling" below)

pool_emb_v2.npz + pool_manifest.json + folds_segmented_v4/*/{train,val,test}_split.csv
    + species_labels_v1.json
    --> build_fold_embeddings.py --(gather by window_id, hard gates, NO inference)-->
        data/embeddings/perch_v2/folds_segmented_v4/fold_{i}_{PROJECT}/{train,val,test}_emb.npz
        + fold_manifest.json (x15) -- records exclusions propagated from the pool,
        per-species-per-fold eligibility table, and both hash lineages (embedding-hash
        and label-recipe-hash -- see "Cache invalidation" below)

fold_{i}/{train,val,test}_emb.npz --> train_perch_logreg.py -->
    consume species_eligibility_fold{i}.csv, select only trainable species columns,
    L2-normalize each embedding row, then fit:
    checkpoints/perch/logreg_species_fold{i}.joblib (OneVsRestClassifier(LogisticRegression))
    + checkpoints/perch/logreg_species_comparator_fold{i}.joblib (optional single-label comparator)
    + species_diagnostics_fold{i}.csv (per-class n_iter_, convergence, coefficient health)
    + comparator_eligibility_fold{i}.csv + comparator_diagnostics_fold{i}.csv
    + fulldata_results_species_fold{i}.csv (per-species metrics, eligibility-masked)
    + macro_ap_core_summary.csv (fixed cross-fold-comparable species set, averaged)
    + macro_ap_per_fold_own_eligible.csv (each fold's own eligible set, never averaged together)
    + comparator_retention.csv (single-label comparator's retained-fraction disclosure)
```

Perch is invoked exactly once, in `extract_embeddings_pteroset.py`. Every downstream step
(`build_fold_embeddings.py`, `train_perch_logreg.py`) is pure array/metadata manipulation -- no audio
I/O, no TensorFlow, no GPU required, and therefore trivially resumable/rerunnable. Only
`extract_embeddings_pteroset.py` needs TensorFlow/kagglehub/librosa/GPU; every other script needs only
pandas/numpy/scikit-learn/joblib, which the standard `bioacoustics` environment already provides for
`train.py`/`prepare_dataset.py` today.

---

## Identity discipline (all gates, hashes, and joins use integer samples, never seconds)

Every identity comparison, hard gate, and cache key in this pipeline is defined over
`(window_id, sound_id, start_sample, end_sample, sample_rate)` -- all integers. Seconds are computed
exactly once, transiently, inside `extract_embeddings_pteroset.py`, immediately before the
`librosa.load(..., offset=start_sample/sample_rate, duration=(end_sample-start_sample)/sample_rate)`
call, and are never written to any persisted artifact, never compared for equality, and never hashed.
This exists specifically so that a future change to how seconds are rounded or represented (e.g. a
different float precision, a different rounding convention) cannot silently change what two systems
consider "the same window" -- the identity contract is the original integer geometry PteroSet already
uses everywhere else (`windows_mapping_4.0overlap_segmented_v4.json`), full stop.

`pool.csv` (from `build_embedding_pool_csv.py`) columns: `window_id, sound_id, sound_filepath,
project, start_sample (int64), end_sample (int64), sample_rate (int32)`. No seconds column exists in
this file or in any downstream artifact.

---

## Failure handling: extraction failures reconciled with the fold-materializer's hard gates

**Pool manifest keys** (`pool_manifest.json`, written by `extract_embeddings_pteroset.py`):

```json
{
  "created_at_utc": "...",
  "git_commit": "<sha>",
  "model": {"name": "perch_v2", "kaggle_slug": "google/bird-vocalization-classifier/tensorFlow2/perch_v2",
            "local_dir": "checkpoints/perch/model_v2", "embedding_key": "<resolved>", "embedding_dim": 1536},
  "identity_hash_inputs": {
    "windows_mapping_json_sha256": "<sha256 of windows_mapping_4.0overlap_segmented_v4.json>",
    "pool_csv_sha256": "<sha256 of the generated pool.csv>",
    "pool_builder_source_sha256": "<sha256 of build_embedding_pool_csv.py>",
    "pool_builder_config": {"window_version": "segmented_v4", "identity_units": "integer_samples"},
    "model_local_dir_sha256": "<sha256 over a sorted listing of the vendored SavedModel dir's files>",
    "extraction_params": {"target_sample_rate": 32000, "window_sec": 5.0, "target_peak": 0.25, "batch_size": 64}
  },
  "dependency_versions": {"tensorflow": "2.21.0", "scikit-learn": "<resolved>", "kagglehub": "1.0.0",
                          "numpy": "2.2.5", "librosa": "0.11.0"},
  "failure_ceiling": {
    "global_max_failures": 0,
    "per_project_max_failures": 0,
    "note": "Defaults are pre-registered. Any reviewed override must be set before extraction and recorded verbatim."
  },
  "counts": {"requested": 160244, "succeeded": 160244, "excluded": 0},
  "expected_success": {"window_id_list_stored_in": "pool_emb_v2.npz:window_id"},
  "excluded": {}
}
```

**Rule**: `extract_embeddings_pteroset.py` aborts the entire run (`SystemExit`) the instant either
ceiling in `failure_ceiling` is exceeded -- checked **both** globally (total excluded count across all
160,244 windows) **and** per-project (excluded count within any single project's windows), because a
failure rate that looks small in aggregate can still mean one project's audio is disproportionately
degraded, which would quietly bias that project's LOPO fold. Both ceilings default to `0` (today's
abort-on-first-failure behavior); a non-zero override must be an explicit, reviewed, pre-registered
CLI flag value, never a number chosen because "that's how many failures actually happened" -- that
would be measuring the data and then calling the measurement a design decision, which this pipeline
treats as a process violation, not a convenience.

**Rule for `build_fold_embeddings.py`**: for every `window_id` present in a fold's
`{train,val,test}_split.csv`, exactly one of the following must hold, or the script aborts with a
named `window_id`:
1. `window_id` is present in `pool_emb_v2.npz` (a successful extraction) -- gather it, subject to the
   identity gates below.
2. `window_id` is present in `pool_manifest.json`'s `excluded` map (an explicit, reasoned exclusion)
   -- do not gather it; instead, record it into that fold/split's own exclusion list with the reason
   propagated verbatim from the pool manifest, and do not count it in that split's row total.
3. Neither (1) nor (2): **abort**. An unexplained missing row is a data-integrity failure, not a
   normal exclusion, and must never be silently treated as either a success or a known exclusion.

`fold_manifest.json` (per fold/split) therefore carries, in addition to the gate results already
specified:
```json
{
  "n_rows_in_split_csv": 12348,
  "n_rows_embedded": 12345,
  "n_rows_excluded": 3,
  "excluded": {"913042": {"reason": "audio_load_error", "propagated_from": "pool_manifest.json"}},
  "gates": {"window_id_existence_or_explicit_exclusion": "pass",
            "sound_id_start_end_sample_rate_match": "pass",
            "filepath_match": "pass",
            "self_verification_sample_size": 20,
            "self_verification_max_l2_diff": 0.0}
}
```

`build_fold_embeddings.py` is the sole producer of `species_eligibility_fold{i}.csv`, because it
already owns the post-exclusion train/test labels and support counts. `train_perch_logreg.py`
consumes this table; it does not recompute or overwrite it.

**Support/eligibility computed after exclusions, never before**: every per-species-per-fold count
used in the eligibility table below (`train_pos`, `train_neg`, `test_pos`, `test_neg`) is computed
only over rows that were actually embedded (excluded rows contribute to none of the four counts, not
even as an implicit negative). This is deliberate: an excluded window is missing evidence, not
negative evidence, and conflating the two would quietly understate a species' true prevalence.

---

## Per-species-per-fold eligibility (five explicit categories, not a single eligible/ineligible flag)

For each of the 5 LOPO folds and each canonical species in `class_list.json`, compute, after
exclusions:

- `train_pos`, `train_neg` -- positive/negative window counts in that fold's train split.
- `test_pos`, `test_neg` -- positive/negative window counts in that fold's test split.

Derive five boolean categories per (fold, species) pair, all independently reportable (a species can
be, e.g., both `trainable` and `test_absent` at once):

| Category | Condition | Meaning |
|---|---|---|
| `trainable` | `train_pos >= min_support` (default 10, configurable) and `train_neg >= 1` | Enough evidence to fit that species' binary classifier at all. |
| `structurally_unseen` | `train_pos == 0` | Zero-shot for this fold; cannot be trained regardless of test composition. Implies not `trainable`. |
| `test_absent` | `test_pos == 0` | No positive test examples; recall/AP for this species in this fold is undefined even if `trainable`. |
| `test_single_class` | `test_neg == 0` | Test split is 100% positive for this species; ROC-AUC/AP is degenerate from the other direction (`sklearn.metrics.roc_auc_score` requires both classes present). |
| `evaluable` | `trainable` and not `test_absent` and not `test_single_class` | The species contributes to that fold's headline per-fold metrics. |

This table (`species_eligibility_fold{i}.csv`) is a required artifact per fold, produced by
`build_fold_embeddings.py` after exclusions and consumed by `train_perch_logreg.py` before any metric
is computed. Every downstream metrics table is filtered through it explicitly -- no metric is ever
computed for a non-`evaluable` (fold, species) pair and silently reported as if it were.

The same table is also a **pre-fit column-selection gate**, not only a reporting filter.
`train_perch_logreg.py` passes only species with `trainable == True` to
`OneVsRestClassifier.fit()`. This prevents scikit-learn from aborting on constant all-negative or
all-positive target columns, which are expected in LOPO folds with project-specific species.
Predictions are then reindexed to the fixed global `class_list.json` order; non-trainable species
receive `NaN` scores plus their explicit eligibility status, never fabricated zero probabilities.
If a fold has no trainable species, that fold fails with a clear error before fitting.

---

## Headline metrics: two deliberately different macro-AP numbers, never blended

- **`macro_ap_core`**: computed only over the **fixed** set of species that are `evaluable` in **all
  5** folds (the intersection) and remain `numerically_trusted` in all 5 fitted models. Because the
  denominator (species set) is identical across folds, the
  per-fold `macro_ap_core` values are directly comparable and are reported as
  `mean +/- std across the 5 folds` in `macro_ap_core_summary.csv` -- this is the single headline
  scalar for the go/no-go decision (Phase 3 acceptance criterion below). The file must report
  `n_species_core` and the exact species list. Phase 0 pre-registers `min_core_species` (recommended
  initial value: 5); if the final intersection is smaller, no headline scalar or go/no-go claim is
  produced -- only the per-fold tables are reported.
- **`macro_ap_fold_own_eligible`**: computed per fold over that fold's own `evaluable` species set
  (which generally differs in membership and size fold to fold, since a project's held-out species
  composition differs). Reported as one row per fold in `macro_ap_per_fold_own_eligible.csv`,
  **each row labeled with its own N_species_eligible and species list**. These five numbers are
  **never averaged into a single scalar** -- doing so would average metrics computed over different,
  incomparable denominators, and any report or downstream consumer of this pipeline must not do so
  either. This restriction is stated here as a hard rule, not a style preference: a script that
  computes `mean(macro_ap_fold_own_eligible)` as if it were a valid summary statistic is a bug.

Per-class AUROC/AP, macro/micro aggregates, and per-project breakdowns remain as previously specified,
all filtered through the eligibility table above.

---

## Numerical trust per per-class estimator (OneVsRestClassifier fits K independent binary models)

For every fitted trainable-species `LogisticRegression` estimator inside `OneVsRestClassifier`,
record, per (fold, species), into `species_diagnostics_fold{i}.csv`:

- `n_iter_` (from the fitted estimator) and the `max_iter` it was given.
- `converged` = `n_iter_ < max_iter` (a value equal to `max_iter` means `lbfgs` did not converge in
  the allotted iterations -- `sklearn` raises a `ConvergenceWarning` in this case; that warning is
  captured, not suppressed).
- `coef_finite` = `True` iff every entry of `coef_` and `intercept_` is finite (no `NaN`/`Inf`).
- `coef_l2_norm` = `np.linalg.norm(coef_)`.
- `numerically_trusted` = `converged and coef_finite and coef_l2_norm <= coef_norm_ceiling`, where
  `coef_norm_ceiling` is an explicit, configurable default (`[RECOMMENDATION, not a measured fact]`:
  start at `50.0` for L2-normalized 1536-dim embeddings and revisit once real fold data is available
  -- an unbounded or very large coefficient norm is the classic symptom of a perfectly- or
  near-perfectly-separable small-sample logistic fit, which happens easily for a rare species with
  very few training windows).

**Rule**: a (fold, species) pair with `numerically_trusted == False` is excluded from
`macro_ap_core`, `macro_ap_fold_own_eligible`, and every other headline aggregate, exactly as if it
were not `evaluable` -- but it still appears, flagged, in `species_diagnostics_fold{i}.csv` and in a
dedicated "excluded for numerical reasons" appendix table in the results report, so nothing is ever
silently dropped without a visible trace distinguishing "not enough data" (the eligibility table)
from "fit was numerically untrustworthy despite having data" (this diagnostics table).

`species_diagnostics_fold{i}.csv` is reindexed to the full global class vocabulary. Trainable,
fitted species contain the diagnostics above; non-trainable species contain `NaN` for fit-derived
fields, `numerically_trusted = False`, and their explicit eligibility columns. The schema must include
`trainable`, `fit_attempted`, and `numerical_exclusion`, where `fit_attempted == trainable` and
`numerical_exclusion == (fit_attempted and not numerically_trusted)`. No diagnostics are fabricated
for estimators that were never fit. The "excluded for numerical reasons" appendix filters
`numerical_exclusion == True`; it must never filter on `numerically_trusted == False` alone, which
would conflate unfit species with failed fits.

Before fitting or predicting, apply `sklearn.preprocessing.normalize(X, norm="l2")` independently to
each split's embedding rows, matching the Orcas Perch protocol. No `StandardScaler` is fitted or
persisted. The L2 operation is stateless and therefore cannot leak fold information.

---

## Optional single-label comparator: retention disclosure and non-generalizability caveat

The comparator is trained only on the subset of species-positive windows that reduce cleanly to one
dominant species (`clean_single` in the six-state label schema: the annotation with the largest
temporal overlap wins, ties broken by earliest `annotation_id`). This subset is, by construction, not
a random sample of all species-positive windows -- it is biased toward windows with less co-occurring
ambiguity, which may correlate with species identity (common species may be more or less likely to
co-occur with others than rare ones).

**Required disclosure, per fold, in `comparator_retention.csv`**:
- `retained_fraction_overall` = `n(clean_single windows used) / n(species-positive windows)` for that
  fold's train split.
- `retained_fraction_by_species` -- one row per species: what fraction of that species' positive
  windows survived the single-label reduction.
- `retained_fraction_by_project` -- one row per project: same, broken down by project instead of
  species.

**Required caveat, verbatim, attached to every table or figure presenting comparator results**: *"The
single-label comparator's metrics describe performance only on the disambiguable subset of windows
with one unambiguous dominant species label. Retention is uneven across species and projects (see
`comparator_retention.csv`); these numbers must not be presented as evidence of, or assumed to
generalize to, performance on the full multilabel deployment population, which is what the primary
`OneVsRestClassifier` branch measures."*

The comparator has its own gates because its retained `clean_single` population is smaller and
non-uniform. `comparator_eligibility_fold{i}.csv` records per-class train/test support after retention
and requires at least two train classes, the configured minimum train support for each reported
class, and test support for every class included in a metric. `comparator_diagnostics_fold{i}.csv`
records the multinomial fit's `n_iter_`, captured `ConvergenceWarning`, coefficient finiteness, and
coefficient norms. Comparator metrics are withheld when these gates fail; the primary multilabel
results are unaffected.

---

## Cache invalidation: two independent hash lineages, so a label-only change never forces GPU re-work

**Pool embedding hash** (governs whether `extract_embeddings_pteroset.py` must re-run): a function of
`windows_mapping_4.0overlap_segmented_v4.json`'s SHA-256, the generated `pool.csv` content SHA-256,
the `build_embedding_pool_csv.py` source hash and relevant config values, the vendored model
directory's content hash, and the extraction params (target sample rate, window seconds, target peak,
batch size). It
depends on **nothing** about species labels, because the pool contains only embeddings and identity
columns, never a label field.

**Label-recipe hash** (governs whether `build_fold_embeddings.py`'s label-derivation step, not
`extract_embeddings_pteroset.py`, must re-run): a function of the SHA-256 of
`annotations_species.json`, `species.csv`, the taxonomy crosswalk file (if present),
`class_list.json`, and the label-derivation config values (`min_overlap_frac`, `min_class_support`,
`excluded_codes`).

Both hashes are recorded, separately, in `fold_manifest.json`:
```json
{
  "pool_embedding_hash": "<hash of windows-json + pool.csv + pool-builder source/config + model-dir + extraction params>",
  "label_recipe_hash": "<hash of annotations_species.json + species.csv + crosswalk + class_list.json + label config>"
}
```

**Consequence, stated explicitly because it is the entire point of separating these two hashes**: if
`annotations_species.json` is corrected (a common event -- re-annotation, a taxonomy fix) but the
underlying audio/window geometry and Perch model are unchanged, only `label_recipe_hash` changes.
`build_fold_embeddings.py` detects this and rebuilds the 15 fold/split label views (seconds, no GPU,
no audio I/O) -- `pool_emb_v2.npz` is untouched and does not need to be re-extracted. If instead the
windows JSON itself changes (e.g. a new segmentation version) or the model is upgraded, both hashes
change and a full re-extraction is required.

---

## Environment: the fallback-env decision applies to `extract_embeddings_pteroset.py` only

`build_species_labels.py`, `build_embedding_pool_csv.py`, `build_fold_embeddings.py`, and
`train_perch_logreg.py` run in the standard `bioacoustics` conda environment already used by
`train.py`/`prepare_dataset.py` -- they need only pandas/numpy/scikit-learn/joblib, all already
present there, and none of them import TensorFlow, kagglehub, or librosa. No smoke test or fallback
logic applies to these four scripts.

Only `extract_embeddings_pteroset.py` needs TensorFlow, kagglehub, librosa, and a CUDA-capable GPU
(Perch v2's SavedModel is XLA-compiled CUDA-only). Its environment is decided by a one-time smoke
test, run once at Phase 0:

```bash
conda activate bioacoustics
python -c "import tensorflow as tf, sklearn, kagglehub, joblib, librosa; \
    print(tf.__version__, sklearn.__version__, tf.config.list_physical_devices('GPU'))"
```

**Pass criteria** (all must hold, else fall through to the fallback below):
1. All five imports succeed with no `ModuleNotFoundError`.
2. `tf.config.list_physical_devices('GPU')` returns at least one device.
3. One real forward pass through the vendored `model_v2` SavedModel on one sample window succeeds
   without error and without a silent CPU fallback.

Handle failures by cause:
1. **Imports fail**: build a separate, minimal extraction-only environment pinned to
   `orcas_dclde2026/pip-requirements.txt`'s verified versions.
2. **No GPU is visible**: an environment rebuild cannot fix missing hardware. Move extraction to a
   GPU host and rerun the smoke test. A CPU Perch export is a separate future design choice, not an
   automatic fallback in this baseline.
3. **GPU is visible but the forward pass fails**: diagnose the CUDA bootstrap, SavedModel integrity,
   and TensorFlow/CUDA compatibility; extraction remains blocked until the real forward pass passes.

Every other script in this plan continues to run in the standard `bioacoustics` environment
regardless of the extraction decision.

Before model vendoring, verify and record the Perch v2 weights license, the source-code license, the
resolved Kaggle model version, and any redistribution constraints in `pool_manifest.json` and the
results report. Model download and publication are blocked if the license cannot be verified.

---

## Files

```
build_species_labels.py          # multilabel target + six-state schema + single-label
                                  # "primary annotation" reduction (for the comparator)
build_embedding_pool_csv.py       # windows_mapping JSON -> pool.csv; identity columns are
                                  # start_sample/end_sample/sample_rate (integers), never seconds
extract_embeddings_pteroset.py    # copied/adapted from eval_perch_ecotype.py:
                                  #   CUDA bootstrap, download_perch_v2, load_perch_v2,
                                  #   inspect_model_outputs, load_audio_segment (computes seconds
                                  #   transiently from start_sample/sample_rate), extract_embeddings
                                  #   + global AND per-project failure-ceiling enforcement
                                  #   + pool_manifest.json writer (successes + reasoned exclusions)
build_fold_embeddings.py          # copied/adapted from remap_perch_embeddings.py, generalized to
                                  #   5 folds x 3 splits: gather by window_id; every split row is
                                  #   required to be present-success or explicit-excluded, else abort;
                                  #   identity gates on (window_id, sound_id, start_sample, end_sample,
                                  #   sample_rate) + filepath, never seconds; computes both hash
                                  #   lineages; sole producer of the per-species-per-fold eligibility
                                  #   table after exclusions
train_perch_logreg.py             # OneVsRestClassifier(LogisticRegression(solver="lbfgs",
                                  #   max_iter=1000, C=1.0)) -- no multi_class kwarg (removed in
                                  #   scikit-learn >= 1.7, raises TypeError if passed); optional
                                  #   --comparator single_label using plain LogisticRegression
                                  #   (also no multi_class kwarg); writes species_diagnostics,
                                  #   comparator_eligibility, comparator_diagnostics,
                                  #   macro_ap_core_summary, macro_ap_per_fold_own_eligible,
                                  #   comparator_retention; fits only trainable target columns and
                                  #   reindexes predictions to the global class vocabulary
```

No new Python package/`__init__.py` layering -- flat, dual-purpose scripts (both CLI and importable),
mirroring the reference repo's own convention.

---

## Commands

```bash
# Phase 0 -- environment smoke test (extract_embeddings_pteroset.py's own dependencies only)
conda activate bioacoustics
python -c "import tensorflow as tf, sklearn, kagglehub, joblib, librosa; \
    print(tf.__version__, sklearn.__version__, tf.config.list_physical_devices('GPU'))"

# Phase 0 -- one-time model vendoring
python extract_embeddings_pteroset.py --download_v2

# Phase 1 -- species labels (multilabel target + single-label comparator subset)
python build_species_labels.py --config data/config.yaml \
    --annotations_species data/annotations_species.json \
    --species_csv data/species.csv \
    --windows_mapping data/windows_mapping_4.0overlap_segmented_v4.json \
    --out_multilabel data/species_labels/segmented_v4/species_labels_v1.json \
    --out_class_list data/species_labels/segmented_v4/class_list.json \
    --min_class_support 10 --min_overlap_frac 0.0

# Phase 1 -- pool CSV (integer sample identity columns; no seconds are persisted)
python build_embedding_pool_csv.py --config data/config.yaml \
    --windows_json data/windows_mapping_4.0overlap_segmented_v4.json \
    --out data/embeddings_pool/segmented_v4_pool.csv

# Phase 2 -- pool extraction (the only GPU-bound, Perch-bound, possibly-fallback-env step)
python extract_embeddings_pteroset.py --extract --model v2 \
    --pool_csv data/embeddings_pool/segmented_v4_pool.csv \
    --out data/embeddings/perch_v2/pool_emb_v2.npz \
    --batch_size 64 --max_failures 0 --max_failures_per_project 0

# Phase 2 -- fold materialization (no GPU, no Perch, standard bioacoustics env, seconds not minutes)
python build_fold_embeddings.py \
    --pool_npz data/embeddings/perch_v2/pool_emb_v2.npz \
    --pool_manifest data/embeddings/perch_v2/pool_manifest.json \
    --fold_dir data/folds_segmented_v4 \
    --species_labels data/species_labels/segmented_v4/species_labels_v1.json \
    --class_list data/species_labels/segmented_v4/class_list.json \
    --out_dir data/embeddings/perch_v2/folds_segmented_v4

# Phase 3 -- training + evaluation, all 5 folds, standard bioacoustics env
python train_perch_logreg.py \
    --embeddings_dir data/embeddings/perch_v2/folds_segmented_v4 \
    --class_list data/species_labels/segmented_v4/class_list.json \
    --out_dir checkpoints/perch \
    --min_support 10 --coef_norm_ceiling 50.0 \
    --comparator single_label
```

---

## Phased milestones and acceptance criteria

**Phase 0 (~1 day)**: environment smoke test passes for `extract_embeddings_pteroset.py`'s
dependencies, or the documented fallback env is built (only for that one script); `checkpoints/
perch/model_v2/` vendored locally; one real PteroSet window embedded end-to-end, shape `(1536,)`
confirmed. Perch v2 weights/code licenses and the resolved Kaggle model version are recorded.
`min_core_species` is pre-registered before training.

**Phase 1 (~1-2 days)**: `species_labels_v1.json`, `class_list.json` produced; `pool.csv` produced
with integer `start_sample`/`end_sample`/`sample_rate` columns, row count == 160,244. Acceptance:
every fold's `test_split.csv` project matches that fold's held-out project (existing repo invariant,
re-checked here, not re-derived).

**Phase 2 (~1-2 days)**: `pool_emb_v2.npz` + `pool_manifest.json` produced; `counts.excluded` is
`0` unless an explicit, reviewed `--max_failures`/`--max_failures_per_project` override was set and
is documented verbatim in the manifest's `failure_ceiling` block. All 5 folds x 3 splits materialize
with every gate reported `"pass"` in `fold_manifest.json`, zero unexplained-missing-row aborts, and
every propagated exclusion traceable to a `pool_manifest.json` reason.
`species_eligibility_fold{i}.csv` is produced for all 5 folds after exclusions.

**Phase 3 (~1-2 days)**: `train_perch_logreg.py` consumes the Phase 2 eligibility tables;
selects only `trainable` species columns, and trains L2-normalized `OneVsRestClassifier` models on all
5 folds without constant-column errors; predictions are reindexed to the global class list with
non-trainable species represented as `NaN` plus status;
`species_diagnostics_fold{i}.csv` produced with `numerically_trusted` computed for every (fold,
trainable species) fit and full-vocabulary placeholder rows for non-trainable species;
`macro_ap_core_summary.csv` reports `mean +/- std` over the fixed cross-fold-eligible
and numerically trusted species set, including `n_species_core` and its exact members;
`macro_ap_per_fold_own_eligible.csv` reports five independent, never-averaged rows. If
`n_species_core < min_core_species`, no headline scalar is produced.
Go/no-go: `macro_ap_core`'s mean beats a per-species prevalence-only baseline by a pre-registered
margin (default +5 percentage points, configurable) . A negative result is a valid, useful outcome
(matches the sibling repo's own documented Stage-1-failure precedent for a structurally similar task
shape) -- it is not a pipeline defect. `comparator_retention.csv` produced alongside the optional
single-label comparator's results, with the non-generalizability caveat attached verbatim wherever
comparator numbers are reported.
`comparator_eligibility_fold{i}.csv` and `comparator_diagnostics_fold{i}.csv` are also produced, and
the comparator is withheld for any fold that fails its own support or numerical-trust gates.

**Phase 4 (~1 day, documentation)**: results report written to
`docs/implementation/species-linear-probe-v1/results.md`, modeled on
`orcas_dclde2026/reports/ecotype_classifier.md`'s structure, explicitly including the eligibility
breakdown, the numerical-trust exclusions, both macro-AP tables (never blended), and the comparator's
retention caveat.

---

## Tests (minimal, highest-value only)

1. `test_fold_cache_gates.py` -- `build_fold_embeddings.py` aborts on a deliberately corrupted pool
   (mismatched `sound_id`/`start_sample`/`end_sample`/`sample_rate`, or a `window_id` that is neither
   in the pool's successes nor its `excluded` map). Highest priority: protects the leakage-free
   property the whole LOPO design depends on, and the new require-present-or-explicitly-excluded rule.
2. `test_perch_io.py` -- `load_audio_segment` on a synthetic sine wave produces exactly
   `window_sec * target_sr` samples and the correct peak value, given a `start_sample`/`sample_rate`
   pair converted to seconds internally; asserts the conversion happens only inside the loader call,
   never persisted.
3. `test_species_label_derivation.py` -- synthetic overlapping annotations produce the expected
   multi-hot vector and the expected single-label "largest-overlap-wins" reduction with its
   deterministic tie-break.
4. `test_eligibility_categories.py` -- synthetic per-fold support tables assert the five categories
   (`trainable`, `structurally_unseen`, `test_absent`, `test_single_class`, `evaluable`) are computed
   correctly at exact boundary conditions (e.g. `train_pos == min_support` is `trainable`,
   `min_support - 1` is not; `test_pos == 0` is `test_absent` regardless of `trainable`).
5. `test_numerical_trust_flagging.py` -- a synthetic perfectly-separable two-class fit is asserted to
   be flagged `numerically_trusted == False` via the coefficient-norm ceiling, and is asserted to be
   excluded from a synthetic macro-AP aggregate computed by the same code path used in
   `train_perch_logreg.py`.
6. `test_cache_hash_independence.py` -- asserts that changing a label-only input (e.g.
   `class_list.json`) changes `label_recipe_hash` but not `pool_embedding_hash`, and vice versa for a
   change to the windows JSON; changing `pool.csv` or `build_embedding_pool_csv.py` must change
   `pool_embedding_hash`.
7. Smoke test: end-to-end run on a tiny synthetic pool/fold NPZ produces non-empty
   `macro_ap_core_summary.csv` and `macro_ap_per_fold_own_eligible.csv` files.
8. `test_l2_normalization.py` -- verifies every row passed to either classifier has unit L2 norm
   (except an all-zero vector, which is rejected upstream).
9. `test_comparator_gates.py` -- verifies comparator support and numerical-trust failures withhold
   comparator metrics without affecting the primary multilabel branch.
10. `test_trainable_column_selection.py` -- a synthetic fold containing trainable, all-negative, and
    all-positive species columns fits without `ValueError`; only trainable columns reach
    `OneVsRestClassifier.fit()`, and outputs are reindexed to the full class list with `NaN` for
    excluded columns.
11. `test_diagnostics_exclusion_reasons.py` -- verifies unfit/non-trainable species have
    `fit_attempted == False` and `numerical_exclusion == False`, while a failed fitted estimator has
    both `fit_attempted == True` and `numerical_exclusion == True`.

No broader unit-test suite is proposed for the sklearn training/eval code itself beyond the above,
matching the reference repo's own convention (it has none) and the principle of testing the parts
that can silently corrupt data or hide a bad number (joins, audio normalization, label derivation,
eligibility/trust flagging) rather than well-tested library calls (`sklearn.fit`).

---

## Key operational failure modes

- Perch v2's SavedModel is CUDA-only; the CUDA `LD_LIBRARY_PATH` bootstrap must re-exec the process
  before any TensorFlow import, or it silently fails to find GPU libraries.
- A window's audio fails to load: default behavior is to abort the whole pool extraction once either
  the global or per-project failure ceiling (default `0` for both) is exceeded; this is deliberate,
  not a bug, to avoid ever training on a silently zero-filled embedding treated as a valid multilabel
  negative, and to avoid one project's degraded audio disproportionately and invisibly biasing its
  LOPO fold.
- A fold's split CSV references a `window_id` that is neither a pool success nor a pool-documented
  exclusion: `build_fold_embeddings.py` aborts rather than silently dropping or silently gathering a
  stale/wrong row.
- `kagglehub` network/auth failure only affects the one-time `--download_v2` step; every subsequent
  run uses the vendored local `checkpoints/perch/model_v2/` directory.
- `LogisticRegression(..., multi_class=...)` must never be passed on scikit-learn >= 1.7 --
  the parameter was deprecated in 1.5 and removed in 1.7, raising a constructor-time `TypeError`.
  Neither classifier construction in this plan passes it (verified against scikit-learn's own
  removal notice; the reference repo's own script still passes it while pinning `scikit-learn==1.7.2`
  in its requirements file, an internal inconsistency in that repo this plan does not inherit).
- A species with zero training examples in a given LOPO fold (`structurally_unseen`) cannot be
  evaluated in that fold; a species with zero positive test examples (`test_absent`) or an all-positive
  test split (`test_single_class`) cannot have a meaningful AUROC/AP in that fold either -- all three
  are excluded from that fold's `evaluable` set and therefore from every headline aggregate, never
  silently scored as zero/`nan`.
- A per-species binary fit that does not converge, produces non-finite coefficients, or has an
  implausibly large coefficient norm is excluded from headline aggregates even if it is otherwise
  `evaluable` by the data-support definition -- data support and numerical trust are independent
  gates, and passing one does not imply passing the other.

---

## Future migration path

Promote the copy/adapted scripts to a shared cross-repo package only when a third consumer needs the
same Perch-loading logic, or when a CUDA/TF upgrade requires a synchronized fix across 2+ repos.
Revisit fine-tuning Perch's backbone or an alternative encoder (BirdNET, SurfPerch) only after this
frozen-embedding baseline has a measured `macro_ap_core` result to beat. Revisit segment-level pooling
as a genuine secondary evaluation view once the window-level baseline is trusted -- PteroSet's
duty-cycled, non-continuous file structure means whole-file pooling is not a valid substitute for
this and should not be attempted without first defining the correct sub-file acoustic unit.

---

STATUS: FINAL
