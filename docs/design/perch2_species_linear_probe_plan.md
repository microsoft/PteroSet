# Perch v2 Species Linear Probe -- Implementation Plan

**STATUS: FINAL (2026-07-28) -- user-approved task-specific split implemented and verified,
superseding both the Round-8 FINAL LOPO design and the interim site-grouped/three-metric-gate
revision. Documentation convergence reached with zero findings in doc-fix Round 3.**

This document is self-contained: every artifact schema and gate rule it relies on is specified in
full below. `docs/design/CHANGELOG.md` records the full history (Round 1-8 LOPO design, the
post-Round-8 correction to a dataset-wide split, and the final approved-design correction this
revision applies) and is not a dependency of the plan itself. Rounds 1-8
(`docs/design/round_01/` .. `docs/design/round_08/`) remain available as historical/exploratory
context only; nothing in this document requires reading them, and their LOPO framing should not be
read as still authoritative.

This is a **technical-validation baseline**: it establishes whether frozen Perch v2 embeddings carry
usable species signal on PteroSet at all, evaluated on one task-specific, in-distribution 70/15/15
split. It is not, and does not claim to be, an unseen-recording or cross-project generalization
result.

---

## Why

PteroSet's existing pipeline (`train.py`, `prepare_dataset.py`) does bird/no-bird detection from
spectrograms, evaluated with 5 leave-one-project-out (LOPO) folds -- a good design for that task,
because it asks "does the detector generalize to an unseen recording project?" and every window has a
detection label. Species-level classification is a different task: labels are only resolvable for a
subset of bird-positive windows, class support is long-tailed, and the scientific question is
narrower ("do frozen embeddings separate PteroSet species at all, given a representative sample of
the data?", not "does a species classifier generalize to an unseen project?"). Reusing the detector's
LOPO/fold structure for this task risks per-fold species pools too small or too skewed to support a
meaningful macro-AP, and answers a harder question than the one being asked. **LOPO is rejected for
the species task; the existing binary detector correctly keeps LOPO.**

A subsequent draft of this plan additionally grouped windows by recording *site*
(`(project, event_indicator)`) rather than by `sound_id`, reasoning that `event_indicator` codes
repeat across recording dates and are not guaranteed unique across projects. **That elaboration was
not requested or approved and has been reverted.** The approved design groups by `sound_id`: no audio
file crosses a split boundary, and no coarser grouping is required for this baseline. See
`docs/lessons.md` ("Confirm split-design decisions before implementation") for the general lesson --
an agent must not substitute its own "safer" design choice for the one the user actually approved,
even when the substitution looks more rigorous.

**Approved correction**: evaluate on one dataset-wide, task-specific 70/15/15 split (not LOPO, not
fold4), grouped by `sound_id`, stratified over canonical (non-overlapping) windows so that each
species' and the no-bird class's proportions approximate 70/15/15. Project distribution is computed
and reported for transparency only -- it is not a stratification target. True no-bird windows are
kept as all-zero multilabel targets at their natural prevalence, never downsampled.

## What (scope)

**In scope**: frozen Perch v2 embeddings extracted directly into train/val/test NPZ files; a species
multi-hot target derived from `annotations_species.json`; one dataset-wide, `sound_id`-grouped,
multilabel-stratified 70/15/15 split over canonical windows, with overlapping windows added to the
train view only after the split is fixed; a multilabel `OneVsRestClassifier(LogisticRegression)` head
with `C` selected from a small pre-defined grid by validation macro-AP; macro-AP (primary),
per-species AP/support, and a separate no-bird/any-bird detection metric; window-level evaluation
only.

**Out of scope, explicitly deferred, not silently dropped**: fine-tuning Perch's backbone, temporal/
sequence modeling, active learning, changes to the existing binary bird-detector pipeline (`train.py`,
`checkpoints_v4`), leave-one-project-out / leave-one-site-out / cross-project or cross-site
generalization evaluation for the species task, any recording- or segment-level pooling, an optional
single-label comparator, cross-fold metrics, an arbitrary coefficient-norm exclusion gate, a Perch
v1-vs-v2 ablation, Perch's own zero-shot classifier as a secondary baseline, and promotion of the
copy/adapted scripts to a shared cross-repo package.

## How -- architecture summary

```
windows_mapping_4.0overlap_segmented_v4.json
annotations_species.json / species.csv
                                                  --> prepare_species_splits.py -->
      data/splits_species_v1/canonical_train_split.csv   (canonical, audited-distribution only)
      data/splits_species_v1/train_split.csv             (canonical + overlapping -- augmented,
                                                             used for fitting)
      data/splits_species_v1/val_split.csv                (canonical only)
      data/splits_species_v1/test_split.csv               (canonical only)
      data/splits_species_v1/class_list.json
      data/splits_species_v1/split_manifest.json
      data/splits_species_v1/species_distribution.csv

data/splits_species_v1/{train,val,test}_split.csv --> extract_perch_embeddings.py --(GPU, Perch v2,
    one pass, direct extraction, no pool/gather step)-->
        data/embeddings/perch_v2/species_v1/{train,val,test}_emb.npz
        + embedding_manifest.json (records successes and every excluded row with a reason)

{train,val,test}_emb.npz + class_list.json --> train_perch_logreg.py -->
    select only trainable species columns, L2-normalize each embedding row, sweep the pre-defined
    C grid (fit on train, score macro-AP on val), select C*, refit at C*, evaluate test once:
    checkpoints/perch/species_v1/logreg_species.joblib (OneVsRestClassifier(LogisticRegression))
    + c_selection.csv (grid: C, val macro-AP, n_species_evaluable_val)
    + species_diagnostics.csv (per-class n_iter_, converged, coef_finite -- informational, not a gate)
    + species_ap.csv (per-species AP + support, eligibility-masked)
    + macro_ap_summary.csv (single headline macro-AP number + the exact eligible species list)
    + no_bird_detection.csv (any-bird AUROC/AP, see "Metrics" below)
```

Perch is invoked exactly once, in `extract_perch_embeddings.py`. `train_perch_logreg.py` is pure
array/metadata manipulation -- no audio I/O, no TensorFlow, no GPU required, and therefore trivially
resumable/rerunnable. Only `extract_perch_embeddings.py` needs TensorFlow/kagglehub/librosa/GPU;
`prepare_species_splits.py` and `train_perch_logreg.py` need only pandas/numpy/scikit-learn/joblib,
which the standard `bioacoustics` environment already provides for `train.py`/`prepare_dataset.py`
today.

---

## Taxonomy resolution: what counts as a "species code" at all

Before any window is assigned a label state (below), every annotation code that appears in
`data/annotations_species.json` is resolved against `data/species.csv` under one rule, applied
uniformly and computed programmatically -- never as a hand-maintained list that can silently drift
from the data:

**A code only resolves to a species if its `species` column in `species.csv` is a real, two-part
species-level binomial (`Genus epithet`)** -- not a literal placeholder (`–`/blank), and not a
higher-taxon name qualified by an unresolved-species marker (a bare family/order name, or a family
name followed by `sp.` / `sp N`). Applying this rule to the current `data/species.csv` snapshot
excludes exactly six codes: `PSITTACIDAE`, `PSITTACIFORMES`, `RHACAR` (all `–`, family/order/
unresolved placeholders), `PICIDA_1` (`Picidae`, a family name), `TYRANN_SP1`
(`Tyrannidae sp 1`), and `PSITTA` (`Psittacidae sp.`). This
list is a **consequence** of the rule applied to today's data, not a separately maintained source of
truth -- if `species.csv` changes, `prepare_species_splits.py` recomputes the exclusions
automatically; the rule, not the enumerated list, is authoritative. A window carrying an excluded
placeholder code is treated identically to a window carrying no species code at all (see "Label
states" below).

**Duplicate scientific-name canonicalization**: two or more codes in `species.csv` can name the exact
same binomial species -- a data-entry duplicate, not two species. Today's snapshot has exactly two
such pairs: `ATAPIL`/`ATRPIL` (both `Atalotriccus pilaris`) and `RAMTUC`/`RHATUC` (both `Ramphastos
tucanus`). Before any support count or multi-hot encoding, `prepare_species_splits.py` groups all
resolvable codes by an exact, case-normalized match on the `species` text and maps every code in a
group to one **canonical code**, chosen deterministically (lowest code, sorted lexicographically --
e.g. `ATAPIL` over `ATRPIL`; `RAMTUC` over `RHATUC`). Every annotation carrying any code in a
duplicate group counts toward the canonical code's support and sets the canonical code's multi-hot
column; the vocabulary contains exactly one entry per distinct species, never both raw codes.

**Recorded, not silently applied**: `class_list.json` is the ordered model vocabulary
(`index`, canonical `code`, scientific `species`). `split_manifest.json.taxonomy_crosswalk` records
the excluded placeholder codes and duplicate-code merge groups, so a reviewer can audit exactly which
raw codes fed each final vocabulary entry without re-deriving the rule by hand.

---

## Label states (three): the definitions that decide what gets modeled at all

Every window is assigned exactly one of three label states by `prepare_species_splits.py`, computed
once per window, using the calls that overlap it:

| State | Condition | Target vector | Included? |
|---|---|---|---|
| `no_bird` | existing binary detector label `== 0` | all-zero multilabel vector | Yes, kept at natural prevalence -- not excluded, not downsampled. |
| `species_window` | **every** overlapping call resolves to a species code (per "Taxonomy resolution") | multi-hot vector with a `1` for each overlapping call whose species is **retained** in `class_list.json` (see "Class vocabulary and support gate") | Yes, **iff at least one** overlapping call's species is retained. If every overlapping call resolves to a species but none is retained (all calls are rare/below-gate), the window is `excluded_rare_only`. |
| `excluded_mixed_unresolved` | **any** overlapping call does not resolve to a species code (placeholder, coarse rank, or simply absent from `species.csv`) -- regardless of whether the window also has other, resolved calls | n/a | No -- excluded entirely, including any resolved calls on the same window. |

**Why a mixed window excludes the whole window, not just the unresolved call**: if a window with one
resolved species and one unresolved call kept only the resolved label, every *other* species' target
for that window would default to `0` -- but the unresolved call could just as easily be that other
species, silently injecting a false negative into every other species' negative class. Excluding the
whole window is the only way to avoid manufacturing absent labels.

**Why a retained+rare window is kept, encoding only the retained species**: a rare species call
co-occurring with a retained species call is real evidence for the retained species and does not need
to be discarded for simplicity; it simply has no vocabulary column of its own, so it contributes
nothing (neither a `1` nor a manufactured `0`) to the target vector. A window whose *only* calls are
rare (below-gate) species is dropped instead of being folded into `no_bird`, since a real bird is
present -- collapsing it to `no_bird` would corrupt the no-bird class.

Only `no_bird` and `species_window` windows are part of the modeled dataset. `excluded_rare_only` and
`excluded_mixed_unresolved` windows are recorded, with counts, in `split_manifest.json`, but never
appear in any split file.

---

## Class vocabulary and support gate: pre-registered, never tuned on test

A species (post taxonomy resolution and duplicate-code canonicalization) is **retained** in
`class_list.json` iff it meets two conditions:

1. **`n_sound_ids >= 7`**: calls resolving to this species span at least 7 distinct `sound_id`s,
   dataset-wide, before the split is constructed.
2. **Present in all three splits**: at least one `species_window` for this species ends up in each of
   train, val, and test after the split is built.

The implementation guarantees condition 2 by construction:

1. **Provisional vocabulary** = every taxonomy-resolved, canonicalized species with
   `n_sound_ids >= 7`, computed from canonical windows before assignment.
2. **Reserve positive audio files** most-constrained-species-first so every retained species has at
   least one positive `sound_id` pinned to train, validation, and test.
3. **Optimize the remaining audio-file assignments** for the 70/15/15 species and no-bird targets
   without moving those hard reservations.
4. **Validate presence after assignment**. If the reservation problem is infeasible, the split
   generator fails with the affected species instead of silently dropping it or emitting a split
   that violates the approved class contract.
5. **Record, in `split_manifest.json`**: the final vocabulary, dataset-wide `n_sound_ids` support,
   reserved/free audio-file counts, selected restart, and the successful all-splits-presence gate.

**This vocabulary gate is distinct from, and prior to, the post-split `trainable` column-selection
gate** in "Species eligibility" below, which separately checks whether a retained species' particular
train allocation has enough positive/negative examples for `LogisticRegression.fit()` to converge at
all -- a much weaker, fit-mechanics check, not a repeat of the retention decision.

**Label-aware split construction is allowed; test metrics are not.** The stratified split, and the
vocabulary retention check above, both inspect window label vectors -- including windows that end up
in val and test -- purely to build a representative, leakage-free partition and a viable class list.
This is not test-set leakage into modeling: no model parameter or hyperparameter is ever chosen by
looking at a post-fit metric computed on the test split. The one place validation *is* used to choose
something is `C` (see "Baseline classifier"), and even there, only the validation split's macro-AP is
read -- test is touched exactly once, after `C` is already fixed.

---

## Split construction: dataset-wide, `sound_id`-grouped, multilabel-stratified 70/15/15

`prepare_species_splits.py` builds **one** split, not per-project folds:

0. **Define the canonical population.** Restrict to canonical, non-overlapping windows
   (`start % window_size_samples == 0`, where `window_size_samples = window_size_sec *
   sample_rate = 5.0 * 48000 = 240000` for `segmented_v4`). This is the population used for
   stratification and for the audited distribution report -- it is *not* the population that ends up
   in the augmented training file (see step 5).
1. **Apply label states** (see "Label states") to the canonical population, keeping only `no_bird` and
   `species_window` rows. This is the modeled canonical population.
2. **Group by `sound_id`, not by site.** Every window belonging to the same `sound_id` is assigned to
   train, val, or test as a single unit -- no audio file crosses a split boundary. No coarser grouping
   (site, project, recorder) is used; `sound_id` disjointness is the full leakage-prevention contract
   for this baseline.
3. **Stratify whole `sound_id`s** to jointly approximate the overall 70/15/15 window-count split, each
   retained species' positive-window proportion, and the `no_bird` proportion -- these are the
   **only** stratification targets. Implementation may use a vendored/adapted iterative-stratification
   algorithm or a custom greedy multi-restart heuristic; the required contract is **determinism given
   (seed, restart count)**, not a specific library. Repeat with a deterministic, pre-registered set of
   seeds/restarts (default `R = 50`) and keep the assignment minimizing a documented max
   percentage-point deviation across species/`no_bird` proportions, with a lowest-seed tie-break.
4. **Project distribution is report-only.** Each project's resulting proportion across train/val/test
   is computed and written to `species_distribution.csv` for transparency, but it is **not** a
   stratification target at any priority tier -- the search in step 3 never adjusts an assignment to
   improve project balance.
5. **Add overlapping windows to train only, after the split is fixed.** Once `sound_id`s are assigned:
   - `canonical_train_split.csv`, `val_split.csv`, `test_split.csv` contain the modeled-canonical rows
     (step 1) for `sound_id`s assigned to that split. These three files, together, are the **audited
     distribution** -- the population `species_distribution.csv` reports proportions over.
   - `train_split.csv` (the **augmented** view, used for embedding/fitting) = every row in
     `canonical_train_split.csv`, **plus** every overlapping (dense, `4.0overlap`-hop) window
     belonging to a train-assigned `sound_id`, passed through the same label-state rule (step 1) and
     the final retained vocabulary. A row-level boolean column `is_canonical` distinguishes the two
     subsets within `train_split.csv`.
   - `val_split.csv` and `test_split.csv` never receive overlapping windows -- they remain canonical
     only, so validation-based `C` selection and the final test evaluation are both measured on the
     same well-defined, non-duplicated population the audited distribution describes.
6. **Record, in `split_manifest.json`**: the chosen seed, `R`, the achieved-vs-target proportion table
   for every retained species and `no_bird` (over the audited/canonical population), the resulting
   train-window count both canonical and augmented, and the label-state exclusion counts
   (`excluded_rare_only`, `excluded_mixed_unresolved`).

**Why canonical windows define the audited distribution, and overlapping windows are added only to
train**: stratifying over canonical windows only means every reported species/no-bird proportion is a
plain count over one well-defined, non-overlapping population, auditable independently of any
augmentation choice. Overlapping (1-second-hop) windows are near-duplicates of their canonical
neighbors, not independent samples -- appropriate as extra training signal once `sound_id` grouping
already guarantees they cannot leak into val/test, but not appropriate as material for val/test
evaluation (which should score genuinely distinct, non-duplicated spans) or for the stratification
search itself (which would otherwise be matching proportions over a structurally different population
than the one actually held out).

**Class balance is preserved, not corrected**: the dataset's natural `no_bird` prevalence in the
modeled canonical population is expected to be high; `prepare_species_splits.py` records the exact
figure in `split_manifest.json` rather than assuming it. This plan does not downsample `no_bird`
windows -- see "Metrics" for why prevalence-dominated aggregates (accuracy, micro-AP) are reported
only as context, never as the headline number, precisely because of this deliberately-preserved
imbalance.

**Current `species_v1` split (generated with the approved defaults)**: 68 retained species;
29,513/6,398/6,474 canonical train/val/test windows (69.64%/15.10%/15.27%); 95,536 augmented
training windows after adding train-only overlaps; 90.96% global no-bird prevalence in the canonical
modeled population. Every retained species is present in all three splits and no `sound_id` crosses a
split boundary. Whole-file grouping makes exact class ratios impossible for some species: the
generated `species_distribution.csv` reports every deviation. Per the approved decision, species are
not dropped solely for exceeding a deviation threshold; presence in all three splits is the hard
requirement, and the imbalance is disclosed in the technical-validation report.

---

## Identity discipline (all gates and joins use integer samples, never seconds)

Every identity comparison and hard gate in this pipeline is defined over `(window_id, sound_id,
start, end, sample_rate)` -- all integers. Seconds are computed exactly once,
transiently, inside `extract_perch_embeddings.py`, immediately before the `librosa.load(...,
offset=start/sample_rate, duration=(end-start)/sample_rate)` call, and are never
written to any persisted artifact, never compared for equality, and never hashed. This exists
specifically so that a future change to how seconds are rounded or represented cannot silently change
what two systems consider "the same window" -- the identity contract is the original integer geometry
PteroSet already uses everywhere else (`windows_mapping_4.0overlap_segmented_v4.json`), full stop.

`{canonical_train,train,val,test}_split.csv` (from `prepare_species_splits.py`) columns: `window_id,
dataset, sample_rate, sound_id, start, end, project, is_canonical, label_state, spec_name,
sound_filename, target_codes, target_vector`. `target_vector` is a JSON-encoded 0/1 vector in
`class_list.json` order. `sound_id` is the only
grouping/identity key the split relies on; `project` is retained as an ordinary, report-only metadata
column (see `species_distribution.csv`), not a join or gate key downstream of
`prepare_species_splits.py`. No seconds column exists in this file or in any downstream artifact.

---

## Failure handling: fail-loud audio loading, enforced globally and per split

**Embedding manifest keys** (`embedding_manifest.json`, written by `extract_perch_embeddings.py`):

```json
{
  "created_at_utc": "...",
  "git_commit": "<sha>",
  "model": {"name": "perch_v2", "kaggle_slug": "google/bird-vocalization-classifier/tensorFlow2/perch_v2",
            "local_dir": "checkpoints/perch/model_v2", "embedding_key": "<resolved>", "embedding_dim": 1536},
  "identity_hash_inputs": {
    "split_manifest_sha256": "<sha256 of data/splits_species_v1/split_manifest.json>",
    "model_local_dir_sha256": "<sha256 over a sorted listing of the vendored SavedModel dir's files>",
    "extraction_params": {"target_sample_rate": 32000, "window_sec": 5.0, "target_peak": 0.25, "batch_size": 64}
  },
  "dependency_versions": {"tensorflow": "2.21.0", "scikit-learn": "<resolved>", "kagglehub": "1.0.0",
                          "numpy": "2.2.5", "librosa": "0.11.0"},
  "failure_ceiling": {
    "global_max_failures": 0,
    "per_split_max_failures": 0,
    "note": "Defaults are pre-registered. Any reviewed override must be set before extraction and recorded verbatim."
  },
  "counts": {"train": {"requested": 0, "succeeded": 0, "excluded": 0},
             "val": {"requested": 0, "succeeded": 0, "excluded": 0},
             "test": {"requested": 0, "succeeded": 0, "excluded": 0}},
  "excluded": {}
}
```

**Rule**: `extract_perch_embeddings.py` aborts the entire run (`SystemExit`) the instant either
ceiling in `failure_ceiling` is exceeded -- checked **both** globally **and** per-split (train/val/
test), because a failure rate that looks small in aggregate could still mean one split's audio is
disproportionately degraded, which would quietly bias that split's labels. Both ceilings default to
`0` (abort-on-first-failure); a non-zero override must be an explicit, reviewed, pre-registered CLI
flag value, never a number chosen because "that's how many failures actually happened." A window that
fails to load is never silently treated as an all-zero embedding and folded into training as a valid
negative; it is either fixed or the run aborts.

---

## Species eligibility (trainable columns, computed once per split)

For each retained species in `class_list.json`, compute positive/negative window counts in each of
`train_emb.npz`, `val_emb.npz`, `test_emb.npz`. Derive, per split, whether the species is **present**
(`pos >= 1`) and **has both classes** (`pos >= 1 and neg >= 1`):

| Category | Condition | Meaning |
|---|---|---|
| `trainable` | both classes present in train | Enough evidence to fit that species' binary classifier at all. |
| `val_evaluable` | both classes present in val | Val macro-AP (for `C` selection) is defined for this species. |
| `test_evaluable` | both classes present in test | Test macro-AP (final headline) is defined for this species. |

Because the class vocabulary gate already requires presence in all three splits (see "Class
vocabulary and support gate"), these are expected to hold for essentially every retained species; they
remain an explicit, checked safety net rather than an assumption. `train_perch_logreg.py` passes only
`trainable` species to `OneVsRestClassifier.fit()`, avoiding scikit-learn aborting on a constant
all-negative/all-positive column. Predictions are reindexed to the fixed `class_list.json` order;
non-trainable species receive `NaN` scores plus their explicit status, never a fabricated zero
probability. If no species is `trainable`, the run fails with a clear error before fitting.

---

## Baseline classifier: `C` selected on validation, `class_weight` fixed, test touched once

`OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced"))` --
no `multi_class` kwarg (removed in scikit-learn >= 1.7, raises a constructor-time `TypeError`).

- **`class_weight="balanced"`**: fixed, not searched. The modeled dataset's `no_bird` share is
  deliberately preserved at its natural, high prevalence and species support is long-tailed;
  unweighted logistic regression would trivially minimize loss by predicting all-negative for rare
  species. `class_weight="balanced"` reweights each binary sub-problem by
  `n_samples / (n_classes * bincount)` from the train split alone -- a formula-derived setting computed
  once at fit time, never a value searched against val/test AP.
- **`C` is selected from a small, pre-defined grid using validation macro-AP, evaluated on test
  exactly once**: for each `C` in a pre-registered grid (default `{0.01, 0.1, 1.0, 10.0}`), fit on
  `train_emb.npz` (trainable columns only, L2-normalized), score macro-AP over `val_evaluable` species
  on `val_emb.npz`, and record every `(C, val_macro_ap, n_species_val_evaluable)` row in
  `c_selection.csv`. Select `C* = argmax(val_macro_ap)` (deterministic tie-break: lowest `C`). Refit
  once at `C*` on train, and evaluate on `test_emb.npz` exactly once to produce
  `macro_ap_summary.csv`. **No parameter is ever adjusted after seeing a test-split metric** -- `C*`
  is fixed before test is touched at all.
- Embeddings are L2-normalized (`sklearn.preprocessing.normalize(X, norm="l2")`) independently per
  split before fitting, scoring, or predicting. No `StandardScaler` is fitted or persisted; the L2
  operation is stateless and therefore cannot leak split information.

---

## Metrics: macro AP primary, per-species AP, and a separate no-bird/any-bird detection metric

- **`macro_ap_summary.csv`** (headline): macro-averaged Average Precision over `test_evaluable`
  species, computed on `test_emb.npz` at the validation-selected `C*`. Reports `n_species_evaluable`
  and its exact members. This is the single headline scalar.
- **`species_ap.csv`**: per-species AP and support (`train_pos`, `test_pos`), reindexed to the full
  vocabulary, eligibility-masked.
- **`no_bird_detection.csv`**: "how well does this model separate any-bird windows from no-bird
  windows at all?" Defined via `any_bird_score = 1 - prod_i(1 - p_i)` (probability at least one
  species is present, from the independent OvR per-species probabilities `p_i`). Reported as AUROC
  and AP of `any_bird_score` against the binary `no_bird`-vs-`species_window` label, over
  `test_emb.npz`.

**Why a separate no-bird metric is required**: `no_bird` prevalence is deliberately preserved at its
natural, high rate, so any aggregate that rewards predicting the majority class (overall accuracy,
micro-averaged AP/F1) is misleading by construction. Macro AP over evaluable species and the
`any_bird_score` AUROC/AP are the two headline metrics; accuracy/micro-AP may be reported as context
only.

**Thin test support is disclosed, not hidden**: whole-file grouping plus the long-tailed species
distribution leaves some retained species with only one or a few positive canonical test windows.
Those species remain in the approved vocabulary because presence in all three splits, not a minimum
test-window count, is the hard requirement. Every result table must show `test_pos`; the technical
validation report must separate or sensitivity-check thin-support species before interpreting the
headline macro-AP as broad evidence across all retained classes.

**Explicitly not in scope** (see "What (scope)"): a single-label comparator, cross-fold metrics, an
arbitrary coefficient-norm exclusion gate, any pooling, and a Perch v1-vs-v2 ablation.

---

## Numerical diagnostics: informational, not an exclusion gate

For every fitted `trainable`-species `LogisticRegression` estimator (at the final `C*`), record, per
species, into `species_diagnostics.csv`:

- `n_iter_` and the `max_iter` it was given; `converged = n_iter_ < max_iter` (the `sklearn`
  `ConvergenceWarning` is captured, not suppressed).
- `coef_finite = True` iff every entry of `coef_` and `intercept_` is finite (no `NaN`/`Inf`).
- `coef_l2_norm = np.linalg.norm(coef_)`, reported for information only.

`coef_finite` is a hard correctness requirement: a species whose fit produced non-finite coefficients
cannot produce valid probabilities and is forced `evaluable = False`, excluded from
`macro_ap_summary.csv` and `no_bird_detection.csv`. `converged` and `coef_l2_norm` are reported for
diagnostic visibility only -- **no arbitrary coefficient-norm ceiling is used as an automatic
exclusion gate**; a large-but-finite coefficient norm is surfaced to a human reader, not silently
used to drop a species from the headline number.

`species_diagnostics.csv` is reindexed to the full class vocabulary; non-trainable species contain
`NaN` for fit-derived fields plus their explicit eligibility status. No diagnostics are fabricated for
estimators that were never fit.

---

## Environment: the fallback-env decision applies to `extract_perch_embeddings.py` only

`prepare_species_splits.py` and `train_perch_logreg.py` run in the standard `bioacoustics` conda
environment already used by `train.py`/`prepare_dataset.py` -- they need only pandas/numpy/scikit-
learn/joblib, all already present there, and neither imports TensorFlow, kagglehub, or librosa. No
smoke test or fallback logic applies to these two scripts.

Only `extract_perch_embeddings.py` needs TensorFlow, kagglehub, librosa, and a CUDA-capable GPU
(Perch v2's SavedModel is XLA-compiled CUDA-only). Its environment is decided by a one-time smoke
test using the extractor itself, so the CUDA-library bootstrap is applied before TensorFlow imports:

```bash
conda activate bioacoustics
python extract_perch_embeddings.py --extract --splits val --limit 1 \
    --output-dir data/embeddings/perch_v2/species_v1_smoke
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
resolved Kaggle model version, and any redistribution constraints in `embedding_manifest.json` and
the results report. Model download and publication are blocked if the license cannot be verified.

---

## Files

```
prepare_species_splits.py    # windows_mapping JSON + annotations_species.json + species.csv ->
                              #   taxonomy resolution (exclude placeholder codes, canonicalize
                              #   duplicate-scientific-name codes -- "Taxonomy resolution"),
                              #   >=7-distinct-sound_ids + present-in-all-3-splits vocabulary
                              #   gate ("Class vocabulary and support gate"), 3-state label
                              #   derivation (no_bird / species_window / excluded_*), sound_id-
                              #   grouped multilabel-stratified 70/15/15 split (species/no_bird
                              #   proportions the only stratification target; project report-only)
                              #   with deterministic seed/restarts, canonical-window population
                              #   for stratification with overlapping windows added to train only
                              #   after assignment; writes canonical_train_split.csv,
                              #   train_split.csv (augmented), val_split.csv, test_split.csv,
                              #   class_list.json (ordered canonical vocabulary),
                              #   split_manifest.json (proportions achieved vs. target, seed, R,
                              #   exclusion counts, taxonomy crosswalk, vocabulary support,
                              #   hard-reservation audit, report-only project counts),
                              #   species_distribution.csv (achieved species/no_bird proportions)
extract_perch_embeddings.py  # copied/adapted from eval_perch_ecotype.py / extract_perch_3class.py:
                              #   CUDA bootstrap, download_perch_v2, load_perch_v2,
                              #   inspect_model_outputs, load_audio_segment (computes seconds
                              #   transiently from start/sample_rate), extract_embeddings;
                              #   directly extracts train/val/test NPZ from the split CSVs
                              #   (train_emb.npz from the augmented train_split.csv); global AND
                              #   per-split failure-ceiling enforcement; embedding_manifest.json
                              #   writer (successes + reasoned exclusions), license fields enforced
                              #   and recorded. NPZ fields: embeddings float32 [N,1536],
                              #   target_vector uint8 [N,K], target_codes, label_state, window_id,
                              #   sound_id, start, end, sample_rate, sound_filepath, sound_filename,
                              #   dataset, project, is_canonical.
train_perch_logreg.py        # OneVsRestClassifier(LogisticRegression(solver="lbfgs",
                              #   max_iter=1000, class_weight="balanced")) -- no multi_class kwarg
                              #   (removed in scikit-learn >= 1.7, raises TypeError if passed);
                              #   fits only trainable target columns; sweeps a pre-defined C grid,
                              #   selects C* by validation macro-AP, refits at C*, evaluates test
                              #   once; reindexes predictions to the full class vocabulary; writes
                              #   c_selection.csv, species_diagnostics.csv, species_ap.csv,
                              #   macro_ap_summary.csv, no_bird_detection.csv
```

No new Python package/`__init__.py` layering -- flat, dual-purpose scripts (both CLI and importable),
mirroring the reference repo's own convention. Exact CLI flag names above may be finalized during
implementation; the artifact names, schemas, and script responsibilities are the binding contract.

---

## Commands

Exact flag names are illustrative; implementation may finalize them, but the artifacts each command
must produce are binding (see "Files").

```bash
# Phase 0 -- bootstrapped GPU/model smoke test (limited artifact; not a full Phase 2 result)
conda activate bioacoustics
python extract_perch_embeddings.py --extract --splits val --limit 1 \
    --output-dir data/embeddings/perch_v2/species_v1_smoke

# Phase 0 -- one-time model vendoring
python extract_perch_embeddings.py --download_v2

# Phase 1 -- dataset-wide species split (sound_id grouping, >=7-sound_id + all-3-splits
# vocabulary gate, canonical-window stratification, overlapping windows added to train only)
python prepare_species_splits.py \
    --windows-mapping data/windows_mapping_4.0overlap_segmented_v4.json \
    --annotations-identification data/annotations_identification.json \
    --annotations-species data/annotations_species.json \
    --species-csv data/species.csv \
    --min-sound-ids 7 \
    --seed 42 --num-restarts 50 \
    --output-dir data/splits_species_v1

# Phase 2 -- direct train/val/test extraction (the only GPU-bound, Perch-bound step)
python extract_perch_embeddings.py \
    --extract \
    --split-dir data/splits_species_v1 \
    --output-dir data/embeddings/perch_v2/species_v1 \
    --batch-size 64 --max-failures 0 --max-failures-per-split 0

# Phase 3 -- training + evaluation, standard bioacoustics env
python train_perch_logreg.py \
    --embeddings-dir data/embeddings/perch_v2/species_v1 \
    --class-list data/splits_species_v1/class_list.json \
    --out-dir checkpoints/perch/species_v1 \
    --c-grid 0.01,0.1,1.0,10.0 --class-weight balanced --n-jobs 16
```

---

## Phased milestones and acceptance criteria

- **Phase 0 -- complete**: the bootstrapped TensorFlow 2.21 runtime sees four GPU devices; the
  vendored `checkpoints/perch/model_v2/` executes a real PteroSet window and exposes the selected
  `embedding` output with shape `(1, 1536)`. The model/code licenses are recorded as Apache-2.0 with
  source URLs, and the immutable SavedModel content hash is recorded in the embedding manifest.
- **Phase 1 -- complete**: `prepare_species_splits.py` produces `class_list.json`,
  `split_manifest.json`, `species_distribution.csv`, and the four split CSVs. Acceptance: no
  `sound_id` in more than one split; `canonical_train_split.csv`/`val_split.csv`/`test_split.csv`
  contain canonical windows only; `train_split.csv` is a strict superset adding only overlapping,
  train-assigned windows (flagged `is_canonical = False`); every retained species has
  `n_sound_ids >= 7` and appears in all three splits; `split_manifest.json.taxonomy_crosswalk`
  matches "Taxonomy resolution"
  (currently: `PSITTACIDAE`, `PSITTACIFORMES`, `RHACAR`, `PICIDA_1`, `TYRANN_SP1`, `PSITTA`
  excluded;
  `ATAPIL`/`ATRPIL` and `RAMTUC`/`RHATUC` canonicalized); every excluded window is accounted for with
  a count; project counts in `split_manifest.json` are explicitly report-only.
- **Phase 2 -- complete**: `extract_perch_embeddings.py` produced
  `data/embeddings/perch_v2/species_v1/{train,val,test}_emb.npz` and
  `embedding_manifest.json`: train `(95,536, 1,536)`, validation `(6,398, 1,536)`, and test
  `(6,474, 1,536)`, all float32, finite, and row-for-row aligned with the source CSV identity and
  target fields. All three splits have `excluded = 0` and `padding = 0`; the manifest records the
  class/split/model/extractor hashes, fixed batch size 64, GPU devices, dependencies, licenses, and
  source URLs. Total artifact storage is approximately 724 MiB. Phase 2 passed 114 focused extractor
  tests and 166 total repository tests.
- **Phase 3 -- complete**: `train_perch_logreg.py` fit all 68 trainable columns across the
  pre-registered C grid and selected `C*=0.1` by validation macro-AP (`0.4379`). The train-only final
  refit produced finite, converged estimators for all species. Test was evaluated after selection:
  macro-AP `0.4089` versus the prevalence-only baseline `0.0020`; support sensitivity is `0.4817`
  for `test_pos>=5` and `0.5738` for `test_pos>=10`; any-bird/no-bird AUROC is `0.9437` and AP is
  `0.7133`. All planned model, prediction, diagnostic, metric, selection, and manifest artifacts were
  generated under `checkpoints/perch/species_v1/`.
- **Phase 4 -- complete**: `docs/implementation/species-linear-probe-v1/results.md` documents the
  split/embedding/model configuration, C selection, headline and support-sensitive results,
  diagnostics, artifacts, limitations, and the technical-validation-only framing.

---

## Tests (minimal, highest-value only)

1. `test_split_group_integrity.py` -- no `sound_id` is ever split across train/val/test.
2. `test_label_state_derivation.py` -- `no_bird` / `species_window` / `excluded_rare_only` /
   `excluded_mixed_unresolved` classification, including a retained+rare window (kept, encoding only
   the retained species) and a resolved+unresolved window (excluded entirely).
3. `test_taxonomy_resolution.py` -- placeholder codes excluded, duplicate scientific names
   canonicalized with combined support; reproduces the six current exclusions and two current
   canonicalization pairs against the real `data/species.csv`.
4. `test_vocabulary_support_gate.py` -- `n_sound_ids < 7` excludes a species; hard reservation gives
   every retained species a positive `sound_id` in all three splits; an infeasible reservation fails
   with the affected species rather than silently changing the vocabulary.
5. `test_stratification_proportions.py` -- achieved species/no-bird proportions are within tolerance
   of target; project counts are reported but never influence assignment choice; same seed reproduces
   the identical assignment.
6. `test_canonical_vs_augmented_train.py` -- `val_split.csv`/`test_split.csv` are canonical-only;
   `train_split.csv` is `canonical_train_split.csv` plus correctly-flagged overlapping rows for
   train-assigned `sound_id`s only.
7. `test_perch_io.py` -- `load_audio_segment` produces exact sample counts from a `start`/
   `sample_rate` pair, converting to seconds only inside the loader, never persisting it.
8. `test_l2_normalization.py` -- every classifier input row has unit L2 norm (all-zero rejected
   upstream).
9. `test_trainable_column_selection.py` -- only `trainable` columns reach `OneVsRestClassifier.fit()`;
   outputs reindexed to the full class list with `NaN` for excluded columns.
10. `test_c_selection.py` -- `C*` is the argmax-validation-macro-AP grid entry (lowest-`C` tie-break);
    test is never consulted before `C*` is fixed.
11. `test_any_bird_score.py` -- `any_bird_score = 1 - prod(1 - p_i)` is correct and monotonic in each
    `p_i`.
12. `test_diagnostics_no_fabrication.py` -- non-trainable species have `NaN` diagnostics; non-finite
    coefficients force `evaluable = False`.

No broader unit-test suite is proposed beyond the above, matching the reference repo's own convention
and the principle of testing the parts that can silently corrupt data or hide a bad number (grouping,
label derivation, stratification, audio normalization, eligibility, `C` selection), not well-tested
library calls (`sklearn.fit`).

---

## Key operational failure modes

- Perch v2's SavedModel is CUDA-only; the CUDA bootstrap must re-exec the process before any
  TensorFlow import, or GPU libraries are silently not found.
- A window's audio fails to load: abort the whole run once either failure ceiling (default `0`) is
  exceeded, rather than ever treating a silently zero-filled embedding as a valid negative.
- `kagglehub` failures only affect the one-time `--download_v2` step; every later run uses the
  vendored local model directory.
- `LogisticRegression(..., multi_class=...)` must never be passed on scikit-learn >= 1.7 (removed,
  raises `TypeError`); this plan's classifier construction does not pass it.
- A species failing `n_sound_ids >= 7` or the present-in-all-3-splits check is excluded entirely, not
  trained on too little evidence; the recheck must reach a fixed point before the vocabulary is final.
- A placeholder scientific-name code is never treated as resolved just for having a `species.csv` row;
  duplicate-named codes are canonicalized before support counting.
- Non-finite coefficients force `evaluable = False`; slow convergence or a large-but-finite norm is
  reported, not auto-excluded by an arbitrary threshold.
- `C` is selected only by validation macro-AP; test is scored exactly once, after `C*` is fixed --
  never add a second, test-informed tuning pass.
- `no_bird` prevalence is deliberately preserved at its natural high rate; accuracy/micro-averaged
  aggregates are misleadingly high for an always-`no_bird` predictor and are never a go/no-go signal.

---

## Future migration path

Promote the copy/adapted scripts to a shared cross-repo package only when a third consumer needs the
same Perch-loading logic, or when a CUDA/TF upgrade requires a synchronized fix across 2+ repos.
Revisit fine-tuning Perch's backbone or an alternative encoder (BirdNET, SurfPerch) only after this
frozen-embedding baseline has a measured `macro_ap_summary` result to beat. Revisit leave-one-site-out
/ leave-one-project-out / unseen-site or unseen-project species generalization, segment-level pooling,
and a single-label comparator only as genuine follow-on questions once this window-level baseline is
trusted -- none of them is answered or approximated by this document.

---

STATUS: FINAL
