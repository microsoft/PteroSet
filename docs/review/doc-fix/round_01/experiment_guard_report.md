# Experiment Guard Report — PteroSet Perch 2 Baseline: Proposed New Stratified Split

**Reviewed proposal**: replace the split used for the Perch v2 species linear-probe baseline with a
new train/val/test distribution, stratified across the *whole* dataset (all 5 projects pooled),
preserving species proportions, excluding "unresolved" bird windows, and encoding genuine no-bird
windows as an all-zero multilabel target.

**Files inspected**: `docs/design/perch2_species_linear_probe_plan.md` (FINAL, Round 8),
`docs/design/CHANGELOG.md`, `docs/design/round_01/architect_minimalist_proposal.md`,
`docs/design/round_02/architect_experiment_proposal.md`, `docs/review/doc-fix/round_01/documenter_report.md`
(pending, unapplied change plan), `prepare_dataset.py::run_splits` (lines 338-490), `data/config.yaml`,
`data/data_reader.py::add_annotations`, `data/folds_segmented_v4/fold_4_PPA4_segmented/*.csv`,
`requirements.txt`.

**No files were edited.**

---

## 0. Governing finding: this proposal contradicts the project's own converged, FINAL design decision

This is the single most important fact this review surfaces, and it must be resolved before any
split code is written.

`docs/design/perch2_species_linear_probe_plan.md` is marked **`STATUS: FINAL`**, converged in Round 8
"with zero new findings" (line 3), and states explicitly in its own second paragraph:

> "evaluation uses 5 fixed leave-one-project-out (LOPO) folds, not one seed-selected stratified
> split" (line ~23)

`docs/design/CHANGELOG.md` (line 23) restates the same decision as load-bearing design history, and
`docs/design/round_02/architect_experiment_proposal.md` (lines 78-80) gives the *reason*, not just the
rule:

> "PteroSet's fold design is fixed and non-negotiable: leave-one-**project**-out ... Project is
> confounded with recording site, equipment, habitat, and species pool. There is **no seed to sweep**
> ... **Do not port `StratifiedGroupKFold` to PteroSet's primary folds.**"

That same document codifies this as a hard leakage rule (**L6**, line 261): any species-stratified,
non-LOPO split "MUST be clearly labeled `[SECONDARY — in-distribution only, does not test cross-project
generalization]` ... and must never be blended with, averaged into, or substituted for the LOPO
cross-fold summary in any headline claim."

Separately, there is already a **pending, unapplied** doc-fix change plan
(`docs/review/doc-fix/round_01/documenter_report.md`) proposing to narrow the FINAL plan's scope to a
*single existing LOPO fold* (`fold_4_PPA4_segmented`, held-out project PPA4) as a technical-validation
baseline — still project-based, still not a new stratified split, and still not yet applied to the
plan document.

**Consequence**: the proposal under review is a *third*, structurally different split design from
both the currently-committed FINAL plan and the currently-pending doc-fix narrowing. It is not
inherently invalid — a pooled, species-stratified split answers a genuinely different and legitimate
question ("does frozen Perch v2 carry linearly separable species signal at all, in-distribution?")
than LOPO does ("does that signal survive an unseen recording site/equipment/habitat?") — but running
it *instead of*, rather than *in addition to and clearly subordinate to*, the project-based split
without disclosure would violate the repo's own pre-registered L6 rule and would silently answer a
different scientific question than the one the FINAL plan was designed to answer. This must be an
explicit, disclosed decision, not an implicit substitution.

**Recommendation (CRITICAL, process gate)**: before writing any split code, the user/team must record
one of the following, in writing, in the plan document or its changelog:
- **(a)** This stratified split is a **new, additional, explicitly SECONDARY** experiment per rule
  L6 — the existing LOPO folds (or the narrowed single fold_4 baseline) remain the primary claim, and
  the stratified split's results are reported separately, always labeled in-distribution-only, never
  averaged with or substituted for the LOPO number; **or**
- **(b)** The team is deliberately superseding the FINAL/Round-8 LOPO decision with a new one, in
  which case `perch2_species_linear_probe_plan.md`'s `STATUS: FINAL` and the CHANGELOG must be updated
  to record *why* the project-confound rationale in Round 2 (site/equipment/habitat leakage across the
  train/test boundary) no longer applies or is being knowingly accepted as an unmeasured risk.

Everything below assumes the split proceeds under one of these two disclosed framings — the technical
recommendations are the same either way, but the scientific claim that may be attached to the results
differs sharply (see §6).

---

## 1. Reproducibility

- **Seeds**: `data/config.yaml`'s existing `splits.random_state: 42` (line ~48) is the only seed
  currently governing any split in this repo (`prepare_dataset.py::run_splits`, line 362, 478). The
  new stratified split must reuse this same config-driven seed field (or a new, equally logged
  `splits.stratified_random_state`), never a bare literal buried in a notebook/script argument.
  **Recommendation**: log the resolved seed value inside the split manifest itself (see §3), not only
  in `config.yaml`, so a stale config change after split generation cannot silently misrepresent what
  seed actually produced the persisted CSVs.
- **Determinism of the stratification algorithm itself**: `scikit-learn` (already the only ML
  dependency listed in `requirements.txt` relevant here) has **no built-in multilabel-aware stratified
  splitter** — `StratifiedShuffleSplit`/`StratifiedKFold` are single-label only and will raise or
  silently mis-stratify against a multi-hot species target. The commonly used tool for this
  (`iterative-stratification`'s `MultilabelStratifiedShuffleSplit`) is **not in `requirements.txt`**
  and must be added and pinned explicitly if used; alternatively a custom greedy/iterative algorithm
  must be implemented and unit-tested (see §7 test list already established as precedent in the FINAL
  plan, e.g. `test_eligibility_categories.py`). Either path must be seeded and its determinism verified
  by a regression test (rerun with the same seed twice, assert identical `window_id` sets per split).
- **Environment/config logging**: same discipline the FINAL plan already requires for the Perch
  pipeline (`pool_manifest.json`'s `dependency_versions` block) must extend to whatever split script is
  written: record `scikit-learn`/`iterative-stratification` version, the config file's resolved
  values, and the git commit hash in a `split_manifest.json` alongside the CSVs.
- **Config-vs-runtime match**: `prepare_dataset.py::run_splits` already reads `config.splits.val_size`
  and `config.splits.random_state` directly from the loaded config object (lines 362-363) with no
  hidden default override observed — this discipline must carry over unchanged to the new split
  script; do not introduce a second, divergent default (e.g. a CLI `--seed` flag whose default differs
  from `config.yaml`'s value).

## 2. Data Integrity

- **Leakage risk — HIGH, must be controlled at the file/group level, not just the window level.**
  PteroSet windows are 5 s with 4 s overlap (`data/config.yaml`: `window_size_sec: 5.0,
  overlap_sec: 4.0`) — adjacent windows from the same audio file share up to 80% of their waveform
  samples. The *existing* split mechanism already treats this as a first-class leakage control: train/
  val partitioning uses `GroupShuffleSplit(groups=sound_id, ...)` (`prepare_dataset.py` line ~472-478)
  so no single audio file straddles train and val, and the test set is additionally restricted to
  **non-overlapping** windows only (`start % window_size_samples == 0`, line ~458). **Any new
  stratified split must preserve both controls**: (1) group by `sound_id` (or the finer segment id
  where PPA1's 9 s-stride crossfade applies, per `CLAUDE.md`'s v2→v3 note) so overlapping sibling
  windows never split across train/val/test, and (2) restrict the eval splits (val and/or test) to
  non-overlapping windows, exactly as today. A window-level (non-grouped) stratified split would place
  near-duplicate, overlap-sharing windows across train and test and silently inflate every reported
  metric — this is the single highest-priority technical risk in this proposal.
- **Residual site/equipment confound, even with correct grouping.** Grouping by file prevents the
  overlap leak but does **not** prevent every project's recording equipment/site acoustic signature
  (background noise floor, gain staging, habitat soundscape) from appearing in *both* train and test
  once all 5 projects are pooled and stratified together — this is precisely the confound the LOPO
  design exists to control for (Round 2, §1.4/§4.3, cited above). This is not a bug to "fix" in the
  split code; it is an inherent property of any pooled, non-LOPO split and must be disclosed as a
  scope limitation on the resulting claim (§6), not silently absorbed into the metric.
- **"Unresolved" bird windows must be excluded from both the multilabel target *and* the stratification
  denominator, not merged into no-bird.** This gap was already diagnosed once in this repo's history:
  `docs/review/doc-fix/round_01/documenter_report.md` §1.3, citing `data/data_reader.py::add_annotations`
  (lines 63-87): `annotations_species.json` only includes a RAVEN row if its `Determination` matches a
  `species.csv` code (`category_match`, line 78-82); rows that fail this match are silently dropped
  from the species-annotation file even though the same window may carry a valid `AVEVOC` (any-bird)
  annotation in `annotations_identification.json`. A window can therefore be bird-positive with **zero**
  resolvable species rows. The six-state schema already defined once in
  `docs/design/round_01/architect_minimalist_proposal.md` §4.2 is the correct population definition to
  reuse verbatim:
  - `has_bird` — any `AVEVOC` (identification-level) annotation overlaps the window.
  - `any_species_known` — at least one overlapping annotation resolved to a `species.csv` code.
  - `unresolved_bird` — `has_bird == 1 and any_species_known == 0` — **exclude these windows entirely**
    from training, evaluation, and from the species-proportion counts used to *build* the stratified
    split. Silently collapsing them to all-zero (as the user's own proposal risks doing if
    "unresolved" and "no-bird" are conflated) would inject an unknown number of false negatives into
    every species' negative class simultaneously — already flagged in this repo's own design history
    (`round_02/architect_experiment_proposal.md`, line 74) as "the single largest correctness risk."
  - `usable_strict` — `unresolved_bird == 0` — this, and only this, is the population that should be
    stratified, trained, and evaluated on.
  - **All-zero target is correct only for genuine no-bird windows**: `has_bird == 0` implies, by
    construction, no overlapping species annotation of any kind, so an all-zero multi-hot vector is a
    true negative for every species, not an artifact of missing annotation. This is the one case where
    the user's "no-bird → all-zero" instruction is exactly correct — the risk is solely in making sure
    `unresolved_bird` windows never reach this same all-zero encoding path.
- **Preprocessing consistency**: the stratification/grouping decision must be computed once from
  `usable_strict` windows' species labels, persisted, and never recomputed differently for train vs.
  val vs. test (e.g. do not compute per-split label vocab or per-split normalization statistics —
  L2-normalization of embeddings is already correctly stateless per the FINAL plan and requires no
  change).

## 3. Training Configuration

- **Split manifest / checkpoint-equivalent artifact**: the FINAL plan's `pool_manifest.json` /
  `fold_manifest.json` pattern is directly reusable here. Persist a `split_manifest.json` recording:
  `resolved_seed`, `windows_mapping_json_sha256`, `annotations_species_json_sha256`, the six-state
  schema version/exclusion rule, the stratification algorithm + package version, per-split window
  counts, per-species per-split positive/negative counts (i.e. the same eligibility categories the
  FINAL plan already defines: `trainable`, `structurally_unseen`, `test_absent`, `test_single_class`,
  `evaluable` — these concepts are split-topology-agnostic and apply verbatim to a single stratified
  split, not only to LOPO folds).
- **Metric consistency**: reuse the FINAL plan's baseline classifier and eligibility-gating mechanism
  unchanged (see §5) — do not invent a second, divergent eligibility or metrics implementation for this
  split type.
- **Model-selection discipline**: the FINAL plan's `train_perch_logreg.py` design fixes `C=1.0` and
  explicitly defers any `C`-grid/calibration sweep as out of scope ("What (scope)" section, and Round-8
  "Future migration path"). With no hyperparameter search specified, there is currently **nothing** in
  this pipeline that reads back from a validation metric into a decision that changes the model or
  the training data before the single, final evaluation pass. This is the load-bearing fact behind §4
  and §6 below.

## 4. Is a validation split necessary?

**Decision: val is not required for model selection under the currently-specified baseline (fixed
`C=1.0`, no early stopping, no threshold sweep), but should be retained in reduced form only if its
purpose is pre-registered — it must not exist as an unused, undocumented artifact.**

Reasoning:
- `sklearn.linear_model.LogisticRegression(solver="lbfgs")` has no early-stopping mechanism that
  consumes a held-out validation set during `.fit()` — `max_iter` is a fixed convergence ceiling, not
  a val-monitored stopping criterion.
- The FINAL plan's own out-of-scope list explicitly defers "a `C`-grid/calibration sweep" — i.e., the
  one thing a val split would be *for* in this baseline is pre-registered as **not happening** in this
  round.
- The pending doc-fix (`documenter_report.md`, line ~240) already reaches the same conclusion for the
  single-fold baseline: val-split diagnostics are **"Reported, not decision-driving ... computed only
  to sanity check"**.
- Given rare species require `min_class_support=10` training positives just to be `trainable`
  (FINAL plan default), every window siphoned into an unused val split is evidence removed from both
  train (reduces trainable species count) and test (reduces eligible-species AP precision) for no
  compensating benefit.

**Recommendation**:
- **CRITICAL (pre-registration)**: state explicitly, before splitting, whether val will be used for
  anything this round (e.g. a future `C`-grid sweep, per-class threshold calibration for the report,
  or purely descriptive monitoring). If the answer is "nothing yet," either (a) keep a small val split
  and label every val-derived number in the results report as diagnostic/non-decision-driving, exactly
  as the doc-fix precedent already does, or (b) fold val into train to maximize rare-species support and
  drop the val split entirely for this baseline, reserving a val mechanism for if/when a real sweep is
  added later. Either is defensible; leaving it undecided is not.
- **SUGGESTION**: if kept, size val conservatively (the existing `config.splits.val_size: 0.15` is a
  reasonable starting point) and report its per-species eligibility table exactly like train/test, so
  an unused-but-present val split doesn't quietly hide a rare species that has zero val representation
  and would have failed a future sweep silently.

## 5. How test remains untouched despite label-aware stratification

Using species labels to **decide split membership** is not, by itself, leakage — this is exactly what
`StratifiedKFold`/`MultilabelStratifiedShuffleSplit` do by design, and it is categorically different
from leakage patterns like fitting normalization statistics on the full dataset or tuning
hyperparameters against test performance. The distinction that must be enforced operationally is:

1. **Compute stratification from aggregate, group-level label statistics only, once.** Aggregate each
   `sound_id`'s (or segment's) `usable_strict` window labels (e.g. union or per-species max) *before*
   assigning that group to a split, then run the multilabel-stratified group assignment once, with the
   fixed logged seed from §1. This uses label information to build the partition — it never uses it to
   adjust a trained model or a reported metric afterward.
2. **Freeze and hash the result immediately.** Write `{train,val,test}_split.csv` plus
   `split_manifest.json` (§3) with a content hash, exactly as the FINAL plan already does for its
   fold artifacts. Any later change to the split must be a new, explicitly versioned split (e.g.
   `species_stratified_v1` → `v2`), never a silent in-place regeneration.
3. **No component downstream of the split may read test rows before the single, final,
   pre-registered evaluation.** Concretely: `min_class_support`, `C`, `coef_norm_ceiling`, and the
   eligibility thresholds are fixed constants from the FINAL plan, not fit or adjusted using val or
   test performance (§3's model-selection-discipline point) — this is the actual mechanism that keeps
   test "untouched": there is no decision point in the current design where a person or script could
   look at test and change anything about the model before the one reported number.
4. **Add a regression test mirroring the FINAL plan's own convention**
   (`test_fold_cache_gates.py`-equivalent): assert that re-running the split script with the same seed
   and inputs reproduces byte-identical `window_id` membership per split, and assert no `sound_id`
   appears in more than one split (the group-leakage gate from §2). This is the direct analog of the
   FINAL plan's existing "test project matches held-out project" invariant (Phase 1 acceptance
   criterion) and should be treated with the same non-negotiable priority.
5. **Do not compute the go/no-go decision, or any reported number, more than once against test.**
   If the first evaluation pass does not clear the pre-registered acceptance margin (§6), that is a
   valid negative result to report — re-splitting, re-tuning `C`, or re-defining eligibility until test
   performance improves would retroactively turn the "untouched" test set into a tuned one.

## 6. Primary metrics, baseline classifier settings, and scientifically supportable claims

**Baseline classifier — reuse the FINAL plan's mechanism verbatim, only the split topology changes**:
`OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0))` (no `multi_class`
kwarg — removed in scikit-learn >= 1.7, already correctly avoided by the FINAL plan), L2-normalized
embeddings (`sklearn.preprocessing.normalize(X, norm="l2")`, stateless, split-independent), only
`trainable` species columns (`train_pos >= min_class_support(=10) and train_neg >= 1`) fit, predictions
reindexed to the full class vocabulary with `NaN` for non-trainable species, and the full eligibility
gate (`trainable` / `structurally_unseen` / `test_absent` / `test_single_class` / `evaluable`) and
numerical-trust gate (`converged`, `coef_finite`, `coef_l2_norm <= 50.0`) applied identically.

**Primary metric**: macro-average AP over the single split's `evaluable ∩ numerically_trusted` species
set (this collapses the FINAL plan's `macro_ap_core` / `macro_ap_fold_own_eligible` distinction to one
table, since there is only one split — report `n_species_evaluable` and the exact species list
alongside the scalar, exactly as the FINAL plan already requires). Report per-species AP/AUROC and, in
place of the FINAL plan's cross-fold `mean ± std`, a **bootstrap 95% CI** over the single test split
(window-resampling respecting `sound_id` groups, stratified by species prevalence, ≥1,000-2,000
resamples — this exact mechanism is already specified once, for a different purpose, in
`docs/design/round_01/architect_experiment_proposal.md` line 315, and is the correct way to express
single-split sampling uncertainty here). Go/no-go: macro-AP beats a per-species prevalence-only
baseline by the same pre-registered margin already used elsewhere in this plan (+5 percentage points,
configurable, pre-registered before looking at results).

**Claims this design can support**:
- "Frozen Perch v2 embeddings carry linearly-separable, above-prevalence-baseline species signal on
  PteroSet audio, evaluated **in-distribution** (train/val/test drawn from the same pool of 5
  projects)." — supportable, provided §0's disclosure requirement is satisfied and §2's grouping
  controls are implemented.

**Claims this design cannot support, and must not be worded to imply**:
- Any claim of cross-project / cross-site / cross-equipment generalization — this is precisely what
  the existing LOPO design measures and this pooled/stratified design does not, per Round 2's
  confound analysis (§0, §2). If the existing LOPO folds (or the pending single-fold PPA4 baseline)
  are not also run and reported, no generalization claim beyond "in-distribution" may be made.
- Any claim that "no-bird" and "species-unresolved" windows were both treated as negatives — the
  six-state schema in §2 exists specifically to make this claim false; the results report must state
  the `unresolved_bird` exclusion count and rule explicitly (mirroring the FINAL plan's own disclosure
  conventions for the single-label comparator's retention statistics).
- A single-split point estimate presented as if it had the same statistical weight as a 5-fold
  cross-validated mean — report the bootstrap CI (above) and label this as a single-split baseline,
  not a cross-validated result.

---

## Experiment Integrity Report

### Reproducibility
- Seeds: existing `config.splits.random_state=42` mechanism present and reusable; multilabel-aware
  stratification algorithm not yet selected/pinned (`iterative-stratification` absent from
  `requirements.txt`) — **missing**, must be added and version-pinned before implementation.
- Environment capture: existing FINAL-plan manifest pattern (`pool_manifest.json`) is a directly
  reusable template — **not yet instantiated** for this split.
- Config logging: `config.yaml`'s `splits:` block is the correct place to add stratified-split
  parameters (grouping unit, `min_class_support` used for stratification eligibility, seed) —
  **currently absent**.
- Determinism: fully deterministic *if* seeded and hash-verified per §1/§5; **not yet demonstrated** —
  no split code exists yet to test.

### Data Integrity
- Leakage risk: **HIGH if window-level (ungrouped) stratification is used** — must group by `sound_id`
  and restrict eval splits to non-overlapping windows, exactly as the existing LOPO mechanism already
  does (§2).
- Preprocessing consistency: six-state population definition (`usable_strict`) must be applied
  identically to stratification-input counts, training, and evaluation — **not yet specified in any
  committed document** for this split type (only for the LOPO-based FINAL plan).
- Pipeline correctness: no-bird → all-zero is correct only for genuine `has_bird == 0` windows;
  conflating `unresolved_bird` into the same encoding is a known, previously-diagnosed risk (§2).

### Training Configuration
- Checkpointing: N/A (linear probe, no iterative training loop) — reuse FINAL plan's manifest/
  diagnostics artifacts unchanged.
- Metric computation: reuse FINAL plan's eligibility + numerical-trust gating unchanged (§6).
- Optimizer/scheduler: N/A (`lbfgs` closed-form-adjacent solver, fixed `C=1.0`, no scheduler).

### Pitfalls Detected
- Proposal contradicts the repo's own `STATUS: FINAL` design decision (LOPO, not stratified) without
  disclosure (§0) — process/governance issue, not a code bug, but blocking.
- No multilabel-aware, group-respecting stratification tool is currently in `requirements.txt`.
- Risk of conflating `unresolved_bird` and genuine no-bird windows into the same all-zero encoding.
- Risk of window-level (ungrouped) stratification leaking overlap-sharing windows across splits.
- Val split currently has no pre-registered purpose under the fixed-`C` baseline — risk of an unused,
  undocumented artifact if not explicitly decided.

### Recommendations
- [ ] CRITICAL: Before writing split code, explicitly record whether this stratified split is (a) a
  disclosed SECONDARY, in-distribution-only experiment per rule L6, run alongside the existing/pending
  project-based split, or (b) a deliberate supersession of the FINAL Round-8 LOPO decision, with the
  Round-2 project-confound rationale explicitly re-addressed. (§0)
- [ ] CRITICAL: Implement the split at the `sound_id`-group level (never window level) and restrict
  eval splits to non-overlapping windows, to prevent overlap-leakage across train/val/test. (§2)
- [ ] CRITICAL: Define and apply the six-state schema (`has_bird`, `any_species_known`,
  `unresolved_bird`, `usable_strict`) so that `unresolved_bird` windows are excluded from training,
  evaluation, and the stratification denominator, and only genuine no-bird windows receive an all-zero
  target. (§2)
- [ ] CRITICAL: Pin and log the exact multilabel-aware stratification algorithm/package/version and
  seed; add a regression test asserting identical `window_id` membership across reruns and zero
  `sound_id` overlap across splits. (§1, §5)
- [ ] WARNING: Pre-register val's purpose (future `C`-sweep vs. pure monitoring vs. none) before
  fixing its size; do not leave an unused val split undocumented. (§4)
- [ ] WARNING: Report a bootstrap 95% CI for the single-split macro-AP rather than presenting a bare
  point estimate as if cross-validated. (§6)
- [ ] SUGGESTION: Reuse the FINAL plan's `split_manifest.json` / eligibility-table / numerical-trust
  patterns verbatim rather than inventing a parallel schema for this split type, to keep the two
  experiments (LOPO and stratified) directly comparable in structure even though their headline claims
  must never be blended. (§3, §6)

---

STATUS: BLOCKED
