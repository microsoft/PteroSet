# Round 5 Convergence Review -- architect-minimalist

Reviewing: `docs/design/perch2_species_linear_probe_plan.md` (current revision, 592 lines).
Scope per instructions: (1) verify the four Round 4 issues and the license item are resolved,
without re-litigating them if they are; (2) search only for genuinely NEW load-bearing
correctness flaws, not optional enhancements or style.

---

## Part 1 -- Verification of Round 4 findings

All four Round 4 issues and the license note are resolved. Evidence, quoted from the current
document:

1. **`macro_ap_core` minimum size + reporting** (Round 4 issue #1) -- RESOLVED.
   Lines 219-223: "The file must report `n_species_core` and the exact species list. Phase 0
   pre-registers `min_core_species` (recommended initial value: 5); if the final intersection is
   smaller, no headline scalar or go/no-go claim is produced -- only the per-fold tables are
   reported." Reinforced in Phase 3 acceptance criteria (line 491): "If `n_species_core <
   min_core_species`, no headline scalar is produced." Both the floor and the mandatory
   denominator disclosure are now explicit and enforced at the acceptance-criteria level, not just
   described in prose.

2. **`pool_embedding_hash` must cover `pool.csv` and its generator** (Round 4 issue #2) --
   RESOLVED. Lines 305-309: "a function of `windows_mapping_...json`'s SHA-256, the generated
   `pool.csv` content SHA-256, the `build_embedding_pool_csv.py` source hash and relevant config
   values, the vendored model directory's content hash, and the extraction params." This closes
   the asymmetry with `label_recipe_hash` that Round 4 flagged. `test_cache_hash_independence.py`
   (line 532) explicitly requires a test asserting a `pool.csv` or `build_embedding_pool_csv.py`
   change moves `pool_embedding_hash`.

3. **L2-normalization must be an explicit, mandatory step, not implied** (Round 4 issue #3) --
   RESOLVED, and now stated in three independent places: the architecture summary (line 68-70,
   not re-quoted here since already read in a prior turn), the numerical-trust section (lines
   263-265: "Before fitting or predicting, apply `sklearn.preprocessing.normalize(X, norm='l2')`
   independently to each split's embedding rows... No `StandardScaler` is fitted or persisted."),
   and a dedicated test (`test_l2_normalization.py`, line 536). This is stronger than the Round 4
   request, which only asked for one unambiguous statement.

4. **Environment fallback must distinguish imports-fail / no-GPU / forward-pass-fails** (Round 4
   issue #4) -- RESOLVED. Lines 361-368 give three numbered causes with three distinct remedies:
   import failure -> pinned fallback env; no GPU visible -> "an environment rebuild cannot fix
   missing hardware... move extraction to a GPU host... A CPU Perch export is a separate future
   design choice, not an automatic fallback in this baseline" (i.e., this is now honestly stated as
   an unresolved hard blocker rather than glossed over); forward pass fails on a visible GPU ->
   diagnose CUDA/SavedModel/TF compatibility, extraction stays blocked. This is exactly the
   three-way distinction requested, including an honest non-remedy for the hardware-absent case
   rather than a false promise of a fix.

5. **License verification, previously silently dropped** (Round 4 non-blocking note) -- RESTORED.
   Lines 373-375: "Before model vendoring, verify and record the Perch v2 weights license, the
   source-code license, the resolved Kaggle model version, and any redistribution constraints in
   `pool_manifest.json` and the results report. Model download and publication are blocked if the
   license cannot be verified." Also present in the Phase 0 acceptance criterion (line 470):
   "Perch v2 weights/code licenses and the resolved Kaggle model version are recorded."

No re-litigation needed on any of the five; all are closed cleanly, in some cases more completely
than requested.

---

## Part 2 -- New issues found this round

### Issue F (new, load-bearing): `OneVsRestClassifier.fit()` will raise and abort the entire
fold's training step the moment any class column is single-class in that fold's train split --
which the plan's own eligibility design guarantees will happen on essentially every LOPO fold

**The mechanism.** `train_perch_logreg.py` is specified (Files section, lines 399-401) to
construct `OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0))` and fit
it once per fold. `OneVsRestClassifier.fit(X, Y)` loops over every column of `Y` (one column per
species in `class_list.json`) and calls the inner `LogisticRegression.fit()` on each column
independently. `LogisticRegression.fit()` raises `ValueError: This solver needs samples of at
least 2 classes in the data, but the data contains only one class: ...` if a column is
constant -- i.e. if a species has either `train_pos == 0` (all-negative column) or
`train_neg == 0` (all-positive column) in that fold's train split.
`OneVsRestClassifier` has no built-in mechanism to catch a single failing column and continue with
the rest: the exception propagates out of the single `.fit()` call and aborts training for **every
other species in that fold**, not just the offending one. (Verified against scikit-learn's
`multiclass.py` fitting behavior and the well-documented `ValueError` message; this is standard,
widely-reported sklearn behavior, not a PteroSet-specific guess.)

**Why this is guaranteed to trigger, not a rare edge case.** The plan itself defines
`structurally_unseen` (`train_pos == 0`) as an expected, normal per-fold outcome precisely because
`class_list.json` is built once, globally, across all five projects (Commands section, Phase 1),
while each LOPO fold trains on only four of the five projects. Any species whose only annotated
occurrences are in the held-out project will have `train_pos == 0` in that fold -- and the whole
point of the `species_eligibility_fold{i}.csv` / `structurally_unseen` category (lines 187-210) is
to name and report this as a routine, anticipated event, not a rare failure. That is exactly the
condition that makes the raw `LogisticRegression` column crash. As written, the pipeline computes
the eligibility table (correctly) and then is not documented to act on it before calling `.fit()`
-- the eligibility table as specified is a **post-hoc reporting/filtering artifact**, computed "before
any metric is computed" (line 207), with no stated step that also **subsets which class columns are
passed into the classifier constructor call itself**. Given five real, heterogeneous projects, at
least one `structurally_unseen` species in at least one fold is close to certain, so the plan as
literally written will not complete Phase 3 on real data at all -- this is a correctness gap in the
critical path, not a cosmetic one.

**What must change (minimal, no new machinery).** `train_perch_logreg.py` must restrict the
species columns actually passed to `OneVsRestClassifier.fit()` (or, more simply, loop over
`LogisticRegression` fits manually per species instead of using the `OneVsRestClassifier`
convenience wrapper, which several projects do anyway once per-class exclusion is needed) to only
those species with **both classes present in that fold's train split**
(`train_pos >= 1 and train_neg >= 1`) -- a strictly weaker, fit-time-only precondition than the
already-defined `trainable` category (which additionally requires `train_pos >= min_support`).
Species excluded from the fit call entirely (not `trainable` by this weaker criterion) must still
appear in `species_eligibility_fold{i}.csv` as `structurally_unseen` (or a new, equally-explicit
`train_single_class` category if the all-positive case is distinguished from the all-negative
case), with no attempted prediction, exactly mirroring how `test_absent`/`test_single_class`
species are already excluded from evaluation without being silently scored. No new hash, gate,
artifact, or config surface is required -- this is a one-line subsetting step (`class_list_this_fold
= [c for c in class_list if train_pos[c] >= 1 and train_neg[c] >= 1]`) applied before the existing
`OneVsRestClassifier` call, using columns the eligibility computation (lines 189-193) already
derives. This keeps the fix inside the plan's existing minimalist machinery; it does not reopen the
multilabel-vs-multiclass decision, the eligibility taxonomy, or the numerical-trust layer, all of
which remain correct and unchanged once this one omission is patched.

**Note on scope**: this is distinct from, and does not reopen, the already-converged
`numerically_trusted` gate (lines 238-265), which handles a *different* failure mode (a column with
both classes present but a degenerate/non-converged/unbounded fit). Issue F is about a column that
cannot be fit at all, and would currently crash the process before any numerical-trust diagnostic
is even reached.

---

## Verdict

**NEEDS-MORE** -- one genuinely new, load-bearing issue (F). All five previously open items
(Round 4 #1-4 and the license note) are confirmed resolved and are not repeated here.

STATUS: DONE
