# Round 6 -- Clean Convergence Review

**Author role**: Chief Architect (architect-pipeline), Round 6
**Scope**: read `docs/design/perch2_species_linear_probe_plan.md` in full (608 lines) as currently
stored. Specifically verify: (1) script ownership of `species_eligibility_fold{i}.csv` is now fully
consistent; (2) Phase acceptance criteria; (3) trainable-column subsetting before
`OneVsRestClassifier.fit()`; (4) artifact schemas; (5) global-vocabulary output reindexing. Search
only for genuinely new load-bearing implementation flaws, no optional enhancements, no repeats.

---

## 1. Round 5 finding: now fully resolved

The leftover sentence flagged in Round 5 (the "Per-species-per-fold eligibility" section attributing
production of `species_eligibility_fold{i}.csv` to `train_perch_logreg.py`) has been corrected. Line
207-209 now reads: "...produced by `build_fold_embeddings.py` after exclusions and consumed by
`train_perch_logreg.py` before any metric is computed." This is consistent with every other
mention of this artifact's ownership in the document (diagram line 65-66, explicit ownership
sentence lines 176-178, Files section lines 406-407, Phase 2/3 milestones lines 493/495). All five
locations now agree. No remaining trace of the prior contradiction.

## 2. New mechanism since Round 5: trainable-column subsetting and reindexing -- specified, mostly
consistent

The document now contains a new paragraph (lines 212-218, under "Per-species-per-fold eligibility")
establishing that `species_eligibility_fold{i}.csv`'s `trainable` flag doubles as a pre-fit
column-selection gate: `train_perch_logreg.py` passes only `trainable == True` species columns into
`OneVsRestClassifier.fit()`, explicitly to avoid the constant-column failure mode that a
`structurally_unseen` or otherwise all-one-class species column would otherwise risk, and predictions
are reindexed back to the fixed `class_list.json` order afterward with `NaN` for non-trainable
species. This is a real, correctly-targeted fix, and it is consistently reflected in the Files
section (lines 415-416), the Commands section (unchanged, consistent), the Phase 3 acceptance
criteria (lines 495-498), and a new dedicated test (`test_trainable_column_selection.py`, lines
552-555).

## 3. New finding: Phase 3 acceptance criterion overclaims `species_diagnostics_fold{i}.csv`'s
scope (load-bearing)

Line 499-500 states, as a Phase 3 acceptance criterion: "`species_diagnostics_fold{i}.csv` produced
with `numerically_trusted` computed **for every (fold, species) pair**." Read literally, "every
(fold, species) pair" means every species in the global `class_list.json`, for every fold.

This directly conflicts with the mechanism introduced elsewhere in the same document:

- The "Numerical trust" section (lines 248-250) scopes the diagnostics table's population correctly:
  "For every one of the **K per-species** `LogisticRegression` fits inside `OneVsRestClassifier`,
  record... into `species_diagnostics_fold{i}.csv`" -- i.e., only over the species that were actually
  fit.
- The newly added column-selection-gate paragraph (lines 212-215) makes explicit that only
  `trainable == True` species are ever passed to `.fit()`. A `structurally_unseen` or otherwise
  non-trainable species is never fit at all in that fold, so it has no `n_iter_`, no `coef_`, and no
  `converged` value to compute `numerically_trusted` from -- there is no fitted estimator to inspect.

"Computed for every (fold, species) pair" is therefore not achievable as literally stated for
non-trainable species in a given fold; the true, mechanism-consistent scope is "every (fold, species)
pair where that species is `trainable` in that fold" (equivalently, every row that actually reached
`.fit()`). This is load-bearing because it is an **acceptance criterion** -- the literal text gives an
implementer/reviewer an unfalsifiable or wrongly-scoped bar to check Phase 3 against: either they
attempt to force a `numerically_trusted` value for species that were never fit (impossible without
contradicting the very column-selection gate added to prevent constant-column crashes), or they
silently narrow the acceptance check without the document ever having said so, which is exactly the
kind of silent scope-narrowing this document elsewhere explicitly forbids (e.g. the eligibility and
numerical-trust sections both stress that nothing should be "silently dropped without a visible
trace").

**Recommended fix** (one clause, no mechanism change): reword line 499-500 to "`species_diagnostics_fold{i}.csv`
produced with `numerically_trusted` computed for every (fold, species) pair **that was actually
fit (i.e. every `trainable` pair)**." This matches the "Numerical trust" section's existing "K
per-species fits" framing and requires no other change.

## 4. New finding: reindexing rule is specified for predictions but not for
`species_diagnostics_fold{i}.csv`'s row population (artifact-schema gap, load-bearing)

The document explicitly specifies the global-vocabulary reindexing rule for one artifact only:
"Predictions are then reindexed to the fixed global `class_list.json` order; non-trainable species
receive `NaN` scores plus their explicit eligibility status" (lines 216-217) -- this covers
`fulldata_results_species_fold{i}.csv`, which the architecture diagram already independently
describes as "eligibility-masked" (line 75), so its full-global-vocabulary row population with
masking for non-evaluable species is doubly confirmed.

No equivalent statement exists for `species_diagnostics_fold{i}.csv`. Its row population is left
genuinely ambiguous between two materially different, both-plausible schemas:

- **(a)** One row per (fold, trainable species) pair only -- i.e., exactly the K species that were
  fit, no rows at all for non-trainable species. This matches the "K per-species fits" framing in the
  Numerical trust section.
- **(b)** One row per (fold, global species) pair, with `NaN`/absent values for `n_iter_`, `coef_l2_norm`,
  etc. on non-trainable rows -- mirroring the reindexing rule applied to predictions, for uniformity
  with the other two per-species CSVs (`fulldata_results_species_fold{i}.csv`,
  `species_eligibility_fold{i}.csv`, both of which do cover the full global vocabulary).

This is a genuine, newly-introduced artifact-schema gap (it did not exist before the trainable-column-
selection mechanism was added, since previously the document's now-superseded assumption was that all
species were fit). It is load-bearing because a downstream consumer (the Phase 4 results report, or
any script joining diagnostics against `class_list.json` by species name) needs to know, without
guessing, whether a missing species in `species_diagnostics_fold{i}.csv` means "not trainable, by
design" (schema (a)) or "a bug -- every species should have a row" (schema (b)). The two other
per-species CSVs in this same pipeline already made the opposite choice from each other for a
comparable reason (`species_eligibility_fold{i}.csv` covers the full vocabulary by definition, since
eligibility itself must be defined for every species including non-trainable ones; the reindexed
predictions table also covers the full vocabulary with explicit `NaN`), which makes it more, not less,
important that this third table's convention is stated rather than left implicit.

**Recommended fix** (one sentence, no mechanism change): add, immediately after the existing
reindexing sentence (end of line 217): "`species_diagnostics_fold{i}.csv` contains rows only for the
`trainable` (fold, species) pairs that were actually fit; it is not reindexed to the full global
vocabulary, unlike the predictions and eligibility tables." (Or the opposite convention, if schema (b)
is preferred -- either is acceptable, but the document must pick one explicitly.)

## 5. Nothing else load-bearing found

Re-read the full document end to end. Checked, and found consistent: all other artifact ownership
assignments (hash lineages, `pool_manifest.json`, `fold_manifest.json`); the Commands section against
the Files section (flag names and script responsibilities agree); the Tests section against the
mechanisms it tests (test #10 correctly targets the new trainable-column-selection gate); the "Key
operational failure modes" section (its `structurally_unseen`/`test_absent`/`test_single_class`
bullet, lines 583-587, is written at the `evaluable`-filtering level of abstraction and does not make
the same "every (fold, species) pair" overclaim that Phase 3's diagnostics bullet does -- no
contradiction there). Not re-raised here, per this round's scope: the previously-considered-and-set-
aside illustrative `pool_manifest.json` example detail (`counts.excluded: 4` alongside a default
`failure_ceiling` of `0`), unchanged and still judged non-load-bearing.

---

## Verdict

**NEEDS-MORE.**

The Round 5 finding (eligibility-table ownership) is now fully and consistently resolved -- no issue
remains there. Two new, load-bearing issues were found this round, both arising from the newly added
trainable-column-selection/reindexing mechanism, which is the right fix but was not propagated
completely to every place it touches:

1. Phase 3's acceptance criterion overclaims that `numerically_trusted` is "computed for every (fold,
   species) pair," when the column-selection gate means it can only be computed for `trainable`
   pairs that were actually fit. One-clause fix given above.
2. `species_diagnostics_fold{i}.csv`'s row population relative to the global species vocabulary is
   unspecified (unlike the predictions table and the eligibility table, both of which explicitly
   state their vocabulary coverage). One-sentence fix given above.

Both are small textual corrections, not mechanism or design changes, and do not reopen any other
decision in the document.

STATUS: DONE
