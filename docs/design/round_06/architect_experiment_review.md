# Round 6 — Clean Convergence Review

**Author**: architect-experiment (Round 6 review)
**Input read**: `docs/design/perch2_species_linear_probe_plan.md`, full current revision, read end to end this round (all sections, not just the diffs since Round 5).

**Task**: verify that the primary multilabel training, eligibility, metrics, numerical trust, comparator isolation, and the newly-added fixed/global class reindexing mechanism together form a scientifically coherent baseline; search only for genuinely new load-bearing flaws.

---

## 1. What changed since Round 5, and whether it is coherent

Two things changed since the Round 5 review:

**(a) The Round 5 ownership contradiction is now fully resolved**, consistently, in every location that mentions it:
- Architecture diagram (line 69): `train_perch_logreg.py` "consume[s] `species_eligibility_fold{i}.csv`."
- "Per-species-per-fold eligibility" section (lines 207-210): "produced by `build_fold_embeddings.py` after exclusions and consumed by `train_perch_logreg.py` before any metric is computed."
- "Failure handling" section (lines 176-178): "`build_fold_embeddings.py` is the sole producer of `species_eligibility_fold{i}.csv`, because it already owns the post-exclusion train/test labels and support counts. `train_perch_logreg.py` consumes this table; it does not recompute or overwrite it."
- Files section and Phased milestones (Phase 2 vs. Phase 3) agree with this. No remaining contradiction anywhere in the document.

**(b) A new mechanism was added: the eligibility table is now also a pre-fit column-selection gate**, not only a reporting filter (lines 212-218). `train_perch_logreg.py` fits `OneVsRestClassifier` only on `trainable == True` species columns, then reindexes the resulting predictions back into the fixed global `class_list.json` order, assigning `NaN` (plus explicit eligibility status) to non-trainable species rather than a fabricated probability. This is checked for internal coherence against every other mechanism in the document:

- **Coherent with the `trainable` definition itself**: `trainable` already requires `train_pos >= min_support` *and* `train_neg >= 1`, which means an all-negative (`train_pos < min_support`, typically `0` for a structurally-unseen species) or all-positive (`train_neg == 0`) column can never reach `OneVsRestClassifier.fit()` under this gate — exactly the two constant-column cases that would otherwise make `sklearn` error or emit degenerate per-class output. The column-selection gate does not need any additional condition beyond the existing `trainable` flag to guarantee no constant column is ever fit.
- **Coherent with `evaluable`**: `evaluable` is a strict subset of `trainable` (`evaluable = trainable AND NOT test_absent AND NOT test_single_class`), so every species that ever contributes to `macro_ap_core` or `macro_ap_fold_own_eligible` necessarily has a real (non-`NaN`) fitted score, never a reindexed placeholder — the headline metrics can never accidentally aggregate over a `NaN`.
- **Coherent with numerical trust**: `species_diagnostics_fold{i}.csv` and `numerically_trusted` are only meaningful for species that were actually fit (the `trainable` set), which is exactly the set the diagnostics table is scoped to; a non-trainable species has no estimator and correctly has no diagnostics row to fabricate.
- **Coherent with the comparator's isolation**: the column-selection/reindexing change is scoped to the primary `OneVsRestClassifier`'s per-class target matrix only; the comparator's own `comparator_eligibility_fold{i}.csv`/`comparator_diagnostics_fold{i}.csv` gates (Round 4's fix, confirmed intact in Round 5) are a separate mechanism over the comparator's own retained population and are untouched by this change.
- **Covered by a dedicated test**: `test_trainable_column_selection.py` (new, #10) directly exercises the constant-all-negative/all-positive-column case and asserts both the no-`ValueError`-on-fit property and the reindex-to-`NaN` behavior — the mechanism is not just described but has an explicit acceptance test, and the Phase 3 acceptance criterion text was updated to match ("trains ... without constant-column errors; predictions are reindexed to the global class list with non-trainable species represented as `NaN` plus status").

No incoherence, gap, or unhandled edge case was found in how this new mechanism interacts with eligibility, headline metrics, numerical trust, or comparator isolation. It closes a real, previously-implicit sklearn-level failure mode (constant target columns are certain to occur in LOPO folds with project-specific species) without introducing a new one.

---

## 2. Search for genuinely new load-bearing flaws

Re-read every remaining section (Identity discipline, Cache invalidation, Environment, Files, Commands, Tests, Key operational failure modes, Future migration path) in full against the rest of the document for the first time this round, not only the sections that changed. No new contradiction, gap, or scientifically load-bearing issue was found beyond what is already covered above (which is a confirmation, not a new finding).

---

## 3. Verdict

**CONVERGED -- nothing to add.**

STATUS: DONE
