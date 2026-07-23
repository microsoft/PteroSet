# Round 7 Convergence Review -- architect-minimalist

Reviewing: `docs/design/perch2_species_linear_probe_plan.md` (current revision, 610 lines),
read in full this round. Scope per instructions: search only for genuinely new load-bearing
correctness flaws; no repeated findings, optional enhancements, or style comments.

---

## What changed since Round 6

One substantive addition, in the "Numerical trust" section (lines 272-275): `species_diagnostics_
fold{i}.csv` is now explicitly stated to be "reindexed to the full global class vocabulary.
Trainable, fitted species contain the diagnostics above; non-trainable species contain `NaN` for
fit-derived fields, `numerically_trusted = False`, and their explicit eligibility status. No
diagnostics are fabricated for estimators that were never fit." The Phase 3 acceptance criterion
(line 504-505) was updated to match: "`numerically_trusted` computed for every (fold, trainable
species) fit and full-vocabulary placeholder rows for non-trainable species."

This closes a small residual ambiguity from the Round 6 fix (whether the diagnostics table's row
set matches the fitted-only subset or the full vocabulary) consistently with how predictions
themselves are already reindexed (Round 5 Issue F). It does not introduce a new gap: it applies the
same reindex-with-explicit-placeholder pattern already used for predictions to the diagnostics
table, and explicitly rules out fabricating diagnostics for unfit estimators.

Every other section (Why, What, architecture summary, Identity discipline, Failure handling,
eligibility categories, headline metrics, comparator, cache invalidation, Environment, Files,
Commands, Phased milestones, Tests, Key operational failure modes, Future migration path) was
re-read in full and is textually unchanged from the version reviewed in Round 6.

---

## Search for new issues

No genuinely new load-bearing correctness flaw was found in this pass. All previously-raised
issues (Rounds 3-5) remain resolved and are not repeated here.

---

## Verdict

CONVERGED -- nothing to add

STATUS: DONE
