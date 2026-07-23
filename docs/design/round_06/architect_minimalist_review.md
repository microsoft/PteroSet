# Round 6 Convergence Review -- architect-minimalist

Reviewing: `docs/design/perch2_species_linear_probe_plan.md` (current revision, 608 lines).
Scope per instructions: (1) confirm Issue F (Round 5) is fully resolved on its four required
points; (2) search only for genuinely new load-bearing correctness flaws, not optional
enhancements, repeated issues, or style.

---

## Part 1 -- Verification of Round 5 Issue F

All four required points are confirmed resolved, with explicit text at multiple levels
(architecture summary, dedicated eligibility section, Files, Phase 3 acceptance criteria, and a
new dedicated test):

1. **`OneVsRestClassifier` receives only trainable species columns** -- CONFIRMED. Lines 212-214:
   "The same table is also a pre-fit column-selection gate, not only a reporting filter.
   `train_perch_logreg.py` passes only species with `trainable == True` to
   `OneVsRestClassifier.fit()`." Reaffirmed in Files (line 415: "fits only trainable target
   columns") and Phase 3 acceptance (line 496: "selects only `trainable` species columns").

2. **Constant columns cannot abort the fit** -- CONFIRMED. Line 214-215: "This prevents
   scikit-learn from aborting on constant all-negative or all-positive target columns, which are
   expected in LOPO folds with project-specific species." Since `trainable` requires
   `train_pos >= min_support` and `train_neg >= 1` (both already-established preconditions in the
   eligibility table, lines 201), no column passed to `.fit()` can be constant by construction --
   this is a correct, minimal closure of the crash mechanism verified in Round 5, not a partial
   patch. `test_trainable_column_selection.py` (lines 552-555) directly tests this: "a synthetic
   fold containing trainable, all-negative, and all-positive species columns fits without
   `ValueError`; only trainable columns reach `OneVsRestClassifier.fit()`."

3. **Outputs reindexed to the global class vocabulary with NaN/status for excluded species** --
   CONFIRMED. Line 216-217: "Predictions are then reindexed to the fixed global `class_list.json`
   order; non-trainable species receive `NaN` scores plus their explicit eligibility status, never
   fabricated zero probabilities." Reaffirmed in Phase 3 acceptance (line 498: "non-trainable
   species represented as `NaN` plus status") and in the test (line 554-555: "outputs are
   reindexed to the full class list with `NaN` for excluded columns").

4. **A zero-trainable fold fails clearly** -- CONFIRMED. Line 218: "If a fold has no trainable
   species, that fold fails with a clear error before fitting." This is a named, explicit failure
   mode rather than an implicit crash or a silently empty output.

The fix is minimal, uses only data the eligibility table already computes, requires no new hash,
gate, or artifact, and does not reopen the eligibility taxonomy, the numerical-trust layer, or the
multilabel/multiclass decision. Issue F is closed.

---

## Part 2 -- Search for new issues

Beyond re-verifying Issue F, the rest of the document (Failure handling, comparator section, cache
invalidation, environment, Files, Commands, Phased milestones, Tests, Key operational failure
modes, Future migration path) was re-read in full this round. One candidate was investigated and
ruled out rather than raised, noted here for completeness of the review record (not a new open
issue):

- `test_l2_normalization.py`'s note "(except an all-zero vector, which is rejected upstream)"
  (line 549) could suggest a pipeline mechanism is needed to keep a degenerate all-zero embedding
  row from reaching `sklearn.preprocessing.normalize`. Checked against `sklearn`'s actual behavior:
  `sklearn.preprocessing.normalize`/`Normalizer` with `norm="l2"` is zero-row-safe by design -- a
  row with zero L2 norm is left unchanged (still all zeros) rather than producing `NaN` or raising,
  because `sklearn` guards the division internally. There is therefore no division-by-zero or
  silent-NaN-propagation risk from the L2-normalization step itself regardless of whether a
  degenerate embedding row ever occurs, and no missing pipeline mechanism to flag. This is not a
  new load-bearing issue.

No other genuinely new load-bearing correctness flaw was found. All previously-open items from
Rounds 3-5 remain resolved and are not repeated here.

---

## Verdict

CONVERGED -- nothing to add

STATUS: DONE
