# Round 5 -- Clean Convergence Review

**Author role**: Chief Architect (architect-pipeline), Round 5
**Scope**: read only `docs/design/perch2_species_linear_probe_plan.md` as currently stored (592
lines). Specifically verify the Round 4 finding was fully resolved (`build_fold_embeddings.py` sole
producer of `species_eligibility_fold{i}.csv`, Phase 2 owns the acceptance criterion,
`train_perch_logreg.py` consumes it), that no new ownership contradiction was introduced by the
latest edits, and search for genuinely new load-bearing implementation flaws only.

---

## 1. Verification of the Round 4 fix: partially applied, one instance missed

The requested fix was applied correctly in four of five locations:

- **Architecture diagram** (`## How`, line 65): the `build_fold_embeddings.py` arrow's output list
  reads "per-species-per-fold eligibility table" among that step's outputs. Correct.
- **Explicit ownership sentence** (line 175-177, immediately following the `fold_manifest.json`
  schema): *"`build_fold_embeddings.py` is the sole producer of `species_eligibility_fold{i}.csv`,
  because it already owns the post-exclusion train/test labels and support counts.
  `train_perch_logreg.py` consumes this table; it does not recompute or overwrite it."* This is a
  clean, unambiguous statement of exactly the resolution requested. Correct.
- **Files section** (line 397): `build_fold_embeddings.py`'s bullet now reads "...computes both hash
  lineages; sole producer of the per-species-per-fold eligibility table after exclusions." Correct.
- **Phase milestones** (lines 483, 485): Phase 2's acceptance criteria now include
  "`species_eligibility_fold{i}.csv` is produced for all 5 folds after exclusions," and Phase 3 opens
  with "`train_perch_logreg.py` consumes the Phase 2 eligibility tables." Correct -- this is exactly
  the requested move of the acceptance criterion from Phase 3 to Phase 2.

**One location was not updated and still contains the original, pre-fix sentence.** In the dedicated
"Per-species-per-fold eligibility" section itself, immediately after the five-category table (line
206-209):

> "This table (`species_eligibility_fold{i}.csv`) is a required artifact per fold, **produced by
> `train_perch_logreg.py`** before any metric is computed, and every downstream metrics table is
> filtered through it explicitly..."

This directly contradicts the sentence 31 lines above it in the same document (line 175-177, quoted
above), which states the opposite: `build_fold_embeddings.py` is the sole producer and
`train_perch_logreg.py` only consumes. It also contradicts the Files section and the Phase 2/3
milestones, all four of which now agree with each other but not with this one remaining sentence.

This is a genuinely new finding for this round in the sense that it is a different textual
manifestation than what Round 4 reported (Round 4 found a 2-vs-2 split across four locations; this
round finds that the fix converged three of those four locations plus added a new explicit ownership
statement, but left the original sentence standing in the fifth, most-topical location -- the section
whose entire subject is this artifact). It is load-bearing for the same reason as before: an
implementer reading only the "Per-species-per-fold eligibility" section in isolation (a plausible
reading path, since it is the section most directly about this artifact) would still conclude
`train_perch_logreg.py` produces the table, contradicting the rest of the document.

**Recommended fix** (one sentence, no other change needed): in line 206-209, replace "produced by
`train_perch_logreg.py` before any metric is computed" with "produced by `build_fold_embeddings.py`
after exclusions (see the ownership statement above); `train_perch_logreg.py` consumes it before any
metric is computed." This makes the sentence consistent with line 175-177 and the rest of the
document without changing its surrounding meaning (the "before any metric is computed" framing and
the "every downstream metrics table is filtered through it explicitly" clause are still accurate and
should be kept).

## 2. No other new ownership contradiction found

Checked all other producer/consumer assignments for internal consistency across every section that
mentions them (diagram, Files section, Failure handling, Cache invalidation, Numerical trust,
Comparator, Commands, Phase milestones): hash-lineage ownership (`build_fold_embeddings.py`,
consistent everywhere), diagnostics/comparator-retention ownership (`train_perch_logreg.py`,
consistent everywhere), `pool_manifest.json` ownership (`extract_embeddings_pteroset.py`, consistent
everywhere), and `fold_manifest.json` ownership (`build_fold_embeddings.py`, consistent everywhere).
No other artifact has a second, conflicting attribution anywhere in the document.

## 3. Nothing else load-bearing found

Re-read the full document end to end this round (Why, What, How, Identity discipline, Failure
handling, Per-species-per-fold eligibility, Headline metrics, Numerical trust, Optional comparator,
Cache invalidation, Environment, Files, Commands, Phased milestones, Tests, Key operational failure
modes, Future migration path). No other new internal contradiction, stale reference, or gap was
found. Not re-raised here, per this round's scope: the pre-existing, previously-considered-and-set-
aside illustrative-example detail in the `pool_manifest.json` sample (`counts.excluded: 4` alongside
a default `failure_ceiling` of `0`) -- unchanged from Round 4, already judged non-load-bearing
(example-only, does not affect any rule, gate, or schema field), and not repeated as a finding here.

---

## Verdict

**NEEDS-MORE.**

The Round 4 fix (`build_fold_embeddings.py` as sole producer of `species_eligibility_fold{i}.csv`)
was applied in four of five locations but missed the fifth: the dedicated "Per-species-per-fold
eligibility" section's own descriptive sentence (line 206-209) still attributes production to
`train_perch_logreg.py`, contradicting the rest of the now-corrected document. This is a small,
one-sentence textual fix (given verbatim in section 1 above), not a new mechanism or design change,
and no other new load-bearing issue was found on this pass.

STATUS: DONE
