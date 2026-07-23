# Round 7 -- Clean Convergence Review

**Author role**: Chief Architect (architect-pipeline), Round 7
**Scope**: read `docs/design/perch2_species_linear_probe_plan.md` in full (614 lines) as currently
stored. Specifically verify: (1) diagnostics are computed only for fitted trainable species; (2) the
diagnostics CSV is reindexed to the full vocabulary with `NaN`/status placeholders for non-trainable
species; (3) Phase 3 wording matches the mechanism. Then search only for genuinely new load-bearing
implementation flaws -- no repeats, optional enhancements, or style comments.

---

## 1. Round 6 findings: both fully resolved

- **Diagnostics scoped to fitted species, and Phase 3 wording matches**: the "Numerical trust"
  section (lines 249-250) now reads "For every fitted trainable-species `LogisticRegression`
  estimator inside `OneVsRestClassifier`, record..." (previously "every one of the K per-species
  fits," which was consistent but the overclaiming Phase 3 sentence was not). Phase 3's acceptance
  criterion (lines 504-505) now reads "`numerically_trusted` computed for every (fold, trainable
  species) fit and full-vocabulary placeholder rows for non-trainable species" -- this matches the
  mechanism exactly and no longer overclaims. Resolved.
- **Diagnostics CSV reindexed to the full vocabulary, explicitly**: a new paragraph (lines 272-275)
  states "`species_diagnostics_fold{i}.csv` is reindexed to the full global class vocabulary.
  Trainable, fitted species contain the diagnostics above; non-trainable species contain `NaN` for
  fit-derived fields, `numerically_trusted = False`, and their explicit eligibility status. No
  diagnostics are fabricated for estimators that were never fit." This closes the artifact-schema gap
  from Round 6 -- the document now states its convention explicitly rather than leaving it implicit,
  and picked schema (b) (full-vocabulary reindex with placeholders) consistently, matching the other
  two per-species tables.

Both fixes are internally consistent with the rest of the document: the "Rule" immediately preceding
(lines 265-270) already excludes any `numerically_trusted == False` pair from headline aggregates
"exactly as if it were not `evaluable`," and non-trainable species are already excluded from
`evaluable` independently (since `evaluable` requires `trainable`), so setting
`numerically_trusted = False` for them does not change any headline-metric computation -- it is
harmless there.

## 2. New finding: the full-vocabulary reindex reuses `numerically_trusted = False` for two
different reasons, undermining the diagnostics table's own stated purpose (load-bearing)

The reindexing fix that resolved Round 6's gap introduces a new, genuine ambiguity specific to
`species_diagnostics_fold{i}.csv`. The "Numerical trust" section states this table's entire purpose,
twice, is to keep two failure reasons visibly distinct:

> "...so nothing is ever silently dropped without a visible trace distinguishing 'not enough data'
> (the eligibility table) from 'fit was numerically untrustworthy despite having data' (this
> diagnostics table)." (lines 268-270)

But the very next paragraph (lines 272-275), added to fix Round 6's gap, assigns
`numerically_trusted = False` to **both**:
- a species that was never fit at all because it lacked training data (`trainable == False` --
  "not enough data"), and
- a species that was fit but failed the convergence/finiteness/coefficient-norm gate
  (`trainable == True` but `converged`/`coef_finite`/`coef_l2_norm` failed -- "numerically
  untrustworthy despite having data").

Both now read `numerically_trusted == False` in the same column, in the same CSV. Any consumer that
builds the document's own required "excluded for numerical reasons" appendix table (line 268, in the
results report) by filtering `species_diagnostics_fold{i}.csv` on `numerically_trusted == False` --
the natural, most direct reading of a column named exactly that -- would silently conflate the two
categories the document says must stay distinguishable, which is precisely the failure mode this
table exists to prevent.

The document's only stated mitigation is the vague phrase "their explicit eligibility status"
(lines 217 and 274), which is never tied to a concrete column name anywhere in the document. The
eligibility table's own schema (section "Per-species-per-fold eligibility") defines five specific
boolean columns (`trainable`, `structurally_unseen`, `test_absent`, `test_single_class`,
`evaluable`) -- but nothing states that any of these five columns is actually copied into
`species_diagnostics_fold{i}.csv` rows, under what name, so that a consumer could reliably filter on
it. "Their explicit eligibility status" reads as a description of intent, not a schema commitment.

This is load-bearing because it directly undermines a requirement stated elsewhere in the same
document (the appendix must distinguish the two reasons) using a mechanism (the reindex) that was
added specifically to close a different, previously-identified gap -- a new instance of the same
class of problem this multi-round review has repeatedly found: a correct, targeted fix that was not
fully propagated to every place its side effects reach.

**Recommended fix** (schema addition, one sentence, no mechanism change): state explicitly that
`species_diagnostics_fold{i}.csv` carries a `trainable` column (copied verbatim from
`species_eligibility_fold{i}.csv` for the same fold/species), and that the "excluded for numerical
reasons" appendix in the Phase 4 results report must be built by filtering
`trainable == True and numerically_trusted == False`, never by filtering on
`numerically_trusted == False` alone. This requires no change to any threshold, gate, or fitting
mechanism -- only naming the already-implied column and stating the one filter rule explicitly.

## 3. Nothing else load-bearing found

Re-read the full document end to end. Checked, and found consistent: the Files section, Commands
section, and Tests section against the now-corrected Numerical trust and Phase 3 text (all agree);
`test_trainable_column_selection.py` and `test_numerical_trust_flagging.py` remain accurate
descriptions of the mechanism as now specified; the "Key operational failure modes" section's
`structurally_unseen`/numerical-trust bullets (lines 589-597) do not themselves make the
column-collision error (they describe the exclusion rule at the right level of abstraction and do
not claim the appendix-table filtering mechanism). No other new ownership, schema, or wording
contradiction was found. Not re-raised here, per this round's scope: the previously-considered
illustrative `pool_manifest.json` example detail, unchanged and still judged non-load-bearing.

---

## Verdict

**NEEDS-MORE.**

Both Round 6 findings are fully and correctly resolved. One new, load-bearing issue was found: the
fix that resolved Round 6's diagnostics-reindexing gap causes `species_diagnostics_fold{i}.csv` to
assign the same `numerically_trusted == False` value to two reasons the document elsewhere requires
to be kept visibly distinct ("not enough data" vs. "numerically untrustworthy despite having data"),
and the document's only mitigating language ("their explicit eligibility status") never commits to a
concrete column name a consumer could filter on. The fix is a one-sentence schema addition (name the
`trainable` column explicitly, state the required two-condition filter for the appendix table), not a
mechanism or design change, and does not reopen any other decision in the document.

STATUS: DONE
