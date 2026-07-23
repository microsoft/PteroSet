# Round 8 Convergence Review -- architect-minimalist

Reviewing: `docs/design/perch2_species_linear_probe_plan.md` (current revision, 623 lines),
read in full this round, with particular attention to the corrected pool-manifest hash schema and
the diagnostics exclusion-reason schema called out in this round's instructions. Scope: search only
for genuinely new load-bearing correctness flaws; no repeats, optional enhancements, or style
comments.

---

## What changed since Round 7

1. **Pool-manifest hash schema, `identity_hash_inputs` (lines 118-125)**: the JSON example now
   explicitly lists `pool_csv_sha256`, `pool_builder_source_sha256`, and `pool_builder_config` as
   separate, named fields, alongside `windows_mapping_json_sha256`, `model_local_dir_sha256`, and
   `extraction_params`. Previously (Round 6/7) these inputs were described only in prose in the
   Cache Invalidation section; the manifest's own JSON example had not been updated to match. This
   is a schema-example/prose consistency fix, not a new mechanism -- the prose in "Cache
   invalidation" (lines 326-332) already required these inputs; the manifest schema now visibly
   matches what it claims to hash.

2. **Diagnostics exclusion-reason schema, `species_diagnostics_fold{i}.csv` (lines 275-282)**: three
   explicit fields are now required -- `trainable`, `fit_attempted` (`== trainable`), and
   `numerical_exclusion` (`== fit_attempted and not numerically_trusted`). The "excluded for
   numerical reasons" appendix is now required to filter on `numerical_exclusion == True`
   specifically, and is explicitly forbidden from filtering on `numerically_trusted == False` alone,
   "which would conflate unfit species with failed fits." This closes a real ambiguity: under the
   Round 6 placeholder convention, a non-trainable (never-fit) species also carries
   `numerically_trusted = False` as its placeholder value, so a naive appendix filter on
   `numerically_trusted == False` would have silently mixed two different exclusion reasons
   (no-data-support vs. failed-numerical-fit) into one table. The three-field formula correctly
   disambiguates all three relevant states:
   - non-trainable (`fit_attempted=False`) -> `numerical_exclusion=False` (correctly excluded from
     the numerical-reasons appendix; its reason lives in the eligibility table instead).
   - trainable, fit succeeds and trusted (`fit_attempted=True`, `numerically_trusted=True`) ->
     `numerical_exclusion=False` (correctly not flagged).
   - trainable, fit attempted but untrusted (`fit_attempted=True`, `numerically_trusted=False`) ->
     `numerical_exclusion=True` (correctly flagged).
   A new test, `test_diagnostics_exclusion_reasons.py` (lines 569-571), directly asserts this
   three-way disambiguation. This is a genuine, correctly-executed fix with no gap re-introduced.

All other sections (Why, What, architecture summary, Identity discipline, remainder of Failure
handling, eligibility categories, headline metrics, comparator, remainder of Cache invalidation,
Environment, Files, Commands, Phased milestones, remainder of Tests, Key operational failure modes,
Future migration path) were re-read in full and are textually unchanged from Round 7.

---

## Search for new issues

No genuinely new load-bearing correctness flaw was found. Both corrections in this revision close
real gaps cleanly without introducing new ones, and are consistent with every other already-
converged mechanism in the document (pre-fit trainable-column selection, prediction reindexing,
the two independent cache-invalidation hash lineages, and the eligibility/numerical-trust
separation). All previously-raised issues (Rounds 3-6) remain resolved and are not repeated here.

---

## Verdict

CONVERGED -- nothing to add

STATUS: DONE
