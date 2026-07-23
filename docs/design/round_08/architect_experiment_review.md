# Round 8 — Clean Convergence Review

**Author**: architect-experiment (Round 8 review)
**Input read**: `docs/design/perch2_species_linear_probe_plan.md`, full current revision, read end to end this round.

**Task**: confirm (1) `pool_manifest.json`'s `identity_hash_inputs` now matches the complete pool-embedding-hash definition, and (2) diagnostics distinguish unfit species from failed fitted estimators; then search only for genuinely new load-bearing scientific flaws.

---

## 1. Confirmation: `identity_hash_inputs` now matches the pool-embedding-hash definition

The `pool_manifest.json` schema in "Failure handling" (lines 118-125) now reads:
```json
"identity_hash_inputs": {
  "windows_mapping_json_sha256": "...",
  "pool_csv_sha256": "<sha256 of the generated pool.csv>",
  "pool_builder_source_sha256": "<sha256 of build_embedding_pool_csv.py>",
  "pool_builder_config": {"window_version": "segmented_v4", "identity_units": "integer_samples"},
  "model_local_dir_sha256": "...",
  "extraction_params": {...}
}
```
This now contains every input the "Cache invalidation" section's prose (lines 326-332) and `fold_manifest.json`'s example (line 343) already required — `pool.csv`'s content hash, `build_embedding_pool_csv.py`'s source hash, and its relevant config values, in addition to the windows-JSON hash, model-dir hash, and extraction params. All three locations (schema, prose, fold-manifest example) and the corresponding test (`test_cache_hash_independence.py`, #6) are now mutually consistent. **Confirmed resolved.**

## 2. Confirmation: diagnostics distinguish unfit species from failed fitted estimators

The "Numerical trust" section (lines 275-282) now states explicitly:

> "`species_diagnostics_fold{i}.csv` is reindexed to the full global class vocabulary. Trainable, fitted species contain the diagnostics above; non-trainable species contain `NaN` for fit-derived fields, `numerically_trusted = False`, and their explicit eligibility columns. The schema must include `trainable`, `fit_attempted`, and `numerical_exclusion`, where `fit_attempted == trainable` and `numerical_exclusion == (fit_attempted and not numerically_trusted)`. No diagnostics are fabricated for estimators that were never fit. The 'excluded for numerical reasons' appendix filters `numerical_exclusion == True`; it must never filter on `numerically_trusted == False` alone, which would conflate unfit species with failed fits."

This directly resolves the ambiguity: a non-trainable species has `fit_attempted == False` and therefore `numerical_exclusion == False` regardless of its placeholder `numerically_trusted = False` value, so it is correctly excluded from the "numerically untrustworthy" appendix (it belongs only in the eligibility-based exclusion, not the numerical-trust one). A genuinely fitted-but-untrustworthy estimator has `fit_attempted == True` and `numerical_exclusion == True`. A new dedicated test, `test_diagnostics_exclusion_reasons.py` (#11), asserts exactly this distinction. **Confirmed resolved.**

## 3. Search for new load-bearing flaws

The remainder of the document (Why/What/How, Identity discipline, Environment's now-expanded failure-cause breakdown, Files, Commands, Phased milestones, remaining tests, Key operational failure modes, Future migration path) was re-read in full this round. No new contradiction, unresolved cross-reference, or scientifically load-bearing gap was found beyond the two items above, both of which are confirmations of already-fixed issues rather than new findings.

---

## Verdict

CONVERGED -- nothing to add.

STATUS: DONE
