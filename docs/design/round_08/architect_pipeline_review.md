# Round 8 -- Clean Convergence Review

**Author role**: Chief Architect (architect-pipeline), Round 8
**Scope**: read `docs/design/perch2_species_linear_probe_plan.md` in full (current state) as
currently stored. Specifically verify: (1) full-vocabulary diagnostics include `trainable`,
`fit_attempted`, `numerical_exclusion`; (2) numerical-exclusion appendix filtering is unambiguous;
(3) pool-manifest hash inputs match the cache-invalidation contract. Search only for genuinely new
load-bearing implementation flaws -- no repeats, optional enhancements, or style comments.

---

## 1. Round 7 finding: fully resolved, with an unambiguous three-column schema

The "Numerical trust" section (lines 275-282) now states:

> "The schema must include `trainable`, `fit_attempted`, and `numerical_exclusion`, where
> `fit_attempted == trainable` and `numerical_exclusion == (fit_attempted and not
> numerically_trusted)`. No diagnostics are fabricated for estimators that were never fit. The
> 'excluded for numerical reasons' appendix filters `numerical_exclusion == True`; it must never
> filter on `numerically_trusted == False` alone, which would conflate unfit species with failed
> fits."

This closes Round 7's gap precisely and unambiguously: the filter rule for the appendix is stated as
an exact boolean expression, not a vague reference to "eligibility status." Cross-checked against
downstream consumers: Test #11 (`test_diagnostics_exclusion_reasons.py`, lines 569-571) correctly
asserts the two distinguishing states (`fit_attempted == False, numerical_exclusion == False` for
unfit species vs. `fit_attempted == True, numerical_exclusion == True` for a failed fit). The Files
section, Phase 3 acceptance criteria, and the "Key operational failure modes" section all remain
consistent with this (none of them make a conflicting claim about how the appendix is filtered).
Resolved.

**Noted, not flagged**: the "Rule" paragraph immediately preceding this fix (lines 268-273, describing
`numerically_trusted == False` exclusion from headline aggregates and mentioning the appendix table)
predates the three-column clarification and is slightly less precise in isolation than the paragraph
that immediately follows and explicitly overrides it. Because the precise, correct, and explicit rule
is stated immediately afterward in the same section (not in some other document or a distant
section), a reader cannot actually be misled -- this is an editorial redundancy, not a new
load-bearing ambiguity, and is not raised as a finding.

## 2. Pool-manifest hash inputs vs. the cache-invalidation contract: now match exactly

The "Cache invalidation" section's **Pool embedding hash** prose (lines 326-330) lists five
components: (1) `windows_mapping_...json`'s SHA-256, (2) the generated `pool.csv` content SHA-256,
(3) the `build_embedding_pool_csv.py` source hash and relevant config values, (4) the vendored model
directory's content hash, (5) the extraction params (target sample rate, window seconds, target peak,
batch size).

`pool_manifest.json`'s `identity_hash_inputs` block (lines 118-125) now lists exactly six keys that
map 1:1 onto these five prose components (component 3 is split into two named keys):
`windows_mapping_json_sha256` (1), `pool_csv_sha256` (2), `pool_builder_source_sha256` and
`pool_builder_config` (3, split into source hash and config values), `model_local_dir_sha256` (4),
`extraction_params` (5, with all four named sub-fields: `target_sample_rate`, `window_sec`,
`target_peak`, `batch_size`). Every component named in the prose has a corresponding concrete JSON
key, and no key exists in the JSON schema that is not accounted for in the prose. This is a real fix
relative to the schema shown in earlier rounds (which carried only three of these keys, omitting
`pool_csv_sha256`, `pool_builder_source_sha256`, and `pool_builder_config`) and is now fully
reconciled with the contract stated in "Cache invalidation." Resolved.

## 3. Nothing else load-bearing found

Read the full document end to end this round (Why, What, How, Identity discipline, Failure handling,
Per-species-per-fold eligibility, Headline metrics, Numerical trust, Optional comparator, Cache
invalidation, Environment, Files, Commands, Phased milestones, Tests, Key operational failure modes,
Future migration path). Checked specifically for any further propagation gap of the
`trainable`/`fit_attempted`/`numerical_exclusion` schema (Files section, Phase 3 milestone, Commands,
comparator section) -- all consistent; the comparator's own diagnostics
(`comparator_diagnostics_fold{i}.csv`) fit a single joint multinomial model over retained classes
rather than K independent per-class binary fits, so it does not have the same
unfit-vs-failed-fit collision class of bug to begin with, and the document does not claim otherwise.
No other new ownership, schema, or cache-input contradiction was found. Not re-raised here, per this
round's scope: the previously-considered illustrative `pool_manifest.json` example detail
(`counts.excluded: 4` alongside a default `failure_ceiling` of `0`), unchanged and still judged
non-load-bearing.

---

## Verdict

CONVERGED -- nothing to add.

STATUS: DONE
