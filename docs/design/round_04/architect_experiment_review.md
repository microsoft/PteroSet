# Round 4 — Clean Convergence Review of the Final Implementation Plan

**Author**: architect-experiment (Round 4 review)
**Input read**: `docs/design/perch2_species_linear_probe_plan.md` (full, all sections) and my own prior `docs/design/round_03/architect_experiment_review.md` findings. No other design-round document was consulted, per the task's explicit scope.

**Task**: verify that the five Round 3 findings are resolved in the plan, then search *only* for genuinely new scientific flaws — not re-litigate resolved items, not propose optional experiments or polish.

---

## 1. Verification of the five Round 3 findings

| Round 3 finding | Where resolved in the plan | Verdict |
|---|---|---|
| **Finding 1** — clean-single comparator selection bias undisclosed | "Optional single-label comparator" section: `comparator_retention.csv` with `retained_fraction_overall`, `retained_fraction_by_species`, `retained_fraction_by_project`; a required verbatim non-generalizability caveat attached to every comparator table/figure. | **Resolved**, matches the requested fix exactly. |
| **Finding 2** — cross-fold macro-AP undefined when eligible-species sets differ by fold | "Headline metrics" section: `macro_ap_core` (fixed intersection of species `evaluable` in all 5 folds, the only value ever averaged/reported as mean±std) vs. `macro_ap_fold_own_eligible` (five independent rows, explicitly never averaged, with a stated rule that doing so "is a bug"). | **Resolved**, matches the requested fix exactly. |
| **Finding 3** — provenance hash covered window geometry only, not the label-derivation chain | "Cache invalidation" section: two independent hash lineages — `pool_embedding_hash` (windows JSON + model dir + extraction params) and `label_recipe_hash` (`annotations_species.json` + `species.csv` + crosswalk + `class_list.json` + label config), both recorded per fold manifest. | **Resolved, and improved** on what was asked (two independent hashes, not one combined hash, so a label-only change never forces GPU re-extraction — a stronger outcome than my original recommendation). |
| **Finding 4** — "no silence fallback" under-specified + eligibility-count ordering | "Failure handling" section: explicit `pool_manifest.json` reasoned-exclusion schema, pre-registered global **and** per-project failure ceilings (default 0, override must be explicit and recorded verbatim before the run), `build_fold_embeddings.py`'s three-way present/excluded/abort rule, and an explicit "Support/eligibility computed after exclusions, never before" rule. | **Resolved**; the plan chose "abort by default, explicit pre-registered override" rather than my suggested "exclude-and-log by default" — a stricter, equally valid choice that still satisfies the underlying requirement (explicit, disclosed behavior; correct exclusion-then-eligibility ordering). |
| **Finding 5** — no convergence/near-separability diagnostic for per-class OvR fits | "Numerical trust per per-class estimator" section: `n_iter_`, `converged`, `coef_finite`, `coef_l2_norm`, `numerically_trusted` (gated on a configurable `coef_norm_ceiling`, explicitly flagged as an unverified starting default), excluded from all headline aggregates but retained, flagged, in a diagnostics table. | **Resolved**, matches the requested fix, with the ceiling constant honestly flagged as provisional. |

All five Round 3 findings are correctly and specifically resolved — no re-litigation needed.

---

## 2. New scientific flaw found

### The single-label comparator has no eligibility gate or numerical-trust diagnostic computed on its own (much smaller, non-uniformly retained) population

**Why this matters.** The "Per-species-per-fold eligibility" section computes `train_pos`, `train_neg`, `test_pos`, `test_neg` and the five eligibility categories "for each of the 5 LOPO folds and each canonical species in `class_list.json`" — this is defined once, ahead of the "Headline metrics" section, and is never re-scoped or duplicated for the comparator. The "Numerical trust" section is explicitly titled and scoped to "`OneVsRestClassifier` fits K independent binary models" — i.e., the mandatory multilabel branch only. The "Optional single-label comparator" section, which comes after both, specifies retention disclosure (Finding 1's fix) but does not state that its own per-species train/test counts (computed on the `clean_single`-retained subset, not the full species-positive population) get their own eligibility categorization, nor that its single multinomial-style `LogisticRegression` fit gets the same `n_iter_`/`coef_finite`/`coef_l2_norm`/`numerically_trusted` check.

This is not a hypothetical edge case independent of what the plan itself already establishes — it is the direct, foreseeable consequence of combining two mechanisms the plan already requires:

1. Finding 1's own fix documents that comparator retention is **uneven per species** (`retained_fraction_by_species`), meaning a species that clears the global `min_class_support`/eligibility bar in the full multilabel population can plausibly have far fewer, or even zero, surviving windows in the `clean_single` subset the comparator actually trains on.
2. Finding 5's own fix exists precisely because small-sample, high-dimensional (1536-d) logistic fits are prone to near-separability and silent non-convergence — and the comparator's per-class sample sizes, after retention shrinkage, are exactly the smaller, higher-risk regime that diagnostic was built to catch.

If the multilabel branch's eligibility table (computed on the full species-positive population) is reused, unmodified, to decide which species the comparator's results table reports, a species could be labeled "evaluable" for the comparator while its true retained training support is far below `min_support`, or even zero, in the population the comparator actually fit on — and no numerical-trust flag exists to catch a resulting non-convergent or degenerate comparator fit for that species. The comparator's entire purpose is direct numeric comparability with `orcas_dclde2026`'s methodology; a number quietly produced by an unconverged or near-separable fit on a handful of retained windows is the one place in this plan where a misleading headline number could still slip through both of the gates the rest of the document otherwise applies rigorously.

**Concrete fix**, scoped narrowly to the one script/table already responsible for the comparator, adding no new phase, artifact family, or modeling choice:
- `train_perch_logreg.py` computes a separate `comparator_eligibility_fold{i}.csv`, with the same five categories, from `train_pos`/`train_neg`/`test_pos`/`test_neg` counted **on the `clean_single`-retained subset only** (not the full species-positive population), per fold.
- The comparator's single multinomial `LogisticRegression` fit records the same fields already specified for the OvR estimators — `n_iter_`, `converged`, `coef_finite`, `coef_l2_norm`, `numerically_trusted` (computed once per fold for the joint fit, not once per class, since it is a single multinomial model rather than K independent binaries) — written to a `comparator_diagnostics_fold{i}.csv` alongside `species_diagnostics_fold{i}.csv`.
- Any comparator metric for a (fold, species) pair not `evaluable` under this comparator-specific table, or not `numerically_trusted` under this comparator-specific diagnostic, is excluded from `comparator_retention.csv`'s companion results table exactly as the primary branch already excludes such pairs from `macro_ap_core`/`macro_ap_fold_own_eligible` — the same rule, applied to the same object, just computed against the comparator's own population instead of reusing the primary branch's.

This is the only new issue found. It does not touch the primary `OneVsRestClassifier` branch, the pool/fold extraction architecture, the two-hash-lineage cache design, the exclusion-handling rules, or any milestone — it is a one-table, one-diagnostic extension of a mechanism the plan already builds twice (once for eligibility, once for numerical trust), applied a third time to a classifier the plan already trains but did not extend the same gating to.

---

## 3. Verdict

**NEEDS-MORE.**

All five Round 3 findings are correctly resolved, in one case (Finding 3) with a stronger mechanism than requested. Exactly one new, load-bearing gap was found: the optional single-label comparator does not get its own eligibility gate or numerical-trust diagnostic computed on its own retained population, even though the plan's own Finding-1 and Finding-5 fixes establish precisely the conditions (uneven, shrunk per-species retention; small-sample near-separability risk) that make this necessary specifically for that classifier. The fix is a narrow, mechanical extension of two mechanisms the plan already specifies elsewhere in full — no new phase, dependency, artifact family, or architectural change is required.

STATUS: DONE
