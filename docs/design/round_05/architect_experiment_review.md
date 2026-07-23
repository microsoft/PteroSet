# Round 5 — Clean Convergence Review

**Author**: architect-experiment (Round 5 review)
**Input read**: `docs/design/perch2_species_linear_probe_plan.md` (full, current revision) and my own prior Round 4 finding. No other design-round document consulted, per the task's explicit scope.

**Task**: confirm the Round 4 comparator-isolation finding is resolved, then search only for genuinely new load-bearing scientific flaws — not repeats, not optional enhancements.

---

## 1. Confirmation of the Round 4 finding

Round 4 found that the optional single-label comparator had no eligibility gate or numerical-trust diagnostic computed on its own (smaller, non-uniformly retained) population, risking a misleading comparator number slipping through both gates the primary branch already enforced.

The current plan resolves this correctly and completely:

- **"Optional single-label comparator" section** now states explicitly: *"The comparator has its own gates because its retained `clean_single` population is smaller and non-uniform. `comparator_eligibility_fold{i}.csv` records per-class train/test support after retention and requires at least two train classes, the configured minimum train support for each reported class, and test support for every class included in a metric. `comparator_diagnostics_fold{i}.csv` records the multinomial fit's `n_iter_`, captured `ConvergenceWarning`, coefficient finiteness, and coefficient norms. Comparator metrics are withheld when these gates fail; the primary multilabel results are unaffected."*
- The architecture diagram, Files section, Phase 3 acceptance criteria, and `test_comparator_gates.py` (new) all consistently reference `comparator_eligibility_fold{i}.csv` and `comparator_diagnostics_fold{i}.csv` as artifacts distinct from `species_eligibility_fold{i}.csv`/`species_diagnostics_fold{i}.csv`, computed over the comparator's own retained population.
- `test_comparator_gates.py` explicitly asserts isolation: "verifies comparator support and numerical-trust failures withhold comparator metrics **without affecting the primary multilabel branch**."
- `macro_ap_core`'s own definition was independently tightened in this revision to also require `numerically_trusted` in all 5 folds (not just `evaluable`), and now includes a `min_core_species` pre-registration guard (Phase 0) that withholds the headline scalar entirely if the fixed cross-fold intersection is too small — a reasonable, self-contained hardening, not a response to any outstanding finding, and not a new issue.

**Confirmed: fully resolved, with correct isolation from the primary branch.**

---

## 2. New load-bearing issue found: contradictory ownership of `species_eligibility_fold{i}.csv`

The document names two different scripts as the producer of the same artifact, in different sections:

- **"Per-species-per-fold eligibility" section** (the section defining the five categories) states: *"This table (`species_eligibility_fold{i}.csv`) is a required artifact per fold, **produced by `train_perch_logreg.py`** before any metric is computed..."*
- **"Files" section** states `build_fold_embeddings.py` is the *"sole producer of the per-species-per-fold eligibility table after exclusions."*
- **"Phased milestones" section** places `species_eligibility_fold{i}.csv` production under **Phase 2** ("`species_eligibility_fold{i}.csv` is produced for all 5 folds after exclusions") and explicitly describes Phase 3 as `train_perch_logreg.py` **consuming** ("`train_perch_logreg.py` consumes the Phase 2 eligibility tables") rather than producing it.
- The "Failure handling" section's "Support/eligibility computed after exclusions, never before" rule is stated as part of `build_fold_embeddings.py`'s rules (the section immediately preceding and cross-referenced by "the eligibility table below"), further supporting Phase 2/`build_fold_embeddings.py` as the intended owner.

Two of three explicit attributions (Files, Phased milestones) agree the artifact is a Phase-2, GPU-free, `build_fold_embeddings.py` output that `train_perch_logreg.py` merely reads; one (the eligibility section's own defining text) says the opposite. This is not a cosmetic wording slip — it directly determines:

1. **Where a leakage/support-integrity-critical computation actually lives.** The whole point of Finding 4's fix (exclusions finalized before any support count is taken) is easiest to guarantee, verify, and test if the computation happens once, in the one script (`build_fold_embeddings.py`) that already has the finalized exclusion set in hand — not re-derived a second time inside the training script from whatever inputs it happens to be given.
2. **What `test_eligibility_categories.py` is actually testing**, and against which script's code path — the test's target is ambiguous until this is fixed.
3. **Whether two independent implementations could diverge or race.** As written, an implementer could reasonably build *either* script to write `species_eligibility_fold{i}.csv`, and a second implementer extending the pipeline later could add the same write to the *other* script, producing either a duplicate/overwriting-writer bug or two runs whose eligibility tables silently disagree because one was computed with an updated `min_support` and the other wasn't re-run.

**This is the only new issue.** It is a specification self-contradiction about artifact ownership, not a statistical, leakage, or calibration defect, and not a repeat of any Round 3/4 finding (none of which touched script-level ownership). It is load-bearing because reproducibility and leakage-control auditability both depend on there being exactly one, unambiguous producer of this artifact.

**Minimal fix**: delete or correct the single sentence in "Per-species-per-fold eligibility" ("produced by `train_perch_logreg.py`") to match the other two sections and the Failure-handling rule: `build_fold_embeddings.py` computes the post-exclusion counts and derives+writes `species_eligibility_fold{i}.csv` as part of Phase 2; `train_perch_logreg.py` reads it at Phase 3 and never recomputes or overwrites it. No new artifact, phase, or mechanism is introduced by this fix — it is a one-sentence correction to make the document internally consistent with itself.

---

## 3. Verdict

**NEEDS-MORE** — one new, narrow, load-bearing issue: an internal contradiction about whether `build_fold_embeddings.py` (Phase 2) or `train_perch_logreg.py` (Phase 3) produces `species_eligibility_fold{i}.csv`. Everything else, including the Round 4 comparator-isolation fix, is confirmed correctly resolved. The fix is a one-sentence documentation correction, not a design change.

STATUS: DONE
