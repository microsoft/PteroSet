# Round 4 Clean Convergence Review -- architect-minimalist

Author: architect-minimalist
Input: `docs/design/perch2_species_linear_probe_plan.md` (the current authoritative plan, read in
full, all 546 lines) and this reviewer's own Round 3 findings
(`docs/design/round_03/architect_minimalist_review.md`).

---

## 1. Verification that every Round 3 issue is resolved

Round 3 (this reviewer) raised five issues, corresponding to the six checks named in this round's
prompt (item A below covers both "success-vs-explicit-exclusion gather semantics" and
"post-exclusion eligibility," since the plan documents them together):

| Round 3 issue | Resolved in the plan? | Where |
|---|---|---|
| A. Load-failure exclusion vs. hard existence gate collision (success-vs-explicit-exclusion gather semantics) | **Yes.** The plan's `build_fold_embeddings.py` rule requires, for every `window_id` in a fold split, exactly one of: (1) a pool success -> gather, (2) an explicit `pool_manifest.json` exclusion -> record into the fold's own exclusion list, not counted in row totals, or (3) neither -> abort. This is precisely the fix requested. | "Failure handling" section, rule for `build_fold_embeddings.py` |
| A (cont). Post-exclusion eligibility (excluded rows must not count as implicit negatives) | **Yes**, stated explicitly and separately: "every per-species-per-fold count ... is computed only over rows that were actually embedded (excluded rows contribute to none of the four counts, not even as an implicit negative)." | "Support/eligibility computed after exclusions, never before" |
| B. Test-side eligibility needs both classes present, not just `test_pos >= 1` | **Yes.** `test_absent` (`test_pos == 0`) and `test_single_class` (`test_neg == 0`) are both explicit, independent categories; `evaluable` requires neither to hold. This is a stronger, more complete version of the fix requested (it also separates the two failure directions rather than collapsing them into one "test-degenerate" bucket, which is at least as good). | "Per-species-per-fold eligibility" table |
| C. Identity gate must use integer sample indices, never re-derived seconds floats | **Yes, and stronger than requested.** A dedicated "Identity discipline" section states every identity comparison, gate, and hash uses `(window_id, sound_id, start_sample, end_sample, sample_rate)` as integers; seconds are computed transiently, exactly once, immediately before the `librosa.load` call, and are "never written to any persisted artifact, never compared for equality, and never hashed." `pool.csv`'s schema is stated to contain no seconds column at all. | "Identity discipline" section |
| D. Fallback-environment scope must be stated explicitly as `extract_embeddings_pteroset.py`-only | **Yes.** A dedicated "Environment" section states the other four scripts need only pandas/numpy/scikit-learn/joblib, already present in `bioacoustics`, "No smoke test or fallback logic applies to these four scripts," and the fallback env, if built, "never propagates beyond the one script that needs TensorFlow/GPU." | "Environment: the fallback-env decision applies to `extract_embeddings_pteroset.py` only" |
| E. Per-project failure ceiling, not only a pooled one | **Yes.** `pool_manifest.json`'s `failure_ceiling` block has independent `global_max_failures` and `per_project_max_failures` fields (both default `0`), and the rule states the run aborts "the instant **either** ceiling ... is exceeded -- checked **both** globally ... **and** per-project." | "Failure handling" section, `pool_manifest.json` schema and rule |

All six named checks are resolved, and in three cases (B, C, D) the plan's resolution is stricter or
more complete than what Round 3 asked for, not merely adequate. No Round 3 finding is repeated
below.

---

## 2. Genuinely new issues

Searching the current plan itself (not the Round 3 resolution text) for load-bearing flaws not
previously identified. Four are found; a fifth, narrower completeness gap is noted separately since
it is a compliance/documentation gap rather than a technical-correctness one.

### 1. `macro_ap_core`'s denominator (species evaluable in all 5 folds) has no floor, and its size is never required to be reported

The plan defines the single headline go/no-go scalar, `macro_ap_core`, as the mean AP over the
species set that is `evaluable` in **all 5** LOPO folds simultaneously (the intersection). Given
PteroSet's five projects (MAP1, PPA1-4) are different sites/habitats with an already-established
likelihood of uneven species composition (this is the entire reason the per-fold eligibility table
exists at all), the intersection could plausibly be small, or in the worst case empty: a species
needs to be `trainable` in each of the four training-project combinations **and** `evaluable`
(non-`test_absent`, non-`test_single_class`) in the one held-out project, for every one of the 5
folds, to count. Two things are missing that would make this safe:

1. **No minimum size.** There is no stated floor (e.g. "if fewer than N species qualify, escalate to
   a design review rather than reporting a go/no-go number") for how small the intersection is
   allowed to be before `macro_ap_core`'s mean stops being a trustworthy summary of anything. A
   1-species or 0-species intersection would make the Phase 3 acceptance criterion ("beats a
   prevalence-only baseline by +5pp") either vacuous or an artifact of one arbitrarily-chosen
   species, not a real headline result.
2. **Size not required to be reported.** Unlike `macro_ap_per_fold_own_eligible.csv`, which is
   explicitly required to label "each row ... with its own N_species_eligible and species list,"
   `macro_ap_core_summary.csv`'s specification (Phase 3 acceptance criteria and the "Headline
   metrics" section) never requires `N_species_core` to be recorded alongside the mean/std. A reader
   of the final report cannot tell, from the artifact contract alone, whether the headline number
   reflects 40 species or 2.

**This is load-bearing**: it is the exact number the go/no-go milestone hinges on, and the plan's
own design (independent per-fold eligibility, explicitly expected to differ fold-to-fold) makes a
small intersection a real possibility, not a hypothetical one.

### 2. `pool_embedding_hash` does not cover `pool.csv`'s own generation, only the raw windows-mapping JSON

The "Cache invalidation" section defines `pool_embedding_hash` as "a function of only
`windows_mapping_4.0overlap_segmented_v4.json`'s SHA-256, the vendored model directory's content
hash, and the extraction params." `pool.csv` (produced by `build_embedding_pool_csv.py`) is a
**derived** artifact, not the windows-mapping JSON itself -- and this hash does not cover `pool.csv`'s
own content or `build_embedding_pool_csv.py`'s own logic/version. If that script's derivation logic
changes (a bug fix to how `start_sample`/`end_sample`/`project` are pulled out of the windows-mapping
JSON, for example) while the underlying windows-mapping JSON file's bytes are unchanged,
`pool_embedding_hash` will not change, and `extract_embeddings_pteroset.py` will treat the existing
`pool_emb_v2.npz` as still valid and skip re-extraction -- silently serving embeddings computed from
a stale or incorrect `pool.csv`.

This is precisely the class of bug this repository has already suffered once (the v2->v3 PPA4 label
regression cited in the Round 2 documents as the motivating precedent for hash-based cache
invalidation in the first place), just one artifact-hop further upstream than where the plan's
current hash coverage stops. The asymmetry is visible within the plan's own design: `label_recipe_hash`
*does* include its own derivation config values (`min_overlap_frac`, `min_class_support`,
`excluded_codes`), but `pool_embedding_hash` includes no equivalent for `pool.csv`'s derivation.

**Fix implied, not proposing new scope**: `pool_embedding_hash`'s inputs should include a hash of the
generated `pool.csv` file itself (or, equivalently, of `build_embedding_pool_csv.py`'s own source),
in addition to the windows-mapping JSON's hash -- this is a one-line addition to an already-designed
hash, not a new mechanism.

### 3. L2-normalization of embeddings before fitting is assumed, never specified as a required step

`sklearn.preprocessing.normalize(embeddings, norm="l2")` immediately before every
`LogisticRegression` fit/predict call is the exact, load-bearing convention this whole plan is built
on reusing from `eval_perch_ecotype.py`, and every prior round (1-3, all three Round 2 documents)
treated it as a required, explicit step. In this plan document, it appears exactly once, obliquely,
as a premise embedded in the `coef_norm_ceiling` default's justification ("start at `50.0` for
**L2-normalized** 1536-dim embeddings") -- it is never stated as a required step in the "How"
architecture diagram, in `train_perch_logreg.py`'s file description, or anywhere a reader could find
it as an instruction rather than infer it from a threshold's rationale. Since this document
explicitly states it is self-contained and "nothing in this document requires reading" Rounds 1-3,
an implementer following only this plan could miss this step entirely -- which would silently
diverge the numerical behavior of every fit from the validated recipe this plan's entire credibility
rests on reusing.

**This is load-bearing**, not cosmetic: omitting L2-normalization changes the effective regularization
geometry of the `C=1.0` L2-penalized fit relative to the sibling's validated numbers, and is exactly
the kind of silent methodological drift this plan otherwise goes to great lengths (two independent
hash lineages, five eligibility categories, a numerical-trust layer) to prevent elsewhere.

### 4. The Phase 0 smoke test bundles three different failure causes under one fallback that only fixes one of them

The smoke test's three pass criteria are (1) all five imports succeed, (2) at least one GPU device is
visible to TensorFlow, (3) one real forward pass succeeds without silent CPU fallback. All three
share a single documented remedy on failure: "build a separate, minimal extraction-only environment
pinned to `orcas_dclde2026/pip-requirements.txt`'s proven versions." That remedy can only address
criterion (1) (missing or wrong package versions) and, at best partially, (3) if the cause is a
software/driver mismatch. It does nothing for criterion (2): if the execution host simply has no
CUDA-capable GPU at all, building a differently-pinned conda environment cannot make one appear. The
plan states flatly, elsewhere, that "Perch v2's SavedModel is XLA-compiled CUDA-only" -- if that is
true without exception, a no-GPU execution context is a hard blocker with no remedy this plan
describes; if it is not an absolute fact (Round 1/2 found a separately-named `perch_v2_cpu` preset in
`perch-hoplite`'s source that neither this plan nor either sibling repo has empirically exercised),
the plan should say so and name it as the actual remedy for criterion-2 failures specifically, rather
than folding a hardware-availability problem into a software-environment fallback that cannot fix it.

**This is load-bearing** for exactly the same reason as the CUDA bootstrap itself is treated as
load-bearing elsewhere in the plan: a Phase 0 blocker with an ineffective documented remedy is worse
than one honestly marked "unresolved," because it will be tried, fail again for the same
unaddressed reason, and consume a cycle before anyone notices the fallback was never going to work
for that failure mode.

---

## 3. Noted separately: a documentation-completeness gap, not a technical flaw

The Perch v2 weights' license/usage-terms re-verification was flagged as an open item in Rounds 1-3
(re-verify against the live Kaggle model card before any publication-facing use) and has never been
marked resolved. This plan, which explicitly positions itself as the sole document an implementer
needs to read, does not mention it anywhere (no open-items list, no unresolved-risks section). This
does not affect the pipeline's technical correctness, but it is a real gap in this specific document's
claim to be self-contained: a compliance-relevant open item that existed in every prior round has
silently disappeared from the one document least likely to be cross-checked against those earlier
rounds. Noted for completeness; not counted as one of the four load-bearing technical findings above.

---

## 4. Verdict

**NEEDS-MORE**

Four genuinely new, load-bearing issues, none of which reopen any already-converged or
already-Round-3-resolved decision:

1. `macro_ap_core`'s cross-fold-intersection species set has no minimum-size floor and its size is
   not required to be reported alongside the headline mean.
2. `pool_embedding_hash` covers the raw windows-mapping JSON but not `pool.csv`'s own generation,
   leaving the same class of stale-cache risk this repo has already been bitten by once, one
   artifact-hop upstream of where the plan's current hash coverage stops.
3. L2-normalization of embeddings before every fit/predict call is assumed via a passing reference,
   never stated as a required step, in a document that claims to need no other reading.
4. The Phase 0 smoke test's single documented fallback (a re-pinned environment) cannot remedy a
   true GPU-hardware-absence failure, only a software/dependency-version failure, and the plan does
   not distinguish which of its three pass criteria that fallback actually fixes.

Plus one noted documentation-completeness gap (Perch v2 license re-verification, silently dropped
from this otherwise self-contained document) that does not itself block convergence but should not
be lost.

All four technical items are narrow, one-paragraph amendments to already-specified mechanisms
(add a floor + reporting requirement to an existing metric definition; add one more input to an
already-designed hash; state one already-assumed step explicitly; name the actual failure-specific
remedy for one of three smoke-test criteria). None require new pipeline stages, new artifacts, or
reopening any architectural decision.

---

STATUS: DONE
