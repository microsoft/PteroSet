# Round 4 -- Clean Convergence Review

**Author role**: Chief Architect (architect-pipeline), Round 4
**Scope**: read only `docs/design/perch2_species_linear_probe_plan.md` as currently stored (the
in-place-replaced file, 546 lines). Verify self-containment, internal consistency, preservation of
the Orcas-style copy/adapt architecture, and clean resolution of all ten Round 3 revision items
(A-J). Search for genuinely new implementation flaws only -- no optional enhancements, no repeats of
findings already made in Rounds 1-3.

---

## 1. Self-containment and cleanliness checks (all pass)

- No reference to the stale filename `perch2_species_linear_probe_plan_REVISED.md` anywhere in the
  document (checked by direct grep).
- No "tool-disclosure" text (the "Disclosed tool limitation" paragraph and the "Note on provenance of
  this file" paragraph from the intermediate revision) remains anywhere in the document.
- No "Exact changes made in this revision (A-J)" changelog section remains -- the document reads as a
  clean specification, not a diff against a prior draft.
- The single remaining reference to earlier rounds (line 7-10: "Rounds 1-3 ... remain available for
  the exploratory reasoning ... but nothing in this document requires reading them") is an
  appropriate, historically-honest pointer, not a load-bearing cross-reference -- the document does
  not ask the reader to go elsewhere for any schema, threshold, or rule it depends on. Confirmed by
  reading the full document: every artifact schema (`pool_manifest.json`, `fold_manifest.json`),
  every gate rule, every threshold default, and every CLI flag is specified in full, in place.
- Document header correctly reads `STATUS: REVISED DRAFT -- pending Inquisitor sign-off`, not FINAL,
  matching Round 3 item (J)'s instruction that FINAL status requires a subsequent sign-off that has
  not yet occurred in this session.

## 2. Round 3 items A-J: each verified present and internally coherent, individually

| Item | Verified present at | Coherent? |
|---|---|---|
| (A) Exclusion/hard-gate reconciliation | "Failure handling" section, `pool_manifest.json`/`fold_manifest.json` schemas, the three-way present/excluded/abort rule | Yes |
| (B) Five eligibility categories | "Per-species-per-fold eligibility" section, exact table | Yes |
| (C) Integer-sample identity discipline | "Identity discipline" section, `pool.csv` column list, Files section | Yes |
| (D) Fallback env scoped to one script | "Environment" section | Yes |
| (E) Global + per-project failure ceiling | "Failure handling" section, `failure_ceiling` block, CLI flags | Yes |
| (F) Comparator retention + caveat | "Optional single-label comparator" section | Yes |
| (G) Two non-blended macro-AP tables | "Headline metrics" section | Yes |
| (H) Two independent cache-hash lineages | "Cache invalidation" section | Yes |
| (I) Per-estimator numerical trust diagnostics | "Numerical trust" section | Yes |
| (J) REVISED DRAFT status, not FINAL | Header and closing line | Yes |

Each item, read in isolation, is internally coherent and specified with concrete field names,
thresholds, and CLI flags rather than vague intent. This is a genuine improvement over the previous
draft's cross-referenced version.

## 3. Orcas-style copy/adapt architecture: preserved

The "Why," "How," and "Files" sections still describe exactly the reference architecture: raw
`tf.saved_model.load` (not `perch_hoplite`), a CUDA bootstrap, a single global pool extracted once,
gather-by-`window_id` fold materialization with hard gates (the direct generalization of
`remap_perch_embeddings.py`), `sklearn` + `joblib` (no PyTorch Lightning), flat dual-purpose scripts
with no new package layering, and CSV outputs. Nothing in this revision reintroduces the previously
removed per-window `.npy`/manifest system, Parquet shards, or a Lightning-based head. Confirmed clean.

## 4. New finding: eligibility-table ownership is stated inconsistently (load-bearing)

The document asserts **two different, mutually exclusive owners** for the same required artifact,
`species_eligibility_fold{i}.csv`:

- The architecture diagram (the `build_fold_embeddings.py` arrow, "How" section) and the "Files"
  section's one-line description of `build_fold_embeddings.py` both state that
  `build_fold_embeddings.py` "computes the per-species-per-fold eligibility table."
- The dedicated "Per-species-per-fold eligibility" section states, in its own words: *"This table
  (`species_eligibility_fold{i}.csv`) is a required artifact per fold, produced by
  `train_perch_logreg.py` before any metric is computed."* The Phase 3 milestone (which is
  `train_perch_logreg.py`'s phase, not `build_fold_embeddings.py`'s Phase 2) repeats this: *"
  `species_eligibility_fold{i}.csv` produced for all 5 folds before any metric is computed"* -- listed
  as a Phase 3, not Phase 2, deliverable.

This is a real, textually-verifiable 2-vs-2 self-contradiction (diagram + Files-section say one
script; the dedicated section + Phase 3 milestone say the other), not a matter of interpretation. It
is load-bearing, not cosmetic, for three concrete reasons:

1. **Phase acceptance criteria do not agree with each other as a result.** Phase 2's acceptance
   criteria (the `build_fold_embeddings.py` phase) make no mention of `species_eligibility_fold{i}.csv`
   at all, while Phase 3's acceptance criteria require it to already exist "before any metric is
   computed." If `build_fold_embeddings.py` is actually the producer (per the diagram/Files section),
   Phase 2's acceptance criteria are incomplete -- they should require this file to exist before Phase
   2 is considered done, but currently do not check for it.
2. **Environment consequences differ by owner.** `build_fold_embeddings.py` is specified elsewhere in
   this same document (the "Environment" section) as needing only pandas/numpy -- a pure metadata
   gather step. `train_perch_logreg.py` is where `scikit-learn`'s `LogisticRegression`/
   `OneVsRestClassifier` machinery lives. Computing eligibility requires only counting positives/
   negatives per split (no model fitting), so it is compatible with either script's stated
   dependencies -- but an implementer cannot know which file to put this function in without the
   document picking one.
3. **Duplication risk.** Left unresolved, an implementer following the diagram/Files section literally
   would implement the counting logic once in `build_fold_embeddings.py`, then encounter the dedicated
   section's explicit instruction and implement it again in `train_perch_logreg.py`, producing two
   divergent code paths for the same artifact with no stated reconciliation rule if they ever disagree
   (e.g. after a partial re-run of only one of the two scripts).

**Recommended resolution** (minimal, does not require redesigning anything else in the document):
make `build_fold_embeddings.py` the sole producer of `species_eligibility_fold{i}.csv`, since it is
the step that already gathers every split's rows and joins the species multi-hot matrix per split (so
computing `train_pos/train_neg/test_pos/test_neg` per species is a natural, cheap byproduct of work
`build_fold_embeddings.py` already does), and it runs earlier and without any GPU/sklearn dependency,
consistent with the pipeline's existing principle that Phase 2 is "pure array/metadata manipulation."
Concretely: (a) remove the sentence in the "Per-species-per-fold eligibility" section attributing
production to `train_perch_logreg.py`, replacing it with `build_fold_embeddings.py`; (b) move the
Phase 3 milestone's "`species_eligibility_fold{i}.csv` produced ... before any metric is computed"
bullet into Phase 2's acceptance criteria instead, rephrased as "... before Phase 2 is considered
complete"; (c) update `train_perch_logreg.py`'s own description in the "Files" section to say it
*consumes* (reads, does not produce) `species_eligibility_fold{i}.csv`. No other section of the
document needs to change; this is a three-sentence-level fix, not a new mechanism.

## 5. Nothing else load-bearing found

No other genuinely new inconsistency, gap, or contradiction was found on this pass. Specifically
checked and found consistent: hash-lineage ownership (`build_fold_embeddings.py`, stated identically
in the diagram, the "Cache invalidation" section, and the Files section -- no contradiction);
diagnostics/comparator-retention ownership (`train_perch_logreg.py`, stated consistently everywhere it
appears); the `multi_class` kwarg removal fact (unchanged from Round 3, correctly applied in both
classifier constructions); the identity-discipline rule (integer samples only, consistently applied in
`pool.csv`'s column list, the failure-handling gates, and the "Identity discipline" section, with no
stray seconds-based field anywhere in a persisted schema); and the two-macro-AP-table
never-averaged-together rule (stated once, referenced consistently, no section attempts to blend
them). This pass deliberately does not re-raise any item already decided in Rounds 1-3, and does not
propose any optional enhancement (e.g. friendlier naming for the two `min_support`-named thresholds at
different pipeline stages was considered and set aside as a non-load-bearing clarity nit, not reported
here per this round's explicit scope).

---

## Verdict

**NEEDS-MORE.**

Exactly one genuinely new, load-bearing issue was found: `docs/design/perch2_species_linear_probe_plan.md`
asserts two incompatible owners for `species_eligibility_fold{i}.csv` (`build_fold_embeddings.py` per
the architecture diagram and Files section, vs. `train_perch_logreg.py` per the dedicated eligibility
section and the Phase 3 milestone), which also leaves Phase 2's acceptance criteria silently
incomplete relative to whichever owner is correct. The fix is a small, three-part textual correction
(section 4 above gives the exact edits), not a redesign, and does not reopen any other decision in the
document. No other issues were found; everything else in the document is self-contained, internally
consistent, and preserves the Orcas-style copy/adapt architecture.

STATUS: DONE
