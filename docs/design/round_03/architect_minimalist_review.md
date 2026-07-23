# Round 3 Convergence Check -- architect-minimalist review of the proposed resolution

Author: architect-minimalist
Scope: evaluate the 9-point proposed resolution given in this round's prompt against the three
Round 2 proposals (docs/design/round_02/architect_minimalist_proposal.md [mine],
architect_experiment_proposal.md, architect_pipeline_proposal.md), looking only for issues
that are genuinely NEW -- i.e., not already raised, addressed, or reconciled somewhere in the
union of those three documents.

Note on inputs: no separate "Round 2 inquisitor findings" file was found on disk
(docs/design/round_03/ was empty before this document, and no *inquisitor* file exists in the
repo), so this review is based on: (a) a full read of all three Round 2 documents in this session,
(b) the 9-point resolution text as given verbatim in the prompt, and (c) one external verification
(scikit-learn's multi_class deprecation timeline) needed to check a factual claim in point 6.

Tagging convention (same as Round 1/2): [VERIFIED: repo - ...], [VERIFIED: source - ...],
[RECOMMENDATION], [NEW ISSUE] for items genuinely not present in any Round 2 document.

---

## 1. Point-by-point: what the resolution gets right and where it already converges

Going through the resolution's 9 points against the union of the three Round 2 documents:

1. **Mandatory multilabel OneVsRestClassifier(LogisticRegression(...)), single-label multinomial
   demoted to a restricted secondary arm, not a primacy-flipping gate.** This sharpens
   architect_experiment_proposal.md's Gate A (which left open the possibility of the single-label
   branch becoming "primary" if co-occurrence measured low) into a firmer, safer default: multilabel
   is always the headline result; single-label is always secondary, and can be *cancelled* by low
   support but never *promoted* to primary. This is a real improvement in ecological correctness
   (a 3% co-occurrence rate still means those windows' true multi-species content would be
   misrepresented by a single-label headline, regardless of how rare they are) and is consistent
   with all three Round 2 documents' shared instinct that multilabel is the safe default. Not a new
   issue -- a good resolution of an ambiguity that was already visible in architect_experiment's
   design.
2. **Extract once into one global pool NPZ; build fold/split views hard-gated by identity fields +
   source-JSON hash.** This matches the pool+remap architecture independently proposed by all three
   Round 2 documents (mine section 8, architect_experiment section 9 artifact naming, and
   especially architect_pipeline section 5's build_fold_embeddings.py, whose strengthened
   (sound_id, start, end) gate and source_windows_json_sha256 hash this point is clearly drawing
   from). The core idea is fully converged. Two genuinely new sub-issues are identified below
   (section 2, items A and C).
3. **Samples-to-seconds conversion, load the existing 5 s bounds at 32 kHz, peak 0.25.** Fully
   converged across all three documents (mine section 4's verbatim-load_audio_segment finding,
   architect_experiment section 5's identical algebraic argument, architect_pipeline line 121).
   No new issue.
4. **Never substitute silence; structured failure manifest; exclude; abort above 0.1%.** The 0.1%
   threshold and "log to manifest, exclude, don't silently zero-fill" policy is already specified
   almost verbatim in architect_experiment_proposal.md section 5 and its Phase 2 acceptance
   criterion. My own Round 2 draft was weaker here (it described the sibling's silence-substitution
   behavior as directly reusable without pushing back on it) -- the resolution correctly adopts
   architect_experiment's stronger position, not mine. Not new relative to the union of the three
   documents, but I flag one real interaction this creates with point 2's hard gate: see section 2,
   item A below.
5. **Test the existing bioacoustics env; use unchanged if the Perch smoke test passes; otherwise
   build a separate extraction-only env from verified pins rather than mutating the training env
   ad hoc.** This is a genuine tightening relative to my own Round 2 draft, which proposed
   pip install-ing any missing packages directly into the shared training env as the fallback --
   exactly the "ad hoc mutation" this point rejects, and a real (if latent) risk: an ad hoc install
   could silently bump numpy/torch-adjacent pins and destabilize train.py's existing Lightning
   pipeline, reintroducing the cross-framework conflict risk that the shared-env discovery was
   supposed to have retired. This is not something the resolution gets wrong; it correctly closes a
   gap in my own draft. One clarification is still missing (section 2, item D).
6. **Drop the multi_class kwarg for scikit-learn 1.7.2.** [VERIFIED: source - scikit-learn
   changelog/deprecation notices] multi_class was deprecated in scikit-learn 1.5 with removal
   targeted at 1.7; the pinned version in orcas_dclde2026/pip-requirements.txt is 1.7.2
   [VERIFIED: repo]. This means passing multi_class="multinomial" explicitly is very likely to
   already error (not just warn) at this pinned version. This is a real, useful catch -- and it is
   not merely cosmetic: architect_experiment_proposal.md's own recipe table (section 3, row for
   "Classifier") explicitly plans to retain multi_class="multinomial" for the single-label branch,
   which would need to be fixed before that branch's code would even run. architect_pipeline's own
   code sketch already omits the kwarg (uses bare LogisticRegression(solver='lbfgs', ...) inside
   OneVsRestClassifier), so it was already compliant by omission, likely by luck rather than by
   verifying this fact. No new issue; this point should be read as a confirmed, necessary fix to one
   of the two other Round 2 drafts' code sketches, not as new information for the design as a whole.
7. **Per-fold eligibility gated on train having positive+negative and test having support;
   structurally unseen classes excluded from the macro denominator.** This matches
   architect_experiment_proposal.md's Gate B almost exactly (eligible / structurally-unseen /
   untestable categories, train-support threshold, exclusion from macro averages). One genuinely
   new gap in how this point is phrased is identified in section 2, item B below.
8. **Window-level mandatory; segment-level optional; whole-file pooling prohibited.** This
   reconciles architect_experiment_proposal.md (which proposed segment-level pooling as a
   required secondary reporting granularity, gated by its own Gate C) with architect_pipeline_
   proposal.md (which rejected recording-level pooling outright and did not address segment-level
   pooling at all, since it never inspected the duty-cycled file structure). Demoting segment-level
   pooling from "required" to "optional" while keeping whole-file pooling "prohibited" is consistent
   with minimalism and does not contradict either source document (architect_pipeline's rejection
   was of *whole-file* pooling; it never considered segment-level pooling to reject). No new issue;
   one implication worth stating explicitly is that Gate C (segment-level co-occurrence) can be
   deferred until segment-level pooling is actually attempted, rather than computed unconditionally
   in Phase 0 -- a further, consistent simplification, not a gap.
9. **Delete Lightning head, per-window npy, Parquet, perch_hoplite, StandardScaler, required
   calibration/C-grid, separate package hierarchy.** Fully converged; every one of these deletions
   is independently proposed in all three Round 2 documents' comparison tables/removal lists. No
   new issue.

---

## 2. Genuinely new issues

These were not raised, in this form, in any of the three Round 2 documents. Each is a concrete
correctness or design-completeness gap surfaced only by reading the resolution's points against
each other and against the Round 2 documents' own stated motivations.

### A. The load-failure exclusion policy (point 4) is not reconciled against the hard existence gate (point 2)

[NEW ISSUE] Point 4 says failed-to-load windows must be **excluded** (not silently zeroed or kept
in the pool). Point 2 says the fold/split materialization step must hard-gate on "every window_id
in a fold's split CSV must exist in the pool -- else abort" (this is exactly
remap_perch_embeddings.py's proven pattern, adopted verbatim by all three Round 2 drafts). These
two rules collide: PteroSet's existing data/folds_segmented_v4/*/{train,val,test}_split.csv files
were built from the complete windows_mapping_4.0overlap_segmented_v4.json (160,244 windows) before
any Perch extraction ever ran. If a small number of windows fail to load during pool extraction and
are excluded per point 4, their window_ids will still be present in the fold CSVs -- and the
existence gate in point 2, applied literally, will then abort on the very first fold it processes,
every time, once even one window has a genuine load failure. Since a 0.1% failure allowance is
explicitly pre-registered as tolerable (point 4), this is not a hypothetical edge case to design
around later; it is the expected steady state.

**Fix, [RECOMMENDATION]:** the failure manifest from point 4 (a small, explicit list of excluded
window_ids) must be treated as an input to the point-2 gate, not an unrelated side effect. Before
running the "every window_id must exist in the pool" check, the fold materializer must first
subtract the failure manifest's window_ids from the fold CSV's expected set, and report the
resulting per-fold, per-split dropped-row count as its own small, visible number (not silently
absorbed into "the fold has fewer rows than expected" with no explanation). Only after this
subtraction should the strict existence gate run and abort on any *remaining* unexplained mismatch.
This keeps the "abort on any unexplained mismatch, never silently drop" property fully intact while
making the *explained* drops (load failures) visible and distinct from *unexplained* ones (a real
bug in the join).

### B. Test-side eligibility ("test has support") is necessary but not sufficient for the metrics actually proposed

[NEW ISSUE] Point 7's "test has support" (read against architect_experiment_proposal.md's Gate
B, which defines this as test_support >= 1, i.e. at least one *positive* window) is not enough to
guarantee that sklearn.metrics.roc_auc_score or average_precision_score can actually be computed
for that species in that fold: both functions require **at least one positive and one negative**
example in the evaluated set, or they raise (ValueError: Only one class present in y_true) or
return a degenerate score. For most species this will never bind, since PteroSet's own prior
measurements put positive rates in the 0.17-0.35 range even for the coarse any-bird task (finer
species-level rates will be lower still) -- but LOPO means the *test* set for a given fold is a
single project's windows only, and a project with very few total test windows combined with a
locally-common species is not a structurally impossible combination.

**Fix, [RECOMMENDATION]:** tighten the "test has support" criterion to require both a positive and
a negative example of that species in the fold's test partition, and add a fourth eligibility
category (distinct from eligible / structurally-unseen / untestable) for "test-degenerate": species
with adequate training data and at least one positive test window, but zero negative test windows
(or vice versa), so that the per-class metric genuinely cannot be computed as specified. This
category should be reported, not silently coerced into a 0.0/NaN score that would corrupt a macro
average if the exclusion logic (already required by point 7) has any gap.

### C. The hard-gate identity check risks comparing independently re-derived floats instead of the original integers

[NEW ISSUE] Point 2 specifies gating on window_id + sound_id + start + end + filepath, but does
not specify in *which units* start/end are compared. architect_pipeline_proposal.md's own
materialize_fold_split sketch (section 5) compares pool.window_start/end against
fold_csv.start/end, explicitly noting "(converted to the same units)" -- meaning both sides of the
comparison are seconds values independently derived by dividing sample indices by a sample rate.
Comparing floats for exact equality after independent unit conversions on both sides is a known
source of two different failure modes: spurious gate *failures* (harmless rounding differences
aborting a correct match) and, more concerning for a leakage-critical gate, spurious gate *passes*
(two genuinely different windows whose seconds values happen to coincide after truncation/rounding,
especially with 4.0overlap windows spaced closely together).

**Fix, [RECOMMENDATION]:** the identity/leakage gate must compare on the **original integer
sample-indexed** start/end (and sound_id, already an integer/stable key per Round 1), which is
exact and requires no floating-point tolerance at all. Seconds conversion (start/sample_rate)
should be treated as strictly a downstream, display-and-audio-loading-only transformation, never
participating in the identity comparison that gates whether a fold's cache is trusted.

### D. The point-5 fallback environment's scope should be stated explicitly, not left implicit

[NEW ISSUE, minor] Point 5 introduces a possible "extraction-only" environment as a fallback. None
of the three Round 2 documents, nor the resolution itself, states explicitly that this fallback (if
ever triggered) would only ever need to hold tensorflow/kagglehub/librosa for the one
Perch-forward-pass script (extract_perch_pool.py); the fold-materialization, label-join, training,
and evaluation scripts are pure NumPy/pandas/scikit-learn operations over already-materialized
.npz/.csv artifacts and have no reason to ever run outside the existing shared bioacoustics
env, regardless of which environment Stage A used. Leaving this unstated risks someone building a
second, fully duplicated environment for the whole pipeline instead of the one TensorFlow-dependent
script, re-introducing exactly the environment-maintenance burden the shared-env discovery (Round 2,
section 2) was meant to eliminate. [RECOMMENDATION]: state this scoping explicitly in the design
doc: at most one script (extract_perch_pool.py) is ever a candidate for a separate environment;
everything downstream of the pool .npz always runs in bioacoustics.

### E. The 0.1% load-failure threshold (point 4) should be checked per-project, not only globally, given LOPO

[NEW ISSUE] A single pooled 0.1% failure rate across all 160,244 windows can mask a much higher
failure rate concentrated in one project (e.g. a batch of corrupted files unique to one recorder
deployment). Under leave-one-project-out, a project-specific audio problem lands entirely inside
whichever single fold holds that project out as the test set -- silently shrinking or biasing
exactly that fold's test set while the global average looks fine. This risk is specific to
PteroSet's LOPO design and was not present in any of the three Round 2 documents' discussion of the
failure threshold (all framed it as a single pooled number).

**Fix, [RECOMMENDATION]:** report the load-failure rate per project as well as globally in the
Phase 1 extraction summary, and apply the 0.1% (or whatever pre-registered threshold is chosen) as a
per-project gate, not only a pooled one -- a project whose own failure rate is high should block
that fold specifically, even if the global average is comfortably under threshold.

---

## 3. Explicitly not re-raising (already settled, no new information)

To keep this review's "NEW-issues-only" scope honest: the CPU-vs-GPU perch_v2/perch_v2_cpu
discrepancy, the Perch 2 weights' license re-verification requirement, the min_support/
min_overlap_frac/co-occurrence-threshold exact numeric defaults, and the species class-list
curation (excluded_codes, OTHER bucket) are all already flagged as open items or already
converged across the three Round 2 documents and are unaffected, positively or negatively, by this
resolution. They are not repeated here as "new."

---

## 4. Verdict

**NEEDS-MORE.**

Five genuinely new, concrete issues (A-E above) were found by reading the resolution's points
against each other and against the union of the three Round 2 documents' own stated motivations --
none of them require reopening any already-converged architectural decision (multilabel-primary,
pool+remap caching, raw-SavedModel loading, sklearn LogisticRegression, no Lightning/perch_hoplite/
StandardScaler/required tuning). All five are narrow, load-bearing correctness fixes to the exact
mechanisms the resolution already specifies, not new machinery or new scope:

- **A.** Reconcile the load-failure exclusion policy with the hard existence gate (subtract the
  failure manifest before gating, report explained vs. unexplained drops separately).
- **B.** Tighten "test has support" to require both classes present in test, add a "test-degenerate"
  eligibility category distinct from structurally-unseen/untestable.
- **C.** Gate identity comparisons on original integer sample indices, never on independently
  re-derived seconds floats.
- **D.** State explicitly that only the Perch-extraction script is a candidate for a fallback
  environment; everything downstream stays in the shared bioacoustics env.
- **E.** Check the load-failure threshold per project, not only pooled, given LOPO's structural
  sensitivity to project-concentrated data problems.

None of these change the phased plan's shape or timeline; they are one-paragraph amendments to
Phase 0/1's acceptance criteria and to the fold-materialization script's specification. Once
incorporated, this reviewer would expect the next pass to reach CONVERGED.

---

STATUS: DONE
