# Design Changelog: Perch v2 Species Linear Probe

This changelog records the design/review history behind
`docs/design/perch2_species_linear_probe_plan.md` and the rationale for how it reached its Round-8
FINAL form, plus the 2026-07-28 correction that superseded that form. It exists so a reader can
understand *why* the plan looks the way it does without having to read all eight rounds of
exploratory and review documents in `docs/design/round_01/` through `docs/design/round_08/`. The
plan document itself remains self-contained for implementation purposes; this changelog is
historical/process context only, not a dependency of the plan.

## Status

Round 8 (2026-07-23) converged the design as **FINAL** under a leave-one-project-out (LOPO) evaluation
structure: all three reviewer perspectives (`architect-pipeline`, `architect-experiment`,
`architect-minimalist`) independently read the plan in full and reported zero new load-bearing issues.

**That FINAL/LOPO design was subsequently superseded on 2026-07-28.** The user rejected fold4/PPA4-
style LOPO evaluation for the species task (see "Post-Round-8 correction" below). The replacement
task-specific split was implemented, tested, independently validated, and promoted to `STATUS: FINAL`
after a zero-finding documentation round. Rounds 1-8 below are preserved as the historical record of
how the *superseded* LOPO design was produced; they are not rewritten to reflect the correction, and
their conclusions about LOPO should not be read as still authoritative for the current plan.

**The post-Round-8 correction itself was then further revised, same day, to match the user's actually
approved design** (see "Approved-design correction" at the bottom of this file). Two elaborations the
agent introduced while drafting the post-Round-8 correction -- site-level `(project, event_indicator)`
grouping in place of `sound_id` grouping, and a three-metric (`n_windows`/`n_events`/`n_sites`)
vocabulary support gate -- were **not requested or approved** and have been reverted. The plan now
groups by `sound_id` and gates the vocabulary on `n_sound_ids >= 7` plus presence in all three splits,
per the user's explicit design. The intermediate entries below (the four "same-day follow-up lead
decisions") are preserved as the historical record of the unapproved elaborations and why they were
initially made; they are **no longer the current design** and should not be read as authoritative --
see "Approved-design correction" for what is current.

## Why this design exists

PteroSet's existing pipeline (`train.py`, `prepare_dataset.py`) performs bird/no-bird detection from
spectrograms. Species-level classification is a new objective. Rather than building a bespoke
extraction/training framework, the plan reuses the already-running, production-validated Perch v2
embedding + linear-probe recipe from the sibling repo `orcas_dclde2026`, adapted for PteroSet's two
structural differences: multilabel species targets (co-occurring birds per window) and evaluation over
5 fixed leave-one-project-out (LOPO) folds rather than one stratified split.

## Round-by-round history

- **Round 1** (`round_01/`): three independent architects (`architect-pipeline`,
  `architect-experiment`, `architect-minimalist`) each proposed a full design from scratch, exploring
  different points on the scope/rigor spectrum (a maximal, fully-instrumented pipeline; a
  minimalist first-answer baseline; and a middle-ground production-style proposal).
- **Round 2** (`round_02/`): each architect revised their own proposal after cross-reading the other
  two, narrowing disagreements.
- **Round 3** (`round_03/`): a convergence/synthesis check adjudicating a user-proposed "final
  substrate" against the three Round 2 proposals; this is where the plan was first assembled into a
  single working document (predecessor of the current `perch2_species_linear_probe_plan.md`).
- **Rounds 4-7** (`round_04/` through `round_07/`): iterative clean-convergence reviews of the
  single, in-place plan document. Each round surfaced genuinely new, narrow, load-bearing issues only
  (e.g. a two-way contradiction over which script produces `species_eligibility_fold{i}.csv`; a
  pool-manifest hash schema that didn't yet list every input the cache-invalidation contract required;
  a diagnostics schema that could conflate "never fit" species with "fit but numerically untrustworthy"
  species). Each issue was closed with a minimal, targeted textual fix before the next round, without
  reopening previously settled decisions.
- **Round 8** (`round_08/`): all three reviewers re-read the full, current plan end to end and
  reported **zero new findings** ("CONVERGED -- nothing to add", `STATUS: DONE` in all three files).
  This is the convergence signal that promoted the plan from draft to **FINAL**.

## What changed as a result of the review process

Concretely, the review rounds resolved (non-exhaustive, see individual round files for full detail):

1. Ambiguous ownership of `species_eligibility_fold{i}.csv` -- resolved to a single producer
   (`build_fold_embeddings.py`), consumed (not recomputed) by `train_perch_logreg.py`.
2. An incomplete `pool_manifest.json.identity_hash_inputs` schema -- extended to list every input
   (`pool_csv_sha256`, `pool_builder_source_sha256`, `pool_builder_config`, in addition to the
   windows-JSON hash, model-dir hash, and extraction params) that the prose already required the pool
   embedding hash to depend on.
3. A diagnostics schema that could not distinguish an un-trained species from a trained-but-untrusted
   fit -- resolved with an explicit three-field schema (`trainable`, `fit_attempted`,
   `numerical_exclusion`) and a dedicated test (`test_diagnostics_exclusion_reasons.py`).
4. Removal of transient in-document scaffolding (an intermediate "exact changes made in this revision"
   changelog section, a "tool-disclosure"/provenance paragraph, and references to a stale
   `_REVISED` filename) so the final plan reads as a clean specification rather than a diff against a
   prior draft. That process-level changelog was deliberately *not* restored inside the plan; this
   file is where that history now lives instead.

## Where to look

- **Implementation**: read only `docs/design/perch2_species_linear_probe_plan.md`. It is
  self-contained -- every artifact schema, gate rule, and threshold it depends on is specified in full
  in that document. As of 2026-07-28 its `STATUS` is `FINAL`; see "Approved-design correction" below.
- **Exploratory alternatives and rationale considered but not adopted**: `docs/design/round_01/` and
  `docs/design/round_02/` (the three independent initial proposals) and
  `docs/design/round_03/architect_pipeline_review.md` (the synthesis that picked among them). Their
  LOPO framing is historical context, not the current design.
- **Issue-by-issue review history**: `docs/design/round_04/` through `docs/design/round_08/`, one
  subdirectory per round, one file per reviewer perspective. This history explains how the
  *superseded* FINAL/LOPO plan reached its Round-8 state; it predates and does not address the
  2026-07-28 dataset-wide-split correction.

These round directories are exploratory/review records and are intentionally left unedited; they
reflect what each reviewer actually found and wrote at the time, not the current state of the plan.

## Post-Round-8 correction (2026-07-28): fold4/PPA4 LOPO rejected for the species task

After Round 8's FINAL sign-off, the user rejected the plan's leave-one-project-out (LOPO) evaluation
structure for the species-classification task specifically (the existing binary bird-detector task
correctly keeps LOPO; only the *new* species task's evaluation design changed). Full rationale lives
in the plan document's own "Why" section; summarized here for changelog purposes only, not as a
substitute for it:

1. **Label availability interacts badly with a per-project split.** Species labels only resolve for a
   subset of bird-positive windows; every window with an unresolved species annotation must be
   excluded from the modeled dataset (including mixed resolved+unresolved windows, to avoid injecting
   false negatives into every other species' negative class). Slicing an already-reduced label space
   five more ways by project risks per-fold species pools too small to support a meaningful macro-AP.
2. **Species distribution is not proportional across the 5 projects**, so a LOPO fold can make a rare
   species `structurally_unseen` for training or `test_absent` for evaluation, purely as an artifact of
   which project happened to be held out -- not a property of the species classification question
   being asked.
3. **LOPO answers a different, harder question** ("does a species classifier generalize to an unseen
   project?") than this baseline asks ("do frozen Perch v2 embeddings separate PteroSet species at
   all?"). Reusing the detector's LOPO structure silently reintroduced the harder question's data cost
   without being asked to answer it.

**Correction applied**: the plan now evaluates on one dataset-wide, group-aware, multilabel-stratified
70/15/15 train/val/test split that preserves species, no-bird, and project proportions, instead of 5
LOPO folds. True no-bird windows are kept as all-zero multilabel
targets at their natural prevalence rather than downsampled. The pipeline that
produces this is also substantially simplified as part of the same correction: three flat scripts
(`prepare_species_splits.py`, `extract_perch_embeddings.py`, `train_perch_logreg.py`) replace the prior
five-script, global-pool-plus-15-fold-view design; one macro-AP headline (`macro_ap_summary.csv`) plus
a dedicated no-bird/any-bird detection metric (`no_bird_detection.csv`) replace the prior two
never-blended per-fold macro-AP tables; the optional single-label comparator, the cross-fold
eligibility/numerical-trust machinery, and the dual pool/label-recipe cache-hash lineage are removed as
no longer applicable to a single split. This is a **technical-validation baseline**, not a
cross-project generalization claim -- LOPO-style species-generalization evaluation remains a legitimate
follow-on question, deliberately deferred, not answered by the corrected plan.

**Same-day follow-up lead decision (2026-07-28, after the correction above)**: canonical
non-overlapping 5-second windows (`start_sample % window_size_samples == 0`) are used for train, val,
*and* test -- not only val/test as the first draft of the correction had it. This makes the
stratified proportion table exact and auditable over one well-defined population and avoids
pseudo-replication from near-duplicate overlapping training windows; the dense/overlapping segmented
view is not retained for train in this baseline. Restricting to this canonical population also
changes the reported natural no-bird prevalence: it is **~92%** of the final modeled canonical
population (canonical windows, after `resolved_out_of_vocab`/`unresolved_or_mixed` exclusion), not the
~89.5% figure first estimated for a denser, non-canonical modeled population. `class_weight="balanced"`
(fixed, with `C=1.0`, no test tuning) remains the baseline classifier decision, now justified against
this ~92% figure and the long-tailed species support.

This correction, and the general lesson it produced (don't inherit a detector's LOPO/fold structure for
a differently-shaped task without first checking label availability and class balance under that
task's own exclusion rules), is also recorded in `docs/lessons.md`.

**Second same-day follow-up lead decision (2026-07-28, after both decisions above)**: the split's
grouping unit changed from `sound_id` to recording **site**, the `(project, event_indicator)` pair
joined from `data/metadata.csv`. `event_indicator` codes (recorder/deployment identifiers such as
`G001`, `CSA-02`) are reused across multiple recording dates within a project and are not guaranteed
unique across projects, so grouping by `event_indicator` alone would risk conflating distinct sites or
letting a name collision leak a site across a split boundary; pairing it with `project` fixes this.
Site grouping is strictly coarser than `sound_id` grouping (every `sound_id` belongs to exactly one
site), so it implies `sound_id` disjointness as a corollary rather than requiring a second, separately
maintained check. This decision also reframes the split's evaluation claim: the design is explicitly
**in-distribution** across stratified recording sites and projects -- every site and project appears,
in proportion, in train, val, and test alike -- and is not, and does not claim to be, an unseen-site or
unseen-project generalization result. Project proportion is now tracked as a **secondary**,
best-effort stratification target, deliberately not weighted equally with the species and `no_bird`
proportions (the primary targets), because this baseline's question does not require exact per-project
balance, only that no project is structurally starved out of a split.

**Third same-day follow-up (2026-07-28, taxonomy resolution rules)**: added an explicit rule for what
counts as a "resolved species code" at all, applied before the four-state label derivation. A code in
`data/species.csv` only resolves to a species if its `species` column is a real, two-part species-level
binomial -- not a dash/blank placeholder and not a higher-taxon name qualified by `sp.`/`sp <digit>`.
Applying this rule to the current `data/species.csv` excludes exactly six codes: `PSITTACIDAE`,
`PSITTACIFORMES`, `RHACAR` (all `–`, family/order/placeholder), `PICIDA_1` (`Picidae`, family-level),
`TYRANN_SP1` (`Tyrannidae sp 1`), and `PSITTA` (`Psittacidae sp.`) -- these are now treated identically
to a code absent from `species.csv`
(feed `unresolved_or_mixed`, never `resolved_clean`/`resolved_out_of_vocab`). Separately, two
duplicate-scientific-name code pairs are canonicalized to one vocabulary entry each, before any support
count or multi-hot encoding: `ATAPIL`/`ATRPIL` (both `Atalotriccus pilaris`, canonicalized to `ATAPIL`)
and `RAMTUC`/`RHATUC` (both `Ramphastos tucanus`, canonicalized to `RAMTUC`) -- both currently resolve
via a deterministic lowest-code tie-break. The rule, not the enumerated list, is the source of truth
`prepare_species_splits.py` implements, so a future `species.csv` edit is honored automatically; the
resulting crosswalk (raw code -> canonical entry or exclusion reason) is recorded in `class_list.json`,
and affected annotation/window counts are recorded in `split_manifest.json`, per "Taxonomy resolution"
in the plan document.

**Fourth same-day follow-up (2026-07-28, three-metric class-vocabulary support gate)**: replaced the
single-metric vocabulary gate (`min_class_support`, a positive-window count alone) with three
independent, pre-registered support metrics that a species must clear simultaneously: `n_windows`
(canonical non-overlapping positive windows, default `min_support_windows = 30`), `n_events`
(independent resolved annotation events -- distinct `annotations_species.json` rows, counted once
regardless of how many adjacent canonical windows one call spans, default `min_support_events = 10`),
and `n_sites` (distinct `(project, event_indicator)` sites with at least one resolved event, default
`min_support_sites = 5`). A single window count alone cannot distinguish many independent detections
from one long call chopped across several adjacent windows, and cannot rule out a species whose entire
evidence base is one recording location (a site-fingerprint risk, not a genuine acoustic-species
signal). Because `n_windows` depends on which windows survive label-state derivation, and label-state
derivation depends on the vocabulary, admission is computed as a bounded, monotonic **iterative
recheck**: a provisional pass computes all three metrics (independent of the vocabulary decision for
`n_events`/`n_sites`; dependent only on the already-fixed `unresolved_or_mixed` state for `n_windows`),
thresholds are applied to produce a tentative vocabulary, label states are derived from it, all three
metrics are recomputed restricted to windows that actually survive as `resolved_clean`, and any species
now failing a threshold is dropped -- repeating until no further species is dropped (guaranteed to
terminate, since the vocabulary only shrinks). The full support table (provisional and final values for
every candidate species, in-vocabulary or not) and each excluded species' drop round and failing
metric(s) are recorded in `class_list.json`/`split_manifest.json`. This three-metric gate is separate
from, and prior to, the existing post-split `trainable` gate in "Species eligibility" (a single,
train-split-only window-count check used purely for classifier column selection at fit time); the two
gates answer different questions and are not a duplicated threshold.

The plan document's own history, prose, and artifact schemas were updated in place to reflect this
correction; this changelog entry records *why*, not the full corrected design, which lives in the
plan document itself.

## Approved-design correction (2026-07-28, later same day): user-confirmed split design finalized

The four same-day follow-ups above were **agent-initiated elaborations of the post-Round-8
correction, not decisions the user had actually reviewed and approved**. When the user's actual
approved design was confirmed, it differed from the second and fourth follow-ups above in ways that
matter, and refined the first and third. **This entry supersedes the second follow-up (site grouping)
and the fourth follow-up (three-metric vocabulary gate) outright; those two are no longer the
design.** The first and third follow-ups' underlying ideas (canonical windows, taxonomy resolution)
are retained but the details below are now authoritative. The plan document
(`perch2_species_linear_probe_plan.md`) has been rewritten in place to match; this entry records what
changed and why.

1. **Grouping reverts to `sound_id`, not recording site.** The second follow-up's `(project,
   event_indicator)` site grouping was a real, defensible idea (coarser grouping is strictly safer),
   but it was never requested by the user and was not part of the approved design. The approved
   design groups by `sound_id` only: no audio file crosses a split boundary, and no coarser grouping
   is required for this technical-validation baseline. The general lesson -- an agent must not
   substitute its own "safer" elaboration for a design choice the user has not actually reviewed, even
   when the substitution looks more rigorous -- is recorded in `docs/lessons.md`.
2. **The vocabulary gate simplifies to two conditions**, replacing the fourth follow-up's three-metric
   (`n_windows`/`n_events`/`n_sites`) gate: a species is retained iff (a) it has calls spanning
   **`n_sound_ids >= 7`** distinct `sound_id`s, dataset-wide, and (b) it appears in **all three**
   splits (train, val, test) once the split is built. Condition (b) is guaranteed by reserving at
   least one positive `sound_id` per species and split before optimizing the remaining assignments;
   the generator fails if that hard reservation is infeasible. There is only one dataset-wide
   sound-based support metric now, not three, and no separate site-support concept.
3. **Label states simplify from four to three, and gain a "retained+rare" rule that the earlier
   drafts did not have.** A window is `no_bird` (all-zero, kept at natural prevalence), a
   `species_window` (kept, multi-hot over only the *retained* species among its overlapping calls) if
   every overlapping call resolves to a species code, or excluded if any overlapping call does not
   resolve to a species code at all (`excluded_mixed_unresolved`, folding the old `unresolved_or_mixed`
   state). The new rule: a window whose overlapping calls mix a retained species with a rare
   (below-gate) species is **kept**, encoding only the retained species -- it is no longer discarded
   outright the way the old `resolved_out_of_vocab` state discarded any window touching an
   out-of-vocabulary species. Only a window whose overlapping calls are *entirely* rare species is
   excluded (`excluded_rare_only`). This keeps genuine evidence for a retained species instead of
   discarding it for the sake of a stricter, but unnecessary, all-calls-in-vocabulary rule.
4. **Canonical windows are retained as the stratification/audit population, but overlapping windows
   are added back for train only, after the split is fixed** -- refining, not reverting, the first
   follow-up. Val and test remain canonical-only. `canonical_train_split.csv` (canonical only, used
   solely for the audited-distribution report) and `train_split.csv` (canonical + overlapping,
   "augmented," used for actual embedding/fitting) are now two distinct artifacts, distinguished by an
   `is_canonical` row flag in the augmented file, so the audited proportions and the actual training
   population are both fully specified without conflating them.
5. **Project distribution is report-only, not a secondary stratification target.** The second
   follow-up's "primary/secondary" tiering (species/no_bird primary, project secondary,
   best-effort-nudged) is dropped along with site grouping; species and `no_bird` proportions are the
   *only* stratification targets, and project's resulting proportion is simply computed and reported
   in `split_manifest.json` for transparency.
6. **`C` is selected from a small pre-defined grid using validation macro-AP, not fixed at `C=1.0`.**
   The classifier fits at every `C` in a pre-registered grid (default `{0.01, 0.1, 1.0, 10.0}`), scores
   validation macro-AP at each, and selects `C* = argmax`; the test split is scored exactly once, after
   `C*` is fixed, and is never used to choose `C` or any other setting. This replaces the earlier
   fixed-`C=1.0`, no-sweep decision (see the "What (scope)" out-of-scope list in earlier plan
   revisions) now that validation-only tuning is confirmed as approved and in scope.
7. **Artifact names and the split directory are renamed** to match the approved design exactly:
   `data/splits_species_v1/{canonical_train_split.csv, train_split.csv, val_split.csv,
   test_split.csv, class_list.json, split_manifest.json, species_distribution.csv}` (previously
   `data/species_splits/v1/{train,val,test}_windows.csv`). `c_selection.csv` is added under
   `checkpoints/perch/species_v1/` to record the `C`-grid sweep for auditability.
8. **Out-of-scope list gains explicit items** matching the approved design: cross-fold metrics and an
   arbitrary coefficient-norm exclusion gate are now named explicitly alongside the already-excluded
   single-label comparator, pooling, and Perch v1-vs-v2 ablation.

**Label-aware split construction remains allowed; test metrics are still never used to alter the split
or the model**, exactly as in the fourth follow-up's framing -- the only addition is that validation
(not test) macro-AP is now explicitly used to choose `C`, and this is the one and only place a
post-fit metric feeds back into any decision.

The plan document (`perch2_species_linear_probe_plan.md`) was rewritten to reflect this correction
directly and remains self-contained. The implemented split passed 52 targeted tests, invariant and
byte-determinism checks, independent data/experiment review, and a fresh zero-finding documentation
round; its `STATUS` is now `FINAL`. See `docs/lessons.md` for the general lesson about confirming
design choices before implementing or elaborating on them.

## Phase 2 implementation (2026-07-29): Perch v2 embeddings extracted

`extract_perch_embeddings.py` now implements the Orcas-aligned Perch v2 extraction boundary with
PteroSet-specific integrity checks: CUDA-library bootstrap before TensorFlow import, GPU-only
enforcement, exact 5-second/32-kHz/peak-0.25 preprocessing, sound-file-local decoding with bounded
global inference batches, fail-loud audio/model errors (no zero-waveform fallback), atomic outputs,
hash-gated resume, output-directory locking, and post-write NPZ-to-CSV round-trip validation.

The completed local artifacts under `data/embeddings/perch_v2/species_v1/` are:

- `train_emb.npz`: 95,536 x 1,536 float32 embeddings.
- `val_emb.npz`: 6,398 x 1,536 float32 embeddings.
- `test_emb.npz`: 6,474 x 1,536 float32 embeddings.
- `embedding_manifest.json`: zero exclusions/padding for all splits; model, extractor, split, class,
  dependency, GPU, and license provenance.

The extractor passed 114 targeted tests and the repository test set passed 166 tests. The 724-MiB
embedding artifacts and vendored Perch model are gitignored; only code, tests, and documentation are
intended for version control.

## Phase 3-4 completion (2026-07-29): linear probe trained and evaluated

`train_perch_logreg.py` now validates and L2-normalizes the Phase 2 embeddings, structurally prevents
test access during the validation sweep, fits balanced one-vs-rest logistic regressions over the
pre-registered C grid, selects the lowest-C validation macro-AP optimum, refits on train only, and
evaluates test once. The selected value is `C=0.1`.

The final test results are macro-AP `0.4089` (prevalence-only baseline `0.0020`), macro-AP `0.4817`
for species with at least five positive test windows, macro-AP `0.5738` for species with at least ten,
any-bird/no-bird AUROC `0.9437`, and any-bird/no-bird AP `0.7133`. All 68 estimators converged with
finite coefficients. Results and limitations are documented in
`docs/implementation/species-linear-probe-v1/results.md`; derived model/prediction artifacts under
`checkpoints/perch/species_v1/` are gitignored.

The final trainer uses `n_jobs=16` with BLAS thread pools limited to one thread per species fit. It
passed 97 focused tests; the complete repository suite passed 263 tests. Hash-gated resume was
verified against the final source, embedding NPZ, and output artifact hashes. Persisted test
probabilities and any-bird scores use float64 so published metrics reproduce exactly from the NPZ.

## Per-species AP/support analysis completion (2026-07-30)

`analyze_species_ap.py` now provides the completed descriptive analysis of per-species AP against
train, validation, and test representation. The generated report is
`docs/implementation/species-linear-probe-v1/ap_support_analysis.md`, with six supporting CSVs under
`analysis/` and three plots under `figures/`.

Across all 68 species, canonical-train positive windows have Spearman rho `0.573` with AP
(95% bootstrap CI `0.376`-`0.724`); validation and test positive-window correlations are `0.448` and
`0.521`. Controlling for test-positive support reduces the canonical-train partial rank association
to `0.303` (BH-adjusted q=`0.0606`), and restricting to species with at least 10 positive test windows
reduces the unadjusted canonical-train correlation to `0.203`. The analysis therefore records a
moderate full-sample association, not a causal or stable dose-response claim.

The final artifacts distinguish canonical from overlap-augmented train support, report both window
and distinct-`sound_id` counts, include AP lift above test prevalence, apply BH correction to the
reported correlation families, and use `sound_id`-cluster bootstrap intervals for per-species AP.
Intervals are explicitly non-estimable for the 10 species represented by only one positive test
audio file.
