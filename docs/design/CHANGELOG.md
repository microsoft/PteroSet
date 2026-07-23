# Design Changelog: Perch v2 Species Linear Probe

This changelog records the design/review history behind
`docs/design/perch2_species_linear_probe_plan.md` and the rationale for how it reached its current,
final form. It exists so a reader can understand *why* the plan looks the way it does without having
to read all eight rounds of exploratory and review documents in `docs/design/round_01/` through
`docs/design/round_08/`. The plan document itself remains self-contained for implementation purposes;
this changelog is historical/process context only, not a dependency of the plan.

## Status

**FINAL**, converged in Round 8 (2026-07-23): all three reviewer perspectives
(`architect-pipeline`, `architect-experiment`, `architect-minimalist`) independently read the plan in
full and reported zero new load-bearing issues.

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
  in that document.
- **Exploratory alternatives and rationale considered but not adopted**: `docs/design/round_01/` and
  `docs/design/round_02/` (the three independent initial proposals) and
  `docs/design/round_03/architect_pipeline_review.md` (the synthesis that picked among them).
- **Issue-by-issue review history**: `docs/design/round_04/` through `docs/design/round_08/`, one
  subdirectory per round, one file per reviewer perspective.

These round directories are exploratory/review records and are intentionally left unedited; they
reflect what each reviewer actually found and wrote at the time, not the current state of the plan.
