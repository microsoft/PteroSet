# Experiment Guard Report — Approved PteroSet Species Split & Revised Perch v2 Plan

**Scope reviewed**: `docs/design/perch2_species_linear_probe_plan.md` (`STATUS: REVISED DRAFT`,
2026-07-28, the user-approved design), `docs/design/CHANGELOG.md` ("Approved-design correction"
section), the implementation `prepare_species_splits.py`, its test suite
`tests/test_prepare_species_splits.py`, and the generated artifacts in
`data/splits_species_v1/` (`train_split.csv`, `canonical_train_split.csv`, `val_split.csv`,
`test_split.csv`, `class_list.json`, `split_manifest.json`, `species_distribution.csv`).

**Not yet implemented / not reviewed as code** (no files exist for these): `extract_perch_embeddings.py`,
`train_perch_logreg.py`, `checkpoints/perch/species_v1/*`, `docs/implementation/species-linear-probe-v1/`.
Only the plan's Phase 1 (split construction) has been executed; Phases 0/2/3 (embedding extraction, `C`
selection, evaluation) exist only as design text. All findings below about `C` selection, no-bird
detection metrics, and test-touched-once are therefore **design-level (textual) verification only**,
not runtime-verified — this is flagged explicitly wherever relevant, not silently assumed.

**Methodology**: read the plan and changelog in full; read `prepare_species_splits.py` end to end;
ran the existing test suite; re-ran the split generator twice into scratch output directories with
the same seed to check byte-for-byte determinism against itself and against the checked-in artifacts;
independently recomputed proportions, hash inputs, and vocabulary/eligibility numbers from the
generated CSVs/JSON rather than trusting the manifest's self-reported figures at face value.

**Note on a live artifact change observed mid-review**: `data/splits_species_v1/` was regenerated in
place during this review (observed via `split_manifest.json`'s `config.num_restarts` changing from
`20` to `50`, and `canonical_base_sizes`/row counts shifting accordingly, with a file `Modify` time
after this review began). This corrected exactly the pre-registration drift noted below (see
"Reproducibility" → `num_restarts`). All findings in this report are based on the **current on-disk
state** as of the final verification pass (re-run and re-diffed after the change was observed), not
the earlier snapshot.

---

## Reproducibility

- **Seeds**: `--seed 42` is the pre-registered default in both the plan text and
  `prepare_species_splits.py`'s CLI, and is recorded in `split_manifest.json.config.seed`. The
  restart search (`assign_groups`) derives per-restart seeds deterministically as `seed + restart`
  (no unseeded `random` calls found in the reviewed script).
- **`num_restarts` pre-registration drift, now corrected**: the plan states "a deterministic,
  pre-registered set of seeds/restarts (default `R = 50`)", and its "Commands" section shows
  `--max_restarts 50`. The script's implemented default was **`20`** when this review began (and the
  first-generated `split_manifest.json` recorded `num_restarts: 20`) — a real, undocumented deviation
  from the plan's pre-registered default, with no reviewed-override note recorded anywhere. This was
  corrected to `50` (both the script default and the regenerated artifacts) during this review
  session. **Current state matches the plan's pre-registered default and is verified.**
  Recommendation: add a regression test asserting the script's `--num-restarts` default equals the
  value stated in the plan document, so this class of silent default drift is caught automatically
  rather than by manual doc-vs-code comparison.
- **Determinism of the actual split (verified empirically, not just by inspection)**: re-ran
  `python prepare_species_splits.py --output-dir <scratch>` twice with default args (seed 42,
  num_restarts 50) into two separate scratch directories, and diffed both against each other and
  against the checked-in `data/splits_species_v1/`. **`train_split.csv`, `canonical_train_split.csv`,
  `val_split.csv`, `test_split.csv`, `class_list.json`, and `species_distribution.csv` are
  byte-identical across both fresh reruns and the checked-in artifacts.** This is strong evidence the
  split assignment, vocabulary, and row construction are genuinely deterministic given `(seed,
  num_restarts)`, exactly as the plan requires ("the required contract is determinism given (seed,
  restart count)").
- **`split_manifest.json` is not byte-reproducible** (WARNING, see "Pitfalls Detected" below): the
  `vocabulary_fixed_point_history[*].support` dict's key order differs between runs of the identical
  script/seed, because it is built by iterating a `frozenset` of species-code strings, whose iteration
  order is subject to Python's default per-process string-hash randomization. The *content* is
  identical (same keys/values), only the serialized order differs, so this does not affect the actual
  split, vocabulary, or any downstream row — but it does mean `split_manifest.json`'s own sha256 is
  not stable across reruns of an otherwise-identical Phase 1, which matters because the plan's Phase 2
  schema (`embedding_manifest.json.identity_hash_inputs.split_manifest_sha256`) uses exactly this
  hash as a cache-invalidation/lineage key.
- **Environment/dependency capture**: `split_manifest.json` records `input_hashes` (sha256) for all
  four input files; independently recomputed sha256 of the four current input files
  (`windows_mapping_4.0overlap_segmented_v4.json`, `annotations_identification.json`,
  `annotations_species.json`, `species.csv`) and confirmed **exact matches** against the recorded
  hashes — the manifest's provenance claims are accurate, not stale. No `git_commit` field is recorded
  in `split_manifest.json` (only `embedding_manifest.json`'s schema requires this per the plan); adding
  it here too would strengthen full-pipeline provenance but is not a plan requirement for this file.
- **Config logging**: `split_manifest.json.config` records every value that actually drove the run
  (`ratios`, `seed`, `num_restarts`, `local_search_iters`, `min_sound_ids`, `geometry_decimals`,
  `grouping_unit`) — logged config matches runtime config; no silently-overridden defaults found in
  the reviewed code path.
- **Test suite**: `tests/test_prepare_species_splits.py` (51 tests) passes in full
  (`51 passed`) against the current script. The plan's "Tests" section lists 12 separate
  single-purpose files; the implementation consolidates equivalent coverage into one file. Coverage
  maps onto the same claims (grouping integrity, label-state derivation, taxonomy resolution,
  vocabulary gate, canonical/augmented separation, leakage validators, determinism) — a SUGGESTION,
  not a defect, but the plan's file list should be reconciled with reality.

## Data Integrity

- **Leakage risk: none detected.** `assign_groups` assigns whole `sound_id`s to exactly one split;
  `validate_no_leakage` and `validate_group_disjoint_from_rows` independently re-derive
  split→`sound_id` sets from the assignment dict and from the emitted rows respectively and assert
  pairwise disjointness. Both are exercised by dedicated tests
  (`test_validate_no_leakage_detects_overlap`,
  `test_validate_group_disjoint_from_rows_detects_overlap`) and both ran clean on the generated
  output (`sound_id_disjointness_verified: true` in the manifest, reproduced independently by
  re-deriving split membership directly from the CSVs).
- **`sound_id` grouping claims are accurate.** Confirmed no audio file (`sound_id`) appears in more
  than one of train/val/test in the actual CSVs. The plan's own "Why" section and
  `docs/lessons.md` explicitly and honestly disclose a residual limitation of `sound_id`-only
  grouping (recorder/site `event_indicator` codes can repeat within a project across dates, so a
  physical site *could* still contribute audio to more than one split) — this is not swept under the
  rug; it is the documented, user-approved trade-off for this *technical-validation* baseline
  ("no coarser grouping is required for this baseline"), not a claim of unseen-site generalization.
  Reviewed and found consistent between prose and implementation.
- **Overlap-window (dense/`4.0overlap`) augmentation claims are accurate.**
  `validate_train_augmentation_source` asserts every non-canonical `train_split.csv` row's `sound_id`
  is also present in `canonical_train_split.csv`; independently re-derived this from the CSVs
  (`train` = 96 107 rows = `canonical_train` 29 589 canonical rows + 66 518 rows flagged
  `is_canonical=0`, all from train-assigned `sound_id`s) and confirmed **zero** non-canonical rows in
  `val_split.csv`/`test_split.csv`/`canonical_train_split.csv` (all `is_canonical=1`). This matches the
  plan's contract exactly: val/test are canonical-only, overlapping windows are added to train only,
  after the split is fixed.
- **Preprocessing consistency (overlap definition)**: `classify_window`'s overlap test
  (`a.t_max > t_start and a.t_min < t_end`) is the same strict-interval convention already used by
  the production binary detector's label derivation in `prepare_dataset.py`
  (`a_min < we_sec and a_max > ws_sec`, `run_segment_windows`) — the new species pipeline does not
  silently introduce a different overlap rule from the one that produced the existing, trusted
  bird/no-bird labels.
- **No test-set leakage into modeling found in the implemented Phase 1 code.** The stratified split
  and the vocabulary gate both inspect val/test label vectors to build the partition and class list
  (explicitly permitted by the plan: "Label-aware split construction is allowed; test metrics are
  not"), but nothing in `prepare_species_splits.py` reads a post-fit metric — there is no fitting in
  this script at all. `C`-selection-on-validation / test-touched-once (the plan's other stated
  no-leakage boundary) **cannot be runtime-verified** because `train_perch_logreg.py` does not exist
  yet; the plan's textual design for it (grid search scored on val only, `argmax` with a lowest-`C`
  tie-break, one refit, one test evaluation) is internally coherent and contains no described path for
  test to influence `C*`, but this is a text review, not a code review, pending Phase 3.

## Training Configuration (design-level only — Phases 2/3 unimplemented)

- **`class_weight="balanced"` / no-bird prevalence coherence: coherent.** The generated
  `split_manifest.json` reports `no_bird_prevalence_canonical = 0.9096` (≈91%), computed exactly from
  the modeled canonical population, not assumed or hand-estimated (this supersedes an earlier ~92%
  estimate in `docs/design/CHANGELOG.md`, which that document itself frames as a preliminary figure
  from an earlier revision — no contradiction). Given this deliberately-preserved, natural ~91%
  majority class plus a long-tailed species-support distribution (per-species canonical support in
  `vocabulary_fixed_point_history` ranges from 7 up to 234 distinct `sound_id`s), fixing
  `class_weight="balanced"` rather than searching it is a reasonable, explicitly-justified choice
  consistent with the plan's stated rationale (unweighted LR would trivially predict all-negative for
  rare species). `C` selection via a small pre-defined grid, validation-macro-AP argmax, lowest-`C`
  tie-break, and exactly-once test evaluation is textually well-specified and does not conflate the two
  decisions (class weight is fixed by formula; only `C` is searched, only on val).
- **Metrics support the technical-validation framing.** Macro-AP over `test_evaluable` species as the
  sole headline, per-species AP+support reported separately, and a dedicated `any_bird_score`
  AUROC/AP against the `no_bird`/`species_window` split are all designed specifically to avoid
  rewarding the ~91% majority-class predictor (accuracy/micro-AP demoted to context-only) — internally
  consistent with the natural-imbalance framing established above.
- **Caveat worth surfacing in the eventual results report** (WARNING, not a design defect): several
  retained species have only 1 positive canonical window in `test_split.csv` (e.g. `AMAFAR`, `COEFLA`,
  `CYAAFF`, `DRYLIN`, `MESCAY`, `PSABIF`, spot-checked directly from `species_distribution.csv`), an
  unavoidable consequence of a `>=7`-`sound_id` vocabulary gate applied to a long-tailed, single
  70/15/15 split. Per-species AP for an `n_test_pos = 1` species is a near-binary, high-variance
  statistic; macro-AP as currently defined averages these equally with well-supported species. The
  plan already reports per-species support (`species_ap.csv`), so this is not hidden, but the go/no-go
  framing ("`macro_ap_summary` beats a per-species prevalence-only baseline by +5 points") should
  explicitly call out how much of any observed lift is attributable to a handful of thin-support
  species, to avoid over-reading the headline number once Phase 3 exists.

## Pitfalls Detected

1. **`split_manifest.json` non-determinism (WARNING)**: `vocabulary_fixed_point_history[*].support`'s
   key order is not stable across reruns of the identical script/seed (root cause: iterating a
   `frozenset` of species-code strings, whose order depends on Python's per-process hash
   randomization). Split/vocabulary/row artifacts themselves are unaffected and fully reproducible
   (empirically verified above); only this one diagnostic sub-object's serialized order varies. Fix:
   sort the `support` dict by key before assigning it into the history entry (or serialize with
   `json.dump(..., sort_keys=True)`), so `split_manifest_sha256` is stable across reruns of identical
   inputs — this matters because the plan's Phase 2 schema uses that hash as a cache-lineage key.
2. **Generated-artifact schema deviates from the plan's stated "binding contract" (WARNING)**: the
   plan text specifies `class_list.json` must include a `code_crosswalk` object mapping *every* raw
   `species.csv` code to its resolved outcome, and that the split CSVs carry `sound_filepath` plus
   "one 0/1 column per retained species." The actual generated artifacts instead: (a) store
   `class_list.json` as a flat `[{index, code, species}, ...]` list with no `code_crosswalk` key (the
   placeholder/duplicate-merge summary lives in `split_manifest.json.taxonomy_crosswalk` instead, and
   only lists the *exceptions* — five placeholders, two merge groups — not a full per-code mapping for
   all 168 known codes); (b) encode per-window species targets as a single `target_codes`
   (semicolon-joined) string plus a `target_vector` JSON-array string column, not one column per
   species; (c) use `sound_filename` (basename only) rather than `sound_filepath`; (d) report project
   distribution inside `split_manifest.json` (`project_distribution_canonical`/`_augmented_train`)
   rather than inside `species_distribution.csv` as the plan's "Files" section states; (e) use
   `resolved_clean` / `excluded_all_oov` / `unresolved_or_mixed` as label-state names in place of the
   plan's `species_window` / `excluded_rare_only` / `excluded_mixed_unresolved`. None of these change
   the underlying semantics (confirmed by direct inspection: the same information is present, just
   shaped/named differently, and is internally self-consistent and fully covered by the test suite),
   but a Phase 2/3 implementer following the plan document literally would write code against columns
   and objects that do not exist. Recommend reconciling the plan text and the script before
   `extract_perch_embeddings.py`/`train_perch_logreg.py` are written against either one.
3. **Vocabulary "present in all 3 splits" mechanism differs from the plan's described algorithm
   (WARNING, functionally sound but undocumented as implemented)**: the plan specifies a bounded,
   monotonic *retry* fixed point — build the split, drop any species absent from a split, rebuild,
   repeat (see "Class vocabulary and support gate," steps 1–5, "for every dropped species, the round
   and the failing condition (`n_sound_ids` or "missing from `<split>`")"). The implementation instead
   front-loads a deterministic `reserve_hard_constraints` pass that pins the minimum sound_ids needed
   so every `n_sound_ids>=7` species is *guaranteed* at least one positive `sound_id` per split before
   the optimizer runs at all, then hard-aborts (`AssertionError`/`ValueError`) if that is infeasible,
   rather than shrinking the vocabulary and retrying. This is arguably a stricter fail-loud posture
   (consistent with the rest of the plan's philosophy — e.g. the embedding-failure-ceiling design) and
   is fully tested (`test_reserve_hard_constraints_*`,
   `test_assign_groups_enforces_hard_species_presence_in_every_split`,
   `test_assign_groups_raises_on_infeasible_hard_constraint`), and it does produce the same *outcome*
   for the current data (`species_present_in_all_splits_verified: true`, 0 species dropped for this
   reason in `vocabulary_fixed_point_history`). But it means the plan's specific narrative — a species
   can be silently dropped for being absent from a split after the fact — is not what actually happens;
   the current behavior is "guaranteed by construction, or the whole run aborts," which is a materially
   different (safer, but different) failure mode than documented. Recommend updating the plan text to
   describe the reservation-based mechanism actually implemented.
4. **CLI flag-name differences from the plan's "Commands" section** are pre-authorized by the plan
   itself ("Exact flag names are illustrative... implementation may finalize them") and are therefore
   **not** findings — noted only as a SUGGESTION to sync the plan's example commands
   (`--windows-mapping`/`--num-restarts`/`--output-dir`/`--min-sound-ids`/`--geometry-decimals` vs. the
   plan's `--windows_mapping`/`--max_restarts`/`--out_dir`/`--min_sound_ids`/no `--config`,
   `--group_by`, or `--require_all_splits` flags at all) for a future reader's convenience.

## Verified Achieved-vs-Target Proportions (current on-disk artifacts)

- Canonical modeled population: 42 386 windows (38 555 `no_bird` + 3 831 `resolved_clean`), split
  29 589 / 6 398 / 6 399 → 69.81% / 15.10% / 15.10% (target 70/15/15; max deviation ≈ 0.19 pp on split
  size, ≈0.71 pp on `no_bird` prevalence per-split — both within the "small deviation" spirit of the
  plan, and both the deviations and the raw counts are recorded in `species_distribution.csv`/
  `split_manifest.json`, not hidden).
- 68/157 taxonomy-resolved candidate species retained (`n_sound_ids >= 7`); 5 placeholder codes and 2
  duplicate-name merges match the plan's stated, code-derived exclusions/canonicalizations exactly
  (`PSITTA`, `PSITTACIDAE`, `PSITTACIFORMES`, `RHACAR`, `TYRANN_SP1`; `ATAPIL`/`ATRPIL`,
  `RAMTUC`/`RHATUC`).
- All 5 projects (MAP1, PPA1–PPA4) present in train/val/test (report-only, per plan — never a
  stratification target, and confirmed the search never traded species/no-bird balance for it).
- Input-hash provenance in `split_manifest.json` independently re-verified against the actual current
  input files — exact match.

---

## Recommendations

- [ ] WARNING: Fix `split_manifest.json`'s `vocabulary_fixed_point_history[*].support` serialization
      to be order-stable (sort keys) so `split_manifest_sha256` is reproducible across reruns of
      identical inputs, before Phase 2 (`extract_perch_embeddings.py`) starts depending on that hash
      for cache-invalidation lineage.
- [ ] WARNING: Reconcile the plan document's stated artifact schema (`class_list.json.code_crosswalk`,
      per-species 0/1 columns, `sound_filepath`, `species_distribution.csv`'s project counts) with
      what `prepare_species_splits.py` actually emits (`taxonomy_crosswalk` summary in
      `split_manifest.json`, `target_codes`/`target_vector`, `sound_filename`, project counts in
      `split_manifest.json`) — pick one as authoritative and update the other before Phase 2/3 code is
      written against either.
- [ ] WARNING: Update the plan's "Class vocabulary and support gate" description to reflect the
      implemented reservation-based (`reserve_hard_constraints`) guarantee-by-construction mechanism,
      rather than the described drop-and-retry fixed point, which is not what the code does.
- [ ] WARNING: When Phase 3 (`train_perch_logreg.py`) is implemented, explicitly verify at runtime
      (not just by text review) that `C*` is selected only from `c_selection.csv`'s validation
      macro-AP column and that `test_emb.npz` is scored exactly once after `C*` is fixed — this report
      could only confirm the design is internally coherent on paper, since the script does not exist.
- [ ] SUGGESTION: Add a regression test asserting the script's `--num-restarts` default matches the
      plan's pre-registered default, to catch future silent drift (this review caught and the project
      already corrected one instance of exactly this class of drift, 20 → 50, during this session).
- [ ] SUGGESTION: Reconcile the plan's 12-file "Tests" list with the single consolidated
      `tests/test_prepare_species_splits.py` that actually implements equivalent coverage (51 tests,
      all passing).
- [ ] SUGGESTION: In the eventual `docs/implementation/species-linear-probe-v1/results.md`, explicitly
      call out per-species test support alongside macro-AP (already planned via `species_ap.csv`), and
      note how much of any macro-AP lift over the prevalence baseline is driven by species with
      `n_test_pos` in the single digits.

---

STATUS: PASS
