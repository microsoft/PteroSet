# Documentation Audit Report -- Round 2

**Scope**: `docs/design/perch2_species_linear_probe_plan.md`, `docs/design/CHANGELOG.md`,
`docs/lessons.md` cross-checked against `prepare_species_splits.py`,
`tests/test_prepare_species_splits.py`, and the generated `data/splits_species_v1/` artifacts
(`class_list.json`, `split_manifest.json`, `species_distribution.csv`,
`canonical_train_split.csv`/`train_split.csv`/`val_split.csv`/`test_split.csv`).

## Changes Applied

- `docs/design/perch2_species_linear_probe_plan.md` ("Tests" section, `test_taxonomy_resolution.py`
  entry): changed "reproduces the **five** current exclusions" to "reproduces the **six** current
  exclusions." The rest of the same document (Taxonomy resolution section, Phase 1 acceptance
  criteria) and `docs/design/CHANGELOG.md` both correctly state six excluded placeholder codes
  (`PSITTACIDAE`, `PSITTACIFORMES`, `RHACAR`, `PICIDA_1`, `TYRANN_SP1`, `PSITTA`), and
  `split_manifest.json.taxonomy_crosswalk.placeholder_codes_excluded` in the generated data confirms
  exactly six entries. The "five" wording was an internal contradiction within the plan document
  itself (not a code/doc mismatch) and has been corrected to match the rest of the document and the
  actual data.

## Findings (verified accurate, no other issues)

Checked and confirmed consistent between docs, code, and generated data:

- **Placeholder codes (6)**: `PSITTACIDAE`, `PSITTACIFORMES`, `RHACAR`, `PICIDA_1`, `TYRANN_SP1`,
  `PSITTA` -- identical across the plan doc, CHANGELOG, `is_placeholder_species_name` /
  `TaxonomyCrosswalk` in `prepare_species_splits.py`, and
  `split_manifest.json.taxonomy_crosswalk.placeholder_codes_excluded`.
- **Duplicate-code canonicalization pairs**: `ATAPIL`/`ATRPIL` -> `ATAPIL`, `RAMTUC`/`RHATUC` ->
  `RAMTUC` -- consistent across doc, code, and `split_manifest.json.taxonomy_crosswalk.duplicate_code_merge_groups`.
  Both are documented as deterministic lowest-code tie-breaks and match the generated manifest.
  `docs/lessons.md` also references the same two pairs accurately.
- **Label states**: doc table (`no_bird` / `species_window` / `excluded_mixed_unresolved`, with
  `excluded_rare_only` as the fourth recorded-but-excluded state) matches the runtime string values
  in `prepare_species_splits.py` (`NO_BIRD = "no_bird"`, `RESOLVED_CLEAN = "species_window"`,
  `UNRESOLVED_OR_MIXED = "excluded_mixed_unresolved"`, `EXCLUDED_ALL_OOV = "excluded_rare_only"`) and
  the test names in `tests/test_prepare_species_splits.py`
  (`test_finalize_label_state_excludes_window_with_only_rare_species`,
  `test_classify_window_mixed_resolved_and_unresolved_is_excluded`, etc.).
- **Reservation-based presence guarantee**: doc's 5-step vocabulary-gate procedure (provisional
  vocabulary -> reserve positive sound_ids per split -> optimize remaining assignment -> validate
  presence -> record) matches `reserve_hard_constraints` / `assign_groups` and their tests
  (`test_reserve_hard_constraints_reserves_one_sound_per_split_per_species`,
  `test_assign_groups_enforces_hard_species_presence_in_every_split`,
  `test_assign_groups_raises_on_infeasible_hard_constraint`).
- **Project report-only**: doc states project distribution is computed/reported but never a
  stratification target; `split_manifest.json` records per-split project counts alongside
  `"validation_gates_passed"` gates that do not include any project-balance gate, and
  `species_distribution.csv` contains no project column (proportions are species/no_bird only) --
  consistent.
- **CLI/schema**: `parse_args` defaults (`--train-ratio 0.70`, `--val-ratio 0.15`,
  `--test-ratio 0.15`, `--seed 42`, `--num-restarts 50`, `--min-sound-ids 7`, default paths under
  `data/` and `data/splits_species_v1`) match the doc's "Split construction" and "Phased milestones"
  sections exactly. Generated CSV header row
  (`window_id,dataset,sample_rate,sound_id,start,end,project,is_canonical,label_state,spec_name,
  sound_filename,target_codes,target_vector`) matches the doc's documented column list verbatim.
  `class_list.json` entry shape (`index`, `code`, `species`) matches the doc's description exactly.
- **Deterministic manifest**: `split_manifest.json` contains `seed: 42`, `num_restarts: 50`,
  `selected_restart_seed`, `min_sound_ids: 7`, `taxonomy_crosswalk`,
  `excluded_mixed_unresolved`/`excluded_rare_only` counts per split, and
  `validation_gates_passed` -- all as described in the plan's "Record, in split_manifest.json" bullets
  throughout the doc.
- **68 species and current counts**: the plan's "Current `species_v1` split" paragraph states 68
  retained species; 29,513/6,398/6,474 canonical train/val/test windows (69.64%/15.10%/15.27%);
  95,536 augmented training windows; 90.96% global no-bird prevalence. All five numbers were
  independently recomputed from the generated data and match exactly:
  `class_list.json` has 68 entries; `canonical_train_split.csv`/`val_split.csv`/`test_split.csv` have
  29,513/6,398/6,474 data rows (sum 42,385, matching `split_manifest.json.total_modeled_canonical_windows`);
  `train_split.csv` (augmented) has 95,536 rows; `split_manifest.json.no_bird_prevalence_canonical`
  is `0.909637843576737` (90.96%).
- **`docs/lessons.md`**: all factual claims (six placeholder/duplicate rows, `sound_id` grouping as
  the approved design, the site-grouping and three-metric-gate reversions, the LOPO-rejection
  rationale) match the current plan and CHANGELOG content and the current code; no stale references
  to superseded designs are presented as current.
- Cross-references to `docs/design/round_01/` through `docs/design/round_08/` and their per-round
  files (e.g. `round_03/architect_pipeline_review.md`) in the CHANGELOG resolve to existing files.

## Issues Found

- `docs/design/perch2_species_linear_probe_plan.md:614` (pre-fix): "five current exclusions" should
  have read "six," contradicting the same document's own "Taxonomy resolution" section (six codes
  listed) and Phase 1 acceptance criteria (six codes listed). **Fixed** (see Changes Applied).

## Needs Decision

- None.

## Suggestions

- No further changes recommended. The plan document, CHANGELOG, and lessons file are all internally
  consistent with each other, with `prepare_species_splits.py`'s actual CLI/schema/label-state
  behavior, with `tests/test_prepare_species_splits.py`'s coverage, and with the generated
  `data/splits_species_v1/` artifacts, once the single "five" -> "six" wording fix above is applied.

STATUS: CONVERGED
