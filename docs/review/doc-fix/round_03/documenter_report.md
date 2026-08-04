# Documentation Audit Report — Round 3

## Scope

Audited `docs/design/perch2_species_linear_probe_plan.md`, `docs/design/CHANGELOG.md`, and
`docs/lessons.md` against `prepare_species_splits.py` and the generated artifacts in
`data/splits_species_v1/` (`class_list.json`, `split_manifest.json`, `canonical_train_split.csv`,
`train_split.csv`, `val_split.csv`, `test_split.csv`, `species_distribution.csv`).

## Verification performed

- **Taxonomy exclusions**: plan and CHANGELOG both state exactly six placeholder codes excluded
  (`PSITTACIDAE`, `PSITTACIFORMES`, `RHACAR`, `PICIDA_1`, `TYRANN_SP1`, `PSITTA`) and two duplicate
  merge pairs (`ATAPIL`/`ATRPIL`, `RAMTUC`/`RHATUC`). This matches
  `split_manifest.json.taxonomy_crosswalk` exactly (`placeholder_codes_excluded` and
  `duplicate_code_merge_groups`). Confirms the Round-2 fix holds and no regression occurred.
- **Vocabulary size**: plan states "68 retained species" and `min_sound_ids >= 7`. Matches
  `split_manifest.json.vocabulary_size` (68), `vocabulary_gate_rationale` text ("yielded 68 species"),
  and `class_list.json` (68 entries, indices 0–67). Matches `prepare_species_splits.py`'s
  `--min-sound-ids` default (7) and the support-gate implementation (`build_vocabulary`,
  `n_sound_ids` gate).
- **Split sizes and proportions**: plan's "Current `species_v1` split" paragraph
  (29,513/6,398/6,474 canonical train/val/test windows, 69.64%/15.10%/15.27%, 95,536 augmented
  training windows, 90.96% no-bird prevalence) matches `split_manifest.json` exactly
  (`canonical_base_sizes`, `augmented_train_size`, `no_bird_prevalence_canonical` =
  0.909637843576737).
- **Seed/restarts**: plan and commands section specify `--seed 42 --num-restarts 50`, matching
  `split_manifest.json.config.seed` (42) and `config.num_restarts` (50).
- **Grouping unit**: plan states `sound_id` grouping (not site-level), matching
  `split_manifest.json.config.grouping_unit` = `"sound_id"` and `prepare_species_splits.py`'s
  argument names/logic (`min_sound_ids`, `n_sound_ids_per_split`, `sound_id_disjointness_verified`).
- **File/artifact names**: `class_list.json`, `split_manifest.json`, `species_distribution.csv`,
  `canonical_train_split.csv`, `train_split.csv`, `val_split.csv`, `test_split.csv` all exist under
  `data/splits_species_v1/` exactly as named in the plan's architecture diagram and "Files" section.
- **CHANGELOG history**: consistent with `docs/lessons.md`'s account of the site-grouping reversion
  and the five→six placeholder-exclusion correction; no new contradictions found between the three
  documents.

## Findings

None. No stale references, numeric mismatches, or contradictions were found in this round.

## Updated

- (none — no edits made or required)

## Issues Found

- (none)

## Needs Decision

- (none)

## Suggestions

- None beyond prior rounds. Documentation is currently consistent with code and generated artifacts.

---

STATUS: CONVERGED
