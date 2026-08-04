# Data Validation Report: `data/splits_species_v1` vs. Approved Species-Split Design

**Scope**: independent, read-only audit of the *generated* artifacts in `data/splits_species_v1/`
(`train_split.csv`, `canonical_train_split.csv`, `val_split.csv`, `test_split.csv`, `class_list.json`,
`split_manifest.json`, `species_distribution.csv`) against the user-approved design in
`docs/design/perch2_species_linear_probe_plan.md` (`STATUS: REVISED DRAFT`, 2026-07-28) and its
implementation, `prepare_species_splits.py`. Every number below was **independently recomputed from
source** (`data/windows_mapping_4.0overlap_segmented_v4.json`, `data/annotations_identification.json`,
`data/annotations_species.json`, `data/species.csv`) using a from-scratch script, not copied from
`split_manifest.json`'s self-reported figures, except where explicitly noted as a manifest
cross-check. No file in the repository was edited as part of this review.

**Note on a live artifact change observed mid-review**: `data/splits_species_v1/` and
`prepare_species_splits.py` were regenerated/edited in place while this review was in progress
(`split_manifest.json`'s `config.num_restarts` changed from `20` to `50` between two reads, matching
the plan's pre-registered default `R = 50`; label-state string values were also renamed from internal
placeholders (`resolved_clean`, `excluded_all_oov`, `unresolved_or_mixed`) to the plan's exact
vocabulary (`species_window`, `excluded_rare_only`, `excluded_mixed_unresolved`)). File mtimes were
polled and confirmed stable for >20 minutes before the findings below were finalized; all findings are
based on that final, stable on-disk state, re-verified end-to-end after the change (not an earlier
snapshot).

---

## Independent recomputation summary

| Check | Independent result | Manifest claim | Match |
|---|---|---|---|
| Placeholder codes (non-species) | `PSITTA, PSITTACIDAE, PSITTACIFORMES, RHACAR, TYRANN_SP1` | same 5 | Yes |
| Duplicate-name canonicalization groups | `{ATAPIL: [ATAPIL, ATRPIL]}, {RAMTUC: [RAMTUC, RHATUC]}` | same 2 groups | Yes |
| Label-state counts, all windows | `no_bird=125171, species_window=13764, excluded_rare_only=855, excluded_mixed_unresolved=20454` | identical | Yes |
| Label-state counts, canonical windows | `no_bird=38555, species_window=3831, excluded_rare_only=218, excluded_mixed_unresolved=6864` | identical | Yes |
| Vocabulary (>=7 distinct sound_ids, canonical `species_window` windows, post-canonicalization) | 68 species, exact set match | identical 68-species list | Yes |
| Vocabulary fixed-point convergence | round 0 -> 68 species (83 dropped for `n_sound_ids<7`), round 1 -> 0 further drops | 2-round history, same drop list, same terminal size | Yes |
| Canonical population size | 42,386 windows | `total_modeled_canonical_windows=42386` | Yes |
| Canonical split sizes | train 29,589 / val 6,398 / test 6,399 | identical | Yes |
| Augmented train size | 96,107 rows | identical | Yes |
| `n_sound_ids` per split | train 387 / val 89 / test 87 (sum 563) | identical | Yes |
| No-bird prevalence (canonical, global) | 90.96% | identical | Yes |

---

## Item-by-item validation

**1. `sound_id` disjointness** -- PASS. Re-derived split membership directly from row contents (not
the assignment dict) for all four CSVs: `canonical_train_split.csv` sound_ids == `train_split.csv`
sound_ids exactly (the augmented view adds rows, never new sound_ids); pairwise intersection of
`{ctrain, val, test}` sound_id sets is empty in all three pairs; `window_id` is unique within every
split file and never repeats across `train`/`val`/`test`. Matches `split_manifest.json`'s
`sound_id_disjointness_verified: true`.

**2. Canonical 70/15/15 sizes** -- PASS. Canonical-only population (`canonical_train_split.csv` +
`val_split.csv` + `test_split.csv` = 42,386 windows) splits **69.81% / 15.09% / 15.10%**, within
~0.9 percentage points of the 70/15/15 target on every split -- a materially better fit than the
pre-review snapshot (69.68/15.12/15.20) observed before the concurrent `num_restarts` 20->50 fix
landed. `n_sound_ids` breakdown (387/89/87 of 563) is consistent with this window-count ratio.

**3. Every retained species present in all three splits** -- PASS. Independently scanned
`target_codes` in `canonical_train_split.csv`, `val_split.csv`, `test_split.csv`; all 68 vocabulary
codes appear with >=1 positive window in every one of the three files (0 missing). Confirmed this is
enforced structurally, not by luck: `reserve_hard_constraints()` pins one positive sound_id per
species per split *before* the free-assignment optimizer runs, and `validate_species_present_in_all_splits()`
re-asserts it post-hoc and would raise `AssertionError` otherwise (code-read, not just manifest trust).

**4. `>= 7` source `sound_id`s** -- PASS. Recomputed per-species distinct-`sound_id` support from raw
annotations/windows completely independently of `prepare_species_splits.py` (own geometry-matching,
taxonomy-resolution, and windowing logic re-implemented from scratch): the resulting 68-species set at
threshold >=7 is an **exact match**, in both membership and the list of 83 species dropped for
insufficient support (e.g. `AMAOCH, AMMAUR, ANTNIG, ...`). The gate is computed over canonical windows
only, before split assignment, per the design's "Provisional vocabulary V0" step.

**5. No-bird all-zero only** -- PASS. Every row in `train_split.csv`/`val_split.csv`/`test_split.csv`
was checked: `target_vector` is all-zero **iff** `label_state == "no_bird"`, in both directions
(0 violations either way, across 96,107 + 6,398 + 6,399 rows).

**6. Unresolved/mixed windows excluded** -- PASS. Only two `label_state` values ever appear in any
output row: `no_bird` and `species_window`. `excluded_mixed_unresolved` and `excluded_rare_only` never
leak into any split file (verified directly, and also enforced by `validate_rows()`'s
`MODELED_STATES` check, read in code).

**7. Placeholder codes excluded** -- PASS. None of the 5 placeholder raw codes
(`PSITTA, PSITTACIDAE, PSITTACIFORMES, RHACAR, TYRANN_SP1`) appear anywhere in any `target_codes`
column across all three splits (0 occurrences).

**8. Duplicate codes merged** -- PASS. Neither raw duplicate code (`ATRPIL`, `RHATUC`) ever appears in
`target_codes`; only the canonical codes (`ATAPIL`, `RAMTUC`) do. Verified this merge is
**load-bearing, not cosmetic**: raw annotation counts are `ATAPIL=4` vs. `ATRPIL=180`, and
`RAMTUC=170` vs. `RHATUC=23` -- `ATAPIL` alone would very likely fail (or sit right at the edge of)
the `>=7 sound_ids` gate without the 180 `ATRPIL` annotations merged in; the canonicalization is
doing real work here, not merging two already-adequate codes.

**9. Retained+rare window semantics** -- PASS, verified on a concrete example. `window_id=108`
(`sound_id=0`) overlaps three raw resolved calls (`CAMNUC`, `LATALB`, `PITSUL`); `CAMNUC`/`LATALB` are
both below the support gate and `PITSUL` is retained. The window appears in the output with
`label_state=species_window` and `target_codes=PITSUL` only -- the rare co-occurring calls are
silently dropped from the target vector without excluding the window or manufacturing an OOV label,
exactly as specified ("a rare species call co-occurring with a retained species call ... contributes
nothing ... to the target vector").

**10. Rare-only windows excluded** -- PASS, verified on a concrete example. `window_id=150`
(`sound_id=0`) overlaps only `ELAFLA`, a species below the support gate with no other overlapping
call. This window is **absent from all three split files** -- correctly excluded as
`excluded_rare_only` rather than being coerced into `no_bird` (which would have corrupted the
no-bird class, per the design's explicit rationale).

**11. Train augmentation only from train `sound_id`s** -- PASS. Every row in `train_split.csv` with
`is_canonical=0` (66,518 augmented rows) has a `sound_id` that is a member of
`canonical_train_split.csv`'s sound_id set (verified as a strict subset check over all rows, not
sampled) -- no augmented row originates from a val/test-assigned sound_id. Matches
`validate_train_augmentation_source()`'s guarantee, independently re-derived.

**12. Val/test canonical-only** -- PASS. `val_split.csv` and `test_split.csv` contain **zero**
`is_canonical=0` rows (checked directly, not inferred from manifest).

**13. Natural no-bird prevalence (not downsampled)** -- PASS. Global canonical no-bird prevalence is
90.96%, with per-split prevalence at 91.26% (train), 90.25% (val), 90.30% (test) -- all close to the
global figure and none artificially rebalanced toward 50/50 or any other target. This is the raw,
naturally-occurring prevalence of the modeled canonical population (`no_bird` windows are never
downsampled anywhere in `build_rows()` -- confirmed by code read: every classified `no_bird`/
`species_window` canonical window for an assigned sound_id is emitted, with no sampling/filtering
step after label-state classification).

**14. Per-species proportion deviations** -- PASS, with an expected, disclosed limitation.
`species_distribution.csv`'s `*_frac_deviation` columns show small **absolute** deviations
dataset-wide: max is 0.71 percentage points (`no_bird`, val split); the largest species-level absolute
deviation is 0.45pp (`CYAVIO`, test). In **relative** terms, several boundary-support species (those
just above the 7-`sound_id` gate, e.g. `SYNCAN`, `ARRCON`, `TYRELA`) show 1-2x relative deviation from
their own global proportion in one split -- this is the same structural limitation the prior
(pre-implementation) `data_validator_report.md` flagged for any group-constrained split of low-support
species, and the design doc explicitly accepts it ("as close to 70/15/15 as the hard grouping/coverage
constraints allow"), rather than promising uniform per-species proportion preservation. Not a defect.

**15. Deterministic regeneration** -- PASS for all data-bearing artifacts, with one cosmetic exception
flagged. Re-ran `prepare_species_splits.py` (unmodified, same CLI defaults) twice into scratch
directories and compared all outputs against each other and against the checked-in
`data/splits_species_v1/` via SHA-256:
- `train_split.csv`, `canonical_train_split.csv`, `val_split.csv`, `test_split.csv`, `class_list.json`,
  `species_distribution.csv`: **byte-identical** across the checked-in artifact and two independent
  fresh reruns.
- `split_manifest.json`: **not** byte-identical across separate process invocations (confirmed with a
  field-by-field structural diff that the two manifests are semantically identical -- same values, same
  keys -- only the iteration order of the `support` sub-dictionaries inside
  `vocabulary_fixed_point_history` differs). Root-caused to Python's per-process string-hash
  randomization (`PYTHONHASHSEED`) affecting `frozenset` iteration order when that dict is built;
  confirmed the fix by re-running twice with `PYTHONHASHSEED=0` pinned, which produces byte-identical
  manifests. This does not affect split membership, vocabulary, row content, or any value a downstream
  consumer reads -- only the informal ordering of one diagnostic sub-object in the manifest file. The
  existing `test_determinism_across_reruns_with_same_seed` test checks content-level determinism
  (vocab, assignment, window_id sets) and would not catch this, since it never serializes/diffs the
  full manifest JSON text.

**16. Manifest accuracy** -- PASS for the fields it contains, with two schema-completeness gaps
against the written design contract (see "Deviations" below). Independently recomputed and matched
exactly against `split_manifest.json`: `canonical_base_sizes`, `augmented_train_size`,
`no_bird_prevalence_canonical`, `n_sound_ids_per_split`, `label_state_counts_*`, `vocabulary`/
`vocabulary_size`, `taxonomy_crosswalk`. All match to full precision.

---

## Deviations from the written design (schema/documentation gaps, not data-correctness defects)

These do not change any split membership, label, or statistic verified above; they affect only where
some already-correct information is recorded, or a docstring's internal accuracy.

- **`class_list.json` is missing the `code_crosswalk` object the design requires.** The design states
  (`docs/design/perch2_species_linear_probe_plan.md`, "Taxonomy resolution"): *"`class_list.json`
  includes a `code_crosswalk` object mapping every raw `species.csv` code to its resolved outcome ...
  `split_manifest.json` records, for each canonicalized group and each excluded code, the number of
  annotations and windows affected."* The actual `class_list.json` is a flat JSON array of
  `{index, code, species}` with no `code_crosswalk` key at all. `split_manifest.json`'s
  `taxonomy_crosswalk` object covers the placeholder list and the two duplicate-merge groups (content
  verified correct, item 7/8 above) but does **not** include per-code/per-group annotation- or
  window-affected counts, and does not cover the other 163 non-placeholder, non-duplicate codes'
  trivial identity mapping. A reviewer following the design doc's stated schema would look in the
  wrong file and, even there, could not get the "number of annotations and windows affected" the doc
  promises without re-deriving it (as this report did).
- **`species_distribution.csv` is missing project-distribution columns.** The design's "Split
  construction" step 4 and "Files" section both state project counts are written to
  `species_distribution.csv` "for transparency" (report-only). The actual file contains only
  `label, global_count, global_frac, {split}_count, {split}_frac, {split}_frac_deviation` -- no
  project column. Project counts do exist, correctly computed and correctly excluded from the
  optimization objective (confirmed by code read: no `project` term appears anywhere in
  `assign_groups()`'s cost function), but only in `split_manifest.json`'s
  `project_distribution_canonical` / `project_distribution_augmented_train` fields, not in
  `species_distribution.csv` as documented.
- **Module docstring overstates the project tie-break.** `prepare_species_splits.py`'s top docstring
  says project distribution "is tie-broken with a token weight only" in the optimizer; the actual
  `assign_groups()`/`split_cost()` code contains no `project`-related term anywhere -- project has
  **zero** influence on assignment, not a token-weighted tie-break. This is stricter than promised
  (better for the "never a stratification target" guarantee this review's item 14 and the design's own
  intent require), but the docstring itself is inaccurate and should be corrected to avoid a future
  reader assuming tie-break code exists that does not.

None of these three items affects `sound_id` disjointness, split sizes, species presence, the support
gate, label-state correctness, augmentation scope, no-bird prevalence, or reproducibility of the actual
split content -- all of which were independently verified above to match the approved design exactly.

---

## Test suite

`tests/test_prepare_species_splits.py` (51 tests) covers taxonomy placeholder/duplicate detection,
geometry-based annotation matching, window classification (no-bird / touching-boundary / clean /
mixed-excluded), retained-vs-rare target restriction, the vocabulary fixed-point gate (including a
cascade case and a mixed-window-keeps-species case), row-validation gates (leakage, canonicalicity,
augmentation-source, species-presence), the hard-reservation mechanism, and cross-run determinism of
vocabulary/assignment/row membership. Ran clean: **51 passed, 0 failed** against the current code.
This is a reasonable, if consolidated-into-one-file, realization of the design's proposed 12-test list
(`test_split_group_integrity.py` etc.) -- content coverage overlaps closely; file-per-test granularity
was not followed, which the design explicitly allows ("exact CLI flag names ... may be finalized
during implementation").

---

## Structured summary

### Pipeline Correctness
- Alignment: verified -- `sound_id` grouping is respected end-to-end; augmented train rows never
  originate from a non-train sound_id; `window_id` uniquely and consistently identifies rows within
  and across splits.
- Type safety: `is_canonical` is a clean 0/1 int; `target_vector` is a valid JSON int list matching
  vocabulary length in every row sampled and in aggregate zero/nonzero checks; no coercion issues found.
- Value ranges: no NaN/negative/out-of-range values found in any checked column; `target_codes` never
  contains an out-of-vocabulary, placeholder, or duplicate-raw code.

### Leakage Check
- Split integrity: clean -- 0 `sound_id` overlaps in any pairwise split comparison, independently
  re-derived from row contents (not just the assignment dict).
- Statistics leakage: N/A at this stage (no normalization/vocabulary fitting exists yet --
  `extract_perch_embeddings.py`/`train_perch_logreg.py` are design-only, not yet implemented).
- Augmentation leakage: clean -- every augmented train row's `sound_id` is a member of
  `canonical_train_split.csv`'s sound_id set; val/test contain zero non-canonical rows.

### Schema Validation
- Expected fields: present in all four split CSVs and `class_list.json`; `split_manifest.json`
  contains most, but not all, of the auxiliary information the design specifies (see "Deviations").
- Types: consistent across all rows checked.
- Completeness: all 68 vocabulary species have >=1 positive window in each of train/val/test; all 563
  sound_ids are assigned to exactly one split; canonical population accounts for the full 42,386-window
  modeled set.

### Edge Cases
- Mixed retained+rare window (`window_id=108`): confirmed correctly restricted to the retained subset,
  not excluded.
- Rare-only window (`window_id=150`): confirmed correctly excluded from every split.
- Duplicate-code species right at the support boundary (`ATAPIL`): confirmed the canonicalization is
  load-bearing (4 vs. 180 raw annotations merged), not a no-op.
- Manifest run-to-run byte reproducibility: confirmed non-deterministic ordering in one diagnostic
  sub-object due to `PYTHONHASHSEED`; confirmed semantically identical; confirmed fixable by pinning
  the hash seed.

### Recommendations
- [ ] WARNING: Add the design-specified `code_crosswalk` object to `class_list.json` (or update the
  design doc if `split_manifest.json`'s `taxonomy_crosswalk` is intended to be the permanent home
  instead), and include per-group/per-code annotation- and window-affected counts so the crosswalk is
  audit-able without re-deriving it by hand.
- [ ] WARNING: Add project-count columns to `species_distribution.csv` (or update the design doc to
  point at `split_manifest.json`'s `project_distribution_*` fields instead) so the artifact a reviewer
  is told to check actually contains what the design promises.
- [ ] WARNING: Make `split_manifest.json` serialization deterministic across separate process runs --
  either sort the `support` dict in `vocabulary_fixed_point_history` before writing, or pin
  `PYTHONHASHSEED` in the run instructions/CI, so the manifest is byte-reproducible like every other
  artifact already is.
- [ ] SUGGESTION: Correct the `prepare_species_splits.py` module docstring's claim that project
  distribution is "tie-broken with a token weight" in the optimizer -- the code gives it zero
  influence, which is fine, but the docstring should describe what the code actually does.
- [ ] SUGGESTION: Extend `test_determinism_across_reruns_with_same_seed` (or add a new test) to
  compare the full serialized `split_manifest.json` text (with `PYTHONHASHSEED` pinned, or after
  sorting the diagnostic dicts) across two runs, so the manifest's own reproducibility is a checked
  invariant, not just the split content's.

---

STATUS: PASS
