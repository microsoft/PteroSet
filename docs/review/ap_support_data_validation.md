# Data Validation Report: Per-Species AP/Support Correlation Analysis (inputs & proposed joins)

**Scope**: no correlation-analysis script exists yet in the repo (verified via `git status` and
repo-wide search — `paper/` has zero hits for `test_pos|species_ap`). This review independently
inspects the candidate input artifacts a per-species "AP vs. support" analysis would join —
`checkpoints/perch/species_v1/species_ap.csv`, `data/splits_species_v1/{train,val,test,canonical_train}_split.csv`
(`target_codes`/`target_vector`), `data/splits_species_v1/class_list.json`,
`data/splits_species_v1/species_distribution.csv`, `data/splits_species_v1/split_manifest.json`, and
`checkpoints/perch/species_v1/test_predictions.npz` — and validates them (and the joins between
them) against the correctness/leakage/pseudoreplication checklist below, so the analysis can be
written correctly the first time. Read-only; nothing edited.

---

## 1. Representation metrics — what "support" can legitimately mean here

At least **four different, non-interchangeable "support" numbers** exist across these files for the
same species, and they must not be treated as equivalent or silently substituted for one another:

| Source | Column | What it counts | Example (AMAAMA) |
|---|---|---|---|
| `species_ap.csv` | `train_pos` | **Augmented** train windows (`train_split.csv`, 95,536 rows = 29,513 canonical + 66,023 overlap-only) | 79 |
| `species_distribution.csv` | `train_count` | **Canonical-only** train windows (5 s tiling, no overlap crops) | 11 |
| `split_manifest.json` → `vocabulary_fixed_point_history[0]["support"]` | — | Distinct **`sound_id`** (recording) count, global (train+val+test combined), canonical windows only | 12 |
| Recomputed from `canonical_train_split.csv` | — | Distinct **`sound_id`** count, **train split only** | 12 |

Independently recomputed (`canonical_train_split.csv`/`train_split.csv`, grouped by `target_vector`
column via `class_list.json` index): `AMAAMA` has **79** augmented train windows but only **11**
canonical train windows and **12** distinct train recordings. All three numbers are internally
consistent (each recomputed exactly from its own source file) but **a 7x spread for the same species
depending purely on which file's "support" you pull**, with no shared column name to warn you.
**A join that pulls `train_pos` from `species_ap.csv` while a figure caption or Methods section cites
`species_distribution.csv`'s `train_count` (or vice versa) will silently report the wrong number for
the same claimed quantity.** State explicitly, in code and in any figure/table caption, which of
these four is being used.

**Recommendation for the analysis**: use `species_ap.csv`'s own `test_pos`/`train_pos` (matches what
the model actually saw/was scored on) as the primary "support" axis since it is scoped to the same
artifact as `ap`, but report the recording-level (`sound_id`) count alongside it (see §2) rather than
silently equating "more windows" with "more independent evidence."

---

## 2. Pseudoreplication trap — window counts are not independent samples

`train_split.csv` is intentionally augmented with every overlapping (non-canonical) window from a
train-assigned `sound_id` (documented in `prepare_species_splits.py`'s module docstring and
`split_manifest.json.augmented_train_extra_windows=66023`); `val_split.csv`/`test_split.csv` are
canonical-only (this is correct design for training, and is *not* a leakage bug — see §3). But it
means **`train_pos`/`test_pos` window counts conflate two different things**: (a) genuinely
independent recording/detection events, and (b) how many overlapping 4 s-stride crops a species'
typical call duration happens to generate — which varies **by species**, not just by "how much data
exists."

Independently recomputed windows-per-recording ratios (68 species, `canonical_train_split.csv` vs.
distinct train `sound_id`s):

- Canonical windows/`sound_id`: ranges **0.92 (AMAAMA)** to **8.42 (ATAPIL)**.
- Augmented windows/`sound_id`: ranges **3.86 (VOLJAC)** to **25.17 (ATAPIL)**.
- Augmented/canonical ratio itself ranges **2.45x–7.18x** across species (mean 3.71, stdev 0.94) —
  i.e. the *same* canonical call-event count inflates to a very different number of augmented
  training windows depending on species-specific call duration/pattern.

Concretely: `ATAPIL` (302 augmented `train_pos`) and `LEPVER` (265 augmented `train_pos`) look like
similar "support" by raw window count, but `ATAPIL` has only 12 distinct train recordings vs.
`LEPVER`'s 13 — nearly identical true event count despite a >30-window difference, driven entirely by
call-duration-driven window multiplicity. **Using raw `train_pos`/`test_pos` as an x-axis for an
"AP vs. amount-of-data" correlation risks measuring "how long/repetitive this species' calls are"
as much as "how much independent evidence the model had."** This is the same trap already flagged
during design review (`docs/review/doc-fix/round_01/data_validator_report.md`, §4: "window-count
support... conflates correlated, non-independent samples of the same acoustic event with genuinely
new evidence") — it applies with equal force to *this* downstream analysis and was not fully closed
by the split design, only by the *vocabulary gate* (which correctly uses `sound_id` counts, not
window counts, per `vocabulary_gate_rationale`).

**Recommendation**: report/plot support in `sound_id` (recording) units as the primary or a
required secondary axis, not raw window counts alone; if window counts are used, disclose the
per-species window/recording ratio alongside so reviewers can judge whether a correlation is an
artifact of call-duration confounding.

---

## 3. Split integrity — independently re-verified (clean)

- `sound_id` disjointness across train/val/test: independently recomputed set-intersections from the
  three split CSVs directly — **0 overlap** in all three pairwise comparisons (train∩val=0,
  train∩test=0, val∩test=0; train=385, val=87, test=91 distinct `sound_id`s). Matches
  `split_manifest.json.sound_id_disjointness_verified=True` and `train_perch_logreg.py`'s own
  `validate_splits_disjoint` gate.
- `test_split.csv`/`val_split.csv` are 100% canonical (non-overlapping 5 s tiles); only
  `train_split.csv` carries `is_canonical=0` rows. This is the correct, and only, place augmentation
  may appear — confirmed by direct inspection, not just by trusting the manifest.
- `test_predictions.npz` row order, `window_id`, and `sound_id` arrays are **identical, row-for-row**,
  to `test_split.csv` (independently checked by direct list comparison, not just shape/dtype).
- No leakage risk from the vocabulary/support gate: it is computed on `sound_id` counts across the
  *canonical* window set before any split assignment (`min_sound_ids=7`, `vocabulary_gate_rationale`),
  so it cannot depend on train-augmentation window counts.

---

## 4. Label/support alignment — independently re-derived, exact match

- **`target_codes` vs. `target_vector` consistency**: for all 6,474 test rows, the semicolon-joined
  `target_codes` string was decoded and compared against the on-bits of the 68-length `target_vector`
  (indexed via `class_list.json` order) — **0 mismatches**.
- **`class_list.json` index/code/species ordering** matches `species_ap.csv`'s `index`/`code`/`species`
  columns and `species_diagnostics.csv`'s `code` column row-for-row — **0 mismatches** across all 68
  entries. This is the join key everything above ultimately keys on; it is sound.
- **`test_pos` recomputation**: recomputed per-species positive-window counts directly from
  `test_split.csv`'s `target_vector` (via the same `class_list.json` index mapping) and compared to
  `species_ap.csv.test_pos` — **exact match**, e.g. AKLMEL=7, CRYCIN=57, QUEPUR=1.
- **AP recomputation**: recomputed `average_precision_score(target_vector[:,i], probabilities[:,i])`
  directly from `test_predictions.npz` for all 68 species and compared to `species_ap.csv.ap` —
  **max abs diff = 0.0** (exact, not merely close).
- **Eligibility flags**: all 68 rows have `trainable=val_evaluable=test_evaluable=test_evaluable_effective=True`
  and `ap` is finite for all 68 (no NaN) at this snapshot — so a naive join with no eligibility
  filter happens to be safe *today*, but the analysis code must still explicitly filter on
  `test_evaluable_effective` (not just `test_evaluable`) before correlating, since a future rerun
  could produce a non-finite coefficient for a thin-support species (`test_evaluable=True` but
  `test_evaluable_effective=False`), which would otherwise silently enter the correlation as
  `ap=NaN` (Pearson/Spearman implementations differ in whether they error, warn, or silently drop —
  don't rely on default behavior).
- **Redundant column trap**: `species_ap.csv.test_prevalence` and `.prevalence_baseline_ap` are
  byte-identical for all 68 rows (both are just `mean(target_vector[:,i])`) — harmless today, but if
  a join brings in both under different aliases it will look like two independent variables when it
  is one, inflating apparent evidence in any multi-variable model.

## 5. Join hazard — `species_distribution.csv`'s `no_bird` pseudo-row

`species_distribution.csv`'s `label` column is **not** the same key space as `species_ap.csv`'s
`code`: its first row is `label="no_bird"` (a non-species aggregate row), not present in
`class_list.json`/`species_ap.csv`'s 68-code vocabulary. A join on `label`/`code` equality without
first dropping `label=="no_bird"` will not error (no matching key -> silently produces a null/NaN row
or, in an outer join, a phantom 69th "species" with no AP data) — filter this row out explicitly
before any join, don't rely on the join failing loudly.

## 6. Taxonomy crosswalk hazard (only relevant if joining against raw/legacy codes)

`species.csv` (raw, 168 codes) and any older artifact still using pre-canonicalization codes must be
passed through `split_manifest.json.taxonomy_crosswalk` before joining to `species_ap.csv`'s 68
canonical codes: `ATRPIL`→`ATAPIL` and `RHATUC`→`RAMTUC` are merged duplicates, and
`PICIDA_1, PSITTA, PSITTACIDAE, PSITTACIFORMES, RHACAR, TYRANN_SP1` are placeholder codes with **no**
row in `species_ap.csv` at all (dropped, not merged). A naive code-string join against `species.csv`
without applying this crosswalk will silently miss 2 species' full counts (undercounting `ATAPIL`/
`RAMTUC` if the raw-code rows aren't remapped first) and produce unmatched rows for the 6 placeholders.

## 7. Statistical hazard for the correlation itself (not a data-integrity bug, but will mislead if unaddressed)

- **n=68** species total; **24/68 have `test_pos<5`, 4/68 have `test_pos==1`** (disclosed in
  `macro_ap_summary.csv`'s `thin_support_ge5`/`ge10` fields — confirms 44/68 and 25/68 respectively,
  consistent with 68−24=44 and 68−43=25 from the complementary disclosure in
  `docs/review/phase3_data_validation.md`). AP is a high-variance point estimate at `test_pos∈{1,2}`
  (e.g. `QUEPUR` has `ap=1.0` on a single test positive — entirely rank-order-of-one, not a stable
  skill estimate). A correlation fit that includes these points with equal weight to a `test_pos=115`
  species will have its slope/CI dominated by estimator noise at the thin-support end, not by a real
  representation-quality signal.
- Quick independent sanity check (Pearson vs. Spearman on the full 68, `ap` vs. raw `test_pos`):
  Pearson r=0.368 (p=0.0020) vs. Spearman ρ=0.521 (p=5.1e-6) — the sizable Pearson/Spearman gap
  itself is evidence of a nonlinear/heavy-tailed relationship (consistent with the AP-estimator-noise
  and window-count-confound points above); a log(test_pos) transform raises Pearson r to 0.481.
  **Use Spearman as the primary statistic (or log-transform support), not raw Pearson**, and report
  sensitivity to dropping the `test_pos<5` subset (macro-AP already computed for `test_pos>=5`/`>=10`
  subsets in `macro_ap_summary.csv` — reuse those instead of re-deriving ad hoc).
- `ap` should be compared against `delta = ap - prevalence_baseline_ap` (already a column in
  `species_ap.csv`) as a robustness check: `prevalence_baseline_ap` is itself mechanically a function
  of `test_prevalence` (≈`test_pos`/6474), so part of any raw `ap`-vs-`test_pos` correlation is a
  built-in floor effect of the AP metric at low prevalence, not new information about the model.

---

## Summary

### Pipeline Correctness
- Alignment: verified — `class_list.json` index/code order, `species_ap.csv`, `species_diagnostics.csv`,
  and `test_predictions.npz` row order all independently cross-checked, 0 mismatches.
- Type safety: verified via `train_perch_logreg.py`'s own schema gate and direct dtype inspection
  (`float32` embeddings, `float64` probabilities, `uint8` targets, `int64` identity fields).
- Value ranges: `ap`/`test_prevalence`/`prevalence_baseline_ap` all finite and in [0,1]; no NaN `ap`
  at this snapshot (all 68 `test_evaluable_effective=True`).

### Leakage Check
- Split integrity: clean — independently recomputed 0 `sound_id` overlap across train/val/test.
- Statistics/augmentation leakage: clean — augmentation confined to `train_split.csv`
  (`is_canonical=0` rows only there); val/test canonical-only, confirmed by direct inspection.

### Schema Validation
- Expected fields: present in all inspected files.
- Types: correct; `target_codes`/`target_vector` cross-consistent (0 mismatches, 6,474/6,474 rows checked).
- Completeness: `species_ap.csv` covers all 68 `class_list.json` codes, no duplicates, no gaps.

### Edge Cases
- Thin test support (24/68 `test_pos<5`, 4/68 `test_pos==1`): present and disclosed upstream, but
  **not yet accounted for** in any proposed correlation methodology — flagged above (§7).
- Redundant/duplicate-looking columns (`test_prevalence`==`prevalence_baseline_ap`): identified, not
  a defect but a join-confusion risk.
- Pseudoreplicated "support" (window counts vs. recording counts): quantified (§1–2); not an error in
  the existing artifacts (each file is internally correct and serves its own documented purpose) but
  a **methodological trap if window counts are naively used as the correlation's independent
  variable**.

### Recommendations
- [ ] CRITICAL: Before joining/correlating, explicitly decide and document which "support" definition
  is used (`species_ap.csv.train_pos`/`test_pos` [augmented/canonical window counts] vs. distinct
  `sound_id` counts) — do not let a figure or table mix numbers from `species_ap.csv` and
  `species_distribution.csv` under an unlabeled shared name like "train count."
- [ ] CRITICAL: Filter on `test_evaluable_effective` (not just `test_evaluable`) before correlating,
  and assert `ap`/support arrays contain no NaN post-filter rather than relying on library defaults
  to silently drop or error.
- [ ] CRITICAL: Drop `species_distribution.csv`'s `label=="no_bird"` row before any join keyed on
  species label/code; it is not a species and has no counterpart in `species_ap.csv`.
- [ ] WARNING: Report support in `sound_id` (recording) units alongside, or instead of, raw window
  counts; disclose the per-species window/recording ratio if window counts are used at all, given the
  demonstrated 2.4x–7.2x cross-species variation in that ratio.
- [ ] WARNING: Use Spearman correlation (or a log-support transform) as the primary statistic, given
  the empirically confirmed Pearson/Spearman divergence (0.368 vs. 0.521) on the current data; report
  results with and without the 24 `test_pos<5` species (reuse `macro_ap_summary.csv`'s existing
  `thin_support_ge5`/`ge10` subsets rather than re-deriving new thresholds ad hoc).
- [ ] WARNING: If joining against `species.csv` or any pre-canonicalization artifact, apply
  `split_manifest.json.taxonomy_crosswalk` first (`ATRPIL`→`ATAPIL`, `RHATUC`→`RAMTUC` merges; 6
  placeholder codes have no `species_ap.csv` counterpart at all).
- [ ] SUGGESTION: Also correlate `delta` (`ap - prevalence_baseline_ap`) against support, not only raw
  `ap`, to separate "genuine skill above chance" from the AP metric's mechanical floor-effect at low
  prevalence.

---

STATUS: PASS

Every artifact inspected — `species_ap.csv`, the four `splits_species_v1` split CSVs
(`target_codes`/`target_vector`), `class_list.json`, `species_distribution.csv`,
`split_manifest.json`, and `test_predictions.npz` — is internally consistent and correctly joined on
the `code`/`index` key today: independent recomputation of `test_pos` and per-species `ap` from raw
predictions and split CSVs reproduces `species_ap.csv` exactly (max abs AP diff = 0.0), split
disjointness and augmentation scoping are independently re-verified clean, and no data-integrity
defect blocks the proposed per-species AP/support correlation analysis. However, the analysis has
**not yet been implemented** in this repo, and the artifacts contain three concrete traps that a naive
implementation will fall into: (1) at least four differently-scoped "support" numbers exist across
these files for the same species with no shared, self-documenting column name, and window-count
support is empirically confounded with per-species call-duration/repetition (2.4x–7.2x cross-species
variation in windows-per-recording); (2) a `no_bird` pseudo-row and un-crosswalked legacy species
codes are silent join hazards; (3) thin per-species test support (24/68 `test_pos<5`) makes raw
Pearson correlation on `ap` unreliable, empirically confirmed by a Pearson/Spearman divergence on
this exact data. PASS is conditioned on the CRITICAL items above being addressed in the analysis
implementation before it is written, not on any change to the existing artifacts.
