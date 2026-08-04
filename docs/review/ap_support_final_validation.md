# Data Validation Report: species-linear-probe-v1 AP/Support Analysis (final, generated outputs)

**Scope**: independent re-validation of the *already-generated* outputs in
`docs/implementation/species-linear-probe-v1/` (`ap_support_analysis.md` and the six
`analysis/*.csv` artifacts, produced by `analyze_species_ap.py`), against the underlying
source-of-truth artifacts: `data/splits_species_v1/{train,canonical_train,val,test}_split.csv`,
`data/splits_species_v1/class_list.json`, `checkpoints/perch/species_v1/species_ap.csv`, and
`checkpoints/perch/species_v1/test_predictions.npz`. This supersedes/complements the two earlier
reviews in `docs/review/` (`ap_support_data_validation.md`, PASS, and
`ap_support_experiment_review.md`, FAIL), which audited the *inputs* to this analysis **before it
was implemented**. All of that FAIL review's CRITICAL items (per-species bootstrap CIs, use of
canonical not augmented train support as the primary axis, prevalence-normalized `ap_lift`) are
verified below to have been addressed in the implementation that now exists. Nothing was edited;
all numbers below are from independent re-derivation, not from re-running `analyze_species_ap.py`
and trusting its own output (a separate exact-determinism re-run was also performed and is reported
in §7, but is treated as a secondary, not primary, check).

**Method**: every check below was performed with a from-scratch Python re-implementation reading
directly from the raw CSV/JSON/NPZ source files (not by importing or reusing code from
`analyze_species_ap.py`), and, where randomness is involved, with an independently chosen RNG seed
and in several cases a different number of bootstrap draws or algorithmic path than the original
script, specifically so agreement is not an artifact of sharing the same code or seed.

---

## 1. AP recomputation (source of truth: `test_predictions.npz`)

- Row order of `test_predictions.npz` (`window_id`, `sound_id`) matches `test_split.csv` row-for-row
  (`np.array_equal`, both arrays, all 6,474 rows) — the join underlying every downstream computation
  is sound.
- Probabilities are finite everywhere (`np.isfinite(probabilities).all() == True`), shape
  `(6474, 68)`, matching `target_vector`.
- Independently recomputed `average_precision_score(target_vector[:, i], probabilities[:, i])` for
  all 68 species and compared to:
  - `checkpoints/perch/species_v1/species_ap.csv.ap`: max abs diff = **9.5e-17** (floating-point
    noise, effectively exact).
  - `docs/implementation/.../analysis/ap_support_per_species.csv.ap`: max abs diff = **9.5e-17**.
- `ap_lift = ap - test_prevalence` independently recomputed: max abs diff vs. reported = **1.3e-16**.
  `test_prevalence` independently recomputed as `test_pos/6474`: max abs diff = **9.7e-17**.

**Result: exact match, no discrepancy.**

## 2. Support counts (source of truth: split CSVs, not `species_distribution.csv`)

Independently parsed `target_codes` (semicolon-delimited) from each of the four split CSVs
(`train_split.csv`, `canonical_train_split.csv`, `val_split.csv`, `test_split.csv`) and recomputed,
per species, window counts and distinct-`sound_id` counts for all seven support columns
(`train_aug_windows`, `train_canonical_windows`, `train_sound_ids`, `val_windows`, `val_sound_ids`,
`test_windows`, `test_sound_ids`):

| column | independent recompute matches `ap_support_per_species.csv` |
|---|---|
| train_aug_windows | exact (`np.array_equal` True) |
| train_canonical_windows | exact |
| train_sound_ids | exact |
| val_windows | exact |
| val_sound_ids | exact |
| test_windows | exact |
| test_sound_ids | exact |

Cross-checked `species_ap.csv`'s own `train_pos`/`val_pos`/`test_pos` against the same independent
recompute — exact match on all three. This confirms the analysis correctly uses
`canonical_train_split.csv` (not the augmented `train_split.csv`) as its primary train-support axis,
closing the CRITICAL gap flagged in the earlier `ap_support_experiment_review.md`.

**Result: exact match, no discrepancy.**

## 3. Point-estimate correlations (Spearman rho/p, Pearson on log1p(support))

Independently recomputed `spearmanr(x, y)` and `pearsonr(log1p(x), y)` for both responses (`ap`,
`ap_lift`) against all 7 support columns (14 predictor/response pairs) directly from
`ap_support_per_species.csv`'s own columns (i.e., verifying the *statistics*, not re-trusting the
*support numbers*, which were separately verified in §2):

- All 14 `spearman_rho` values match to **1e-9**.
- All 14 `pearson_log1p_r` values match to **1e-9**.
- Benjamini-Hochberg q-values independently recomputed with `scipy.stats.false_discovery_control`
  (a different implementation than the analysis script's hand-rolled BH loop): max abs diff vs.
  reported `spearman_q_bh` = **6e-17**.

**Result: exact match, no discrepancy.**

## 4. Partial rank correlations

Independently re-derived partial Spearman correlation (rank-transform predictor/response/controls,
OLS-residualize on an intercept + ranked controls, Pearson correlation of residuals) for all 4
reported predictors (`train_aug_windows`, `train_canonical_windows`, `train_sound_ids`,
`val_windows`), both control sets (`{test_windows}` and, where applicable,
`{test_windows, val_windows}`):

| predictor | `controls_test_rho` match | `controls_val_test_rho` match |
|---|---|---|
| train_aug_windows | 0.2454 = 0.2454 | 0.2298 = 0.2298 |
| train_canonical_windows | 0.3027 = 0.3027 | 0.2954 = 0.2954 |
| train_sound_ids | 0.0024 = 0.0024 | -0.0261 = -0.0261 |
| val_windows | 0.0890 = 0.0890 | 0.0890 = 0.0890 |

The headline prose number "0.303 (p=0.0121)" for canonical train positives controlling for test
support reproduces exactly from `ap_support_partial_correlations.csv` (`0.30267...`, `p=0.01211...`).

**Result: exact match, no discrepancy.**

## 5. Bin tables (`test_support_bin`, `test_sound_bin`)

Independently re-binned `test_windows` into `["1-2","3-4","5-9","10-19","20+"]` and `test_sound_ids`
into `["1","2-3","4-9","10+"]` (identical bin edges as stated in the report's own tables, applied
independently to the per-species table) and recomputed `n_species`/`mean_ap`/`median_ap`/`std_ap`/
`min_ap`/`max_ap` per bin:

- Both tables match the reported CSVs to `1e-9` on all six numeric columns, for every bin.
- `n_species` sums to 68 in both tables (no species dropped or double-counted by binning).
- Independently confirmed **10** species have `test_sound_ids == 1` — matches the report's prose
  ("Ten species have positives from only one test audio file").

**Result: exact match, no discrepancy.**

## 6. Sensitivity table (support-restricted subsets)

Independently recomputed Spearman rho for 4 predictors × 4 subsets (`all` n=68, `test_pos_ge5` n=44,
`test_pos_ge10` n=25, `test_positive_sound_ids_ge2` n=58) directly from the per-species table — all
16 rho values and all 4 subset sizes match exactly. In particular the headline sensitivity claim in
the report ("rho changes from 0.573 (68 species) to 0.433 for `test_pos>=5` and 0.203 for
`test_pos>=10`") reproduces exactly (0.5733 → 0.4330 → 0.2028).

Independently recomputed the underlying thin-support counts directly from `test_windows`:
`test_pos<5`: **24/68**; `test_pos<10`: **43/68** — consistent with the counts previously verified
in `docs/review/ap_support_data_validation.md` (§7) from `species_ap.csv`/`macro_ap_summary.csv`,
confirming no drift between the earlier input-validation snapshot and the final analysis snapshot.

**Result: exact match, no discrepancy.**

## 7. Cluster-bootstrap AP confidence intervals and estimability

Because this computation is randomized, it was cross-checked with a **deliberately different**
seed (777 vs. the script's seeded-from-42 stream) and independently written resampling code
(grouping row indices by `sound_id`, drawing `n_boot=1000` group-index bootstrap replicates,
skipping replicates where the resampled label column is constant, taking the 2.5/97.5 percentiles):

- **Estimability flag**: independently recomputed "≥2 distinct positive test `sound_id`s" per
  species and compared to the reported `ap_ci_estimable` column — **exact match for all 68 species**
  (10 non-estimable, all and only the species with `test_sound_ids==1`; 0 species with
  `test_sound_ids>=2` unexpectedly marked non-estimable; 0 species with `test_sound_ids==1`
  unexpectedly marked estimable).
- **CI sanity**: for all 58 estimable species, the reported `[ap_ci_low, ap_ci_high]` brackets the
  reported point-estimate `ap` — 0 violations.
- **CI magnitude cross-check**: for the three primary predictors' companion Spearman bootstrap CIs,
  an independently seeded (seed=999), independently implemented bootstrap (`n_boot=4000`, resampling
  species directly and calling `scipy.stats.spearmanr` per draw, rather than reimplementing rank
  correlation as the script does) produced intervals overlapping the reported ones in all 3 cases and
  containing the reported point estimate, e.g. `train_canonical_windows`: independent
  `[0.378, 0.725]` vs. reported `[0.376, 0.724]`; `test_windows`: independent `[0.289, 0.717]` vs.
  reported `[0.295, 0.714]`. No sign of a systematic bias or an implausibly narrow/wide interval.
  (Exact equality is neither expected nor achieved here, by design, since seed/draw-count differ —
  this is a plausibility/consistency check, not an exactness check.)

**Result: no discrepancy; bootstrap methodology and estimability gating behave as documented.**

## 8. Top/bottom-10 tables, prose statements, and cross-references

- Independently recomputed `nlargest(10, "ap")` / `nsmallest(10, "ap")` from the per-species table
  and parsed the corresponding markdown tables out of `ap_support_analysis.md` — codes and AP values
  (rounded to 3 dp, as rendered) match exactly and in the same order for both top and bottom tables.
  No tie sits at the rank-10/rank-11 boundary in either direction (rank 10 vs. 11 differ by
  ≈0.028 AP descending, ≈0.017 ascending), so the top/bottom-10 selection is unambiguous.
- Spot-checked derived prose claims against the CSVs:
  - "Training and test supports are themselves highly correlated by the stratified split": Spearman
    `train_canonical_windows` vs. `test_windows` = **0.807**; vs. `val_windows` = **0.790** — supports
    the "highly correlated" characterization.
  - "AP lift above prevalence gives nearly the same correlations": `ap` vs. `ap_lift` Spearman rho
    differ by ≤0.005 for all three primary predictors (0.573→0.570, 0.448→0.443, 0.521→0.517).
  - "Canonical train support correlates slightly more strongly with AP than augmented train support":
    0.573 (canonical) vs. 0.544 (augmented) — confirmed, and in the claimed direction.
  - "The distinct-train-audio association largely disappears after this control": uncontrolled
    `train_sound_ids` vs. `ap` rho = 0.387, drops to 0.0024 controlling for test support — confirmed.
  - "the validation set only selected one global C; it did not tune a separate classifier per
    species": confirmed by inspection of `train_perch_logreg.py` — a single `C` is chosen via a grid
    sweep maximizing macro-AP on the (shared) validation set (`sweep_c_grid`/`select_best_c`), then
    one `OneVsRestClassifier(LogisticRegression(C=C*))` is fit; no per-species `C` search exists.
- All 68 rows in `species_ap.csv` have `trainable=val_evaluable=test_evaluable=test_evaluable_effective=True`
  and finite `ap` at this snapshot — the eligibility-filter CRITICAL item from the earlier
  `ap_support_data_validation.md` review is a no-op today (nothing to filter), consistent with that
  review's own finding; no species were silently dropped or NaN-suppressed in this analysis.
- Split `sound_id` disjointness independently re-verified from the four split CSVs directly:
  train=385, val=87, test=91 distinct `sound_id`s; 0 pairwise overlap — matches the counts previously
  reported in `docs/review/ap_support_data_validation.md` §3, confirming no drift.

**Result: no discrepancy.**

## 9. Determinism (secondary check)

Re-ran `analyze_species_ap.py` end-to-end (default `--seed 42`) into a scratch directory and
byte-compared every one of the 6 CSVs and the markdown report against the committed outputs: all 7
files are byte-identical. This confirms reproducibility given the seed but, by itself, cannot rule
out a bug shared between the script and its own report (which is why §1-8 above are done from
scratch, not by trusting this re-run).

---

## Summary

### Pipeline Correctness
- Alignment: verified — `test_predictions.npz` row order matches `test_split.csv` exactly;
  `class_list.json`/`species_ap.csv` code ordering consistent throughout.
- Type safety: probabilities finite float64, targets uint8, no coercion issues observed.
- Value ranges: AP, `ap_lift`, `test_prevalence` all finite and consistent with independent
  recomputation to floating-point noise (≤1.3e-16).

### Leakage Check
- Split integrity: clean — 0 `sound_id` overlap across train/val/test (independently re-verified).
- Statistics/support leakage: clean — the primary train-support axis correctly uses
  `canonical_train_split.csv`, not the overlap-augmented `train_split.csv`; augmented and canonical
  counts are both reported, addressing the earlier FAIL review's core critique.
- Augmentation leakage: N/A — this is a downstream analysis, not a training pipeline; augmentation
  scoping to train-only was independently re-verified as part of the source-of-truth split check.

### Schema Validation
- Expected fields: all present in all 6 analysis CSVs and cross-checked against source files.
- Types/joins: `code`/`index` join verified consistent across `class_list.json`, `species_ap.csv`,
  and the analysis table; no phantom or dropped species (n=68 throughout, sums to 68 in every
  bin/subset table).
- Completeness: no missing rows, no unexpected NaNs outside the documented (and correctly gated)
  non-estimable bootstrap CIs.

### Edge Cases
- Thin test support (24/68 `test_pos<5`, 10/68 with only 1 positive test recording): present,
  correctly gated out of bootstrap-CI estimability, and explicitly discussed via the sensitivity
  table rather than hidden.
- Tie-at-cutoff for top/bottom-10 tables: checked, none present.
- Redundant `test_prevalence`/`prevalence_baseline_ap` columns (flagged in the earlier input-only
  review): not used redundantly here — the analysis uses `test_prevalence` once, to build `ap_lift`.

### Recommendations
- [ ] SUGGESTION: The Benjamini-Hochberg correction is applied jointly across all 14 (7 predictors ×
  2 responses: `ap`, `ap_lift`) tests in one pooled correction. This is defensible (all are reported
  in the same family of tests) but is a design choice worth stating explicitly in the report text,
  since a reader could otherwise assume `spearman_q_bh` was corrected only within the 3-predictor
  "Main Result" table shown.
- [ ] SUGGESTION: Consider adding one sentence noting that the reported bootstrap CIs (Spearman-rho
  CI over species, and per-species cluster-bootstrap AP CI) both resample only within-analysis units
  (species, or test-window clusters) and therefore — as already stated in Limitation #2 — do not
  capture re-split/re-training variance; this is already disclosed but could be tied more explicitly
  to the specific CI columns readers will see in the CSVs.

No CRITICAL or WARNING items were found. Every numeric artifact in
`docs/implementation/species-linear-probe-v1/` (the per-species table, both correlation tables, the
partial-correlation table, both bin tables, the sensitivity table, the bootstrap CIs, the top/bottom
tables, and every quoted number in the report prose) was independently reproduced from the raw split
CSVs and `test_predictions.npz`, using separately written code and, for randomized components,
deliberately different seeds/algorithms — with zero discrepancies beyond floating-point noise
(≤1.3e-16). The two earlier reviews' CRITICAL items (per-species bootstrap CIs; canonical vs.
augmented train support as the representation axis; prevalence-normalized `ap_lift`) are all
resolved in the artifacts validated here.

STATUS: PASS
