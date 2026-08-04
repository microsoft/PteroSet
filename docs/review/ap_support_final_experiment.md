# Experiment Integrity Review — `ap_support_analysis.md` (Species Linear-Probe v1)

**Scope**: independent audit of the scientific claims in
`docs/implementation/species-linear-probe-v1/ap_support_analysis.md` against the generated tables it
cites (`analysis/*.csv`) and the script that produces both (`analyze_species_ap.py`), plus its
downstream framing in `results.md`. Focus areas per request: causal language, prevalence/confounding,
multiple comparisons, thin-support/cluster-bootstrap interpretation, sensitivity subsets, and
technical-validation implications. Read-only — nothing edited.

**Prior context**: this is the second generation of this analysis. The first attempt was audited in
`docs/review/ap_support_experiment_review.md` (**STATUS: FAIL** — no per-species uncertainty, and
`train_pos` conflated augmented/canonical window counts) and `docs/review/ap_support_data_validation.md`
(**STATUS: PASS**, conditioned on fixing those same traps before implementation). The current
`ap_support_analysis.md` + `analyze_species_ap.py` is the fix. This review checks whether the fix is
correct and whether the resulting document's claims are honest.

**Method**: every number quoted in the prose was checked against its source CSV; every CSV was checked
against the raw split/prediction artifacts by re-running `analyze_species_ap.py` end-to-end and
independently recomputing Spearman rho, Benjamini–Hochberg q-values, and cross-variable correlations
in a fresh script (not by reusing the module's own functions where avoidable). Nothing here trusts the
prose on its word.

---

## Reproducibility check (prerequisite for trusting any of the claims below)

Re-ran `python analyze_species_ap.py --seed 42 --bootstrap 2000 --cluster-bootstrap 1000
--output-dir /tmp/rerun_ap_support` from the currently checked-out (dirty) tree against the same
on-disk `data/splits_species_v1/` and `checkpoints/perch/species_v1/` inputs.

- `ap_support_analysis.md`: **byte-identical** to the committed copy (`filecmp.cmp(..., shallow=False)
  == True`).
- All six `analysis/*.csv` outputs: **byte-identical**.
- This confirms the seeded bootstrap (species-resample for Spearman CIs, sound_id cluster-resample for
  per-species AP CIs) is fully deterministic given fixed inputs, and that the markdown is generated
  programmatically from the same data it displays — there is no hand-transcription drift between table
  and prose.

## Independent statistical recomputation

Recomputed directly from `analysis/ap_support_per_species.csv` in a standalone script (own
`spearmanr`, own from-scratch BH implementation, no import of `analyze_species_ap`):

| Check | Reported | Independently recomputed | Match |
|---|---|---|---|
| Spearman rho, `train_canonical_windows` vs `ap` | 0.573 | 0.5733 | yes |
| Spearman rho, `val_windows` vs `ap` | 0.448 | 0.4485 | yes |
| Spearman rho, `test_windows` vs `ap` | 0.521 | 0.5214 | yes |
| BH q-values (14-test family: 7 predictors × {ap, ap_lift}) | as in CSV | max abs diff vs. independent BH = 5.98e-17 | yes |
| Partial rho, `train_canonical_windows` controlling `test_windows` | 0.303 (p=0.0121) | matches CSV `ap_support_partial_correlations.csv` exactly | yes |
| Partial rho, `train_sound_ids` controlling `test_windows` | "largely disappears" | 0.0024 (p=0.984) | yes, disappears |
| Sensitivity: train_canonical rho at 68 / `test_pos>=5` / `test_pos>=10` | 0.573 / 0.433 / 0.203 | 0.5733 / 0.4330 / 0.2028 | yes |
| Species with `test_positive_sound_ids==1` (non-estimable CI) | "Ten species" | 10, exact code list confirmed (CROANI, CYAAFF, CYCGUJ, DRYLIN, HYLFLA, MESCAY, NYCALB, PIOMEN, QUEPUR, VOLJAC) | yes |
| `ap_lift = ap - test_prevalence` | claimed | max abs diff = 9.996e-17, min lift = 0.00056 (no negative lifts) | yes |
| Pairwise collinearity among support predictors | "highly correlated by stratified split" | rho(train_canonical, test_windows)=0.807, rho(train_canonical, val_windows)=0.790, rho(test_windows, val_windows)=0.767 | yes, strongly correlated |

No discrepancy found between any quoted statistic and its source data. Every number in the document is
traceable to a CSV cell or is exactly reproducible from raw predictions/splits.

---

## Causal Language

- **Framing is explicit and consistent**: opens with "Correlations are across the 68 species, not
  across windows, and therefore describe association rather than causation," and the Interpretation
  section explicitly states test support "must not be interpreted as a causal improvement in the
  trained classifier." Limitation #5 explicitly disclaims the technical-validation-relevant
  misreading: "cannot establish that adding a specific number of annotations will cause a corresponding
  AP increase." This is the correct guard against the most likely misuse of this document (arguing for
  more annotation effort on the strength of this correlation alone).
- **One residual soft-causal phrase**: "window support and split-wide species commonness, rather than
  audio-file count alone, **drive** most of the observed relationship" (Main Result) and "explain more
  than recording count alone" (Interpretation). "Drive" is a common shorthand in partial-correlation
  writeups for "the variance in the correlation is better attributed to X than Y," but it reads more
  causally than the rest of the document's careful hedging and could be lifted out of context (e.g.
  into a paper draft) without its surrounding disclaimers. **Not a validity problem** — the underlying
  partial-correlation logic is sound (see below) — but a wording inconsistency worth tightening to
  "is more strongly associated with" for internal consistency with the rest of the document.
- No claim anywhere states or implies that the trained classifier's species-level skill was *caused
  by* representation in a way that isn't immediately qualified.

## Prevalence / Confounding

- The core prevalence confound (raw AP's floor scales with test prevalence, and prevalence correlates
  with support — flagged as CRITICAL/WARNING in both prior reviews) is now addressed with a computed
  `ap_lift = ap - test_prevalence` column and its own correlation family. Independently confirmed:
  `ap_lift` correlations for canonical train windows are 0.570 vs. raw `ap`'s 0.573 — nearly identical,
  which correctly supports the stated claim "AP lift above prevalence gives nearly the same
  correlations, so the result is not explained only by AP's prevalence baseline."
- One nuance the document does not spell out explicitly: `ap_lift` only neutralizes the *test-side*
  prevalence floor, not a train-side prevalence confound (a species that is generally common has more
  train support *and* plausibly a more separable acoustic signature/dataset representation
  independent of pure count). This is fine — that residual channel is the actual hypothesis under test
  (does representation associate with skill), not something that should be regressed away — but the
  document could state in one sentence that `ap_lift` controls the test-metric floor, not a general
  "commonness" confound, to preempt a reader assuming prevalence is fully neutralized.
- Partial correlation controlling for `test_windows` (and additionally `val_windows`) is the right tool
  to attempt to separate correlated support signals, and the result direction (canonical train support
  retains ~0.30 residual rho, p=0.012; distinct train audio files drop to ~0.00, p=0.98) is
  independently reproduced and correctly interpreted as suggesting window-volume/commonness rather than
  file-count is doing the explanatory work.

## Multiple Comparisons

- The primary correlation family (7 support predictors × 2 responses `ap`/`ap_lift` = 14 tests) is
  Benjamini–Hochberg corrected (`spearman_q_bh` column), and the corrected q-values are quoted in the
  Main Result table rather than raw p-values — correct practice, independently verified exact.
- **Gap**: the partial-correlation table (4 predictors × 2 control sets = 8 tests) and the
  sensitivity-subset table (4 subsets × 4 predictors = 16 tests) are **not** BH-corrected, and the
  prose quotes an unadjusted p-value from the partial-correlation table (p=0.0121 for
  `train_canonical_windows` controlling `test_windows`). This is a secondary/exploratory interrogation
  of the same underlying relationship already established (and BH-corrected) in the primary table, so
  the risk of a false-discovery narrative is low, but it is still an inconsistency: one family gets
  formal multiplicity control and two adjacent families invoked in the same document do not.
  **Recommendation (non-blocking)**: label the partial/sensitivity p-values in prose as "unadjusted,
  exploratory" or extend BH correction to the full test bank.

## Thin-Support / Cluster-Bootstrap Interpretation

- Per-species AP confidence intervals use a **cluster bootstrap keyed on `sound_id`** (not window),
  which is the statistically correct resampling unit here since windows from the same recording are not
  independent — this directly closes the pseudoreplication gap flagged in the prior data-validation
  review. Verified in code (`cluster_bootstrap_ap`): resamples whole sound-id groups, requires
  `positive_sound_ids >= 2` before attempting an interval, and additionally requires at least half the
  bootstrap draws to yield a class with both labels present (`len(values) >= n_bootstrap // 2`) before
  marking `ap_ci_estimable=True`. This is a defensibly conservative estimability gate.
- The document is explicit and correct that a single-positive-sound-id species (10 of 68, exact list
  independently confirmed above) gets `ap_ci_low/high = NaN` rather than a fabricated interval — "Their
  AP confidence interval is marked non-estimable because resampling cannot create independent positive
  evidence that does not exist." This is exactly the right statistical stance and is visible in the
  Highest/Lowest AP tables (e.g., QUEPUR shows `ap=1.000, ap_ci_low=nan, ap_ci_high=nan`) rather than
  hiding the instability behind a headline number, addressing the prior review's WARNING to "either
  exclude species with `test_pos<5` from headline tables or annotate them explicitly as unstable point
  estimates."
- The Test-Support Strata table backs the qualitative claim: mean AP for `test_windows` in {1,2} is
  0.221 with std 0.314 and a range from 0.001 to 1.000 — i.e., genuinely bimodal/unstable, not merely
  "lower on average." Correctly interpreted in prose as instability, not as a clean low-support penalty.
- **Minor gap**: the claim that test-positive support "also has a strong association because thin test
  sets produce high-variance AP estimates" is a plausible interpretation but is not directly
  demonstrated (e.g., via a null-simulation showing how much of the observed rho is attributable to
  estimator variance vs. a genuine dose-response signal). The bin table itself is fairly monotonic
  (0.221 → 0.366 → 0.360 → 0.537 → 0.630 across increasing support bins, with only a small
  non-monotonic dip from the 3-4 to 5-9 bin), which is consistent with a real, not purely
  noise-driven, relationship. This doesn't invalidate the claim, but it is currently asserted as
  established mechanism rather than flagged as one plausible contributor among others (a genuine
  dose-response effect is at least as consistent with the bin table). **Recommendation (non-blocking)**:
  soften to "may in part reflect" rather than stating the variance explanation as the given reason.

## Sensitivity Subsets

- Sensitivity to thin support is checked at three thresholds (`test_pos>=5`, `test_pos>=10`,
  `test_positive_sound_ids>=2`) and independently reproduced exactly. The direction of the finding
  (correlation strength drops sharply as thin-support species are excluded: 0.573 → 0.433 → 0.203 for
  canonical train windows) is real and correctly used to support the document's own caveat that "the
  raw correlation should not be read as a stable dose-response relationship" — this is a genuinely
  self-undermining sensitivity check the authors report against their own headline number, which is
  good scientific practice.
- Sensitivity subsets reuse the same underlying `test_pos>=5`/`>=10` gates already established in
  `results.md`'s macro-AP sensitivity table rather than inventing new ad hoc thresholds — consistent
  with the prior data-validation review's recommendation to reuse existing gates.
- One point worth flagging for a future reader: at `test_pos>=10` the subset is only n=25 species, and
  `train_sound_ids`'s sensitivity rho flips sign (-0.133, p=0.53, n.s.) relative to the full-sample
  0.387 (p=0.001). This sign flip is not discussed in prose (only the canonical-train-windows numbers
  are narrated). It is not a contradiction — n=25 with p=0.53 is consistent with "no detectable
  association," not "reversed effect" — but a reader skimming only the sensitivity CSV without the
  prose's guidance could momentarily misread the sign change as a substantive reversal. Non-blocking;
  the CSV is disclosed as a reproducible artifact specifically so a reader can find this, but a one-line
  caveat in prose would preempt misreading.

## Technical-Validation Implications

- `results.md` (the parent technical-validation document this analysis supports) states "performance
  improves as the minimum test support increases" referring to the **threshold** macro-AP sensitivity
  (`test_pos>=5`: 0.4817, `>=10`: 0.5738) — this is mechanically true (raising a inclusion threshold on
  a positively-AP-correlated variable cannot decrease the subset mean by more than sampling noise) and
  is not contradicted by `ap_support_analysis.md`'s bin table, which instead adds necessary nuance
  (per-bin non-monotonicity, thin-support instability) that a reader of the headline threshold numbers
  alone would miss. The two documents are complementary, not in tension, and `results.md` links directly
  to `ap_support_analysis.md` for this nuance rather than asserting the stronger dose-response claim
  unqualified.
- Limitation #5 ("this analysis is descriptive and cannot establish that adding a specific number of
  annotations will cause a corresponding AP increase") is the single most important sentence for
  technical-validation purposes: it correctly forecloses the most likely downstream misuse of this
  document (justifying an annotation-effort decision on the strength of an observational correlation).
- The document is not currently referenced from `paper/` (confirmed by grep) — no live discrepancy
  exists between this analysis's careful hedging and any published/drafted claim. Flagged as a
  forward-looking recommendation only: when/if this analysis is folded into the paper's technical
  validation section, the non-causal framing and thin-support caveats must be carried over verbatim,
  not compressed into a bare correlation coefficient.
- Validation-support correlation is correctly pre-empted from a circularity critique: the document
  notes "the validation set only selected one global C; it did not tune a separate classifier per
  species," which forecloses the objection that validation-support correlating with AP is an artifact
  of per-species overfitting to validation data (C selection was global, per `results.md`'s C-selection
  table, not per-species).

---

## Pitfalls Checked, Not Found

- No fabricated confidence intervals for non-estimable species (correctly `NaN`).
- No mixing of augmented and canonical train counts under one column name (the exact trap that caused
  the prior FAIL) — `train_canonical_windows` and `train_aug_windows` are separate, both reported, and
  the document explicitly discusses the direction of their difference.
- No raw Pearson-only correlation presented as primary (Spearman is primary; Pearson-on-log1p is
  reported as a secondary robustness check, consistent with the prior data-validation review's
  recommendation).
- No `no_bird` pseudo-row or taxonomy-crosswalk leakage in the current join (this analysis, unlike the
  originally reviewed candidate design, is scoped entirely to `class_list.json`'s 68 codes via
  `species_ap.csv`/`test_predictions.npz`, sidestepping that hazard entirely).
- No NaN silently entering a correlation: `ap_ci_estimable`/non-finite gating is handled explicitly in
  `cluster_bootstrap_ap`, and the correlation tables use `ap`/`ap_lift`, which are finite for all 68
  species (verified in `ap_support_per_species.csv`).

## Recommendations

- [ ] WARNING: Extend Benjamini–Hochberg correction (or explicitly label as "unadjusted, exploratory")
  to the partial-correlation and sensitivity-subset p-values quoted in prose, for consistency with the
  primary correlation family's treatment.
- [ ] WARNING: Soften "drive most of the observed relationship" / "explain more than ... alone" to
  non-causal phrasing ("is more strongly associated with") for internal consistency with the document's
  own explicit association-not-causation framing.
- [ ] SUGGESTION: Add one sentence clarifying that `ap_lift` neutralizes the test-metric prevalence
  floor specifically, not a general train-side "commonness" confound, so the "not explained only by
  AP's prevalence baseline" claim isn't over-read as fully deconfounded.
- [ ] SUGGESTION: Soften the "thin test sets produce high-variance AP estimates" explanation for the
  strength of the test-support correlation to "may in part reflect," since the bin table is also
  consistent with a genuine (non-noise) dose-response contribution and the document does not
  decompose the two.
- [ ] SUGGESTION: Add a one-line caveat about the `train_sound_ids` sign flip in the `test_pos>=10`
  sensitivity subset (rho -0.133, n.s., n=25) to preempt a reader misreading a non-significant sign
  change as a reversal.
- [ ] SUGGESTION: When this analysis is eventually incorporated into `paper/`, carry over the
  association-not-causation framing and Limitation #5 verbatim rather than compressing to a bare
  correlation coefficient.

---

## Conclusion

Every quoted statistic in `ap_support_analysis.md` was independently traced to its source CSV and, for
the core correlation/BH/partial-correlation numbers, independently recomputed from scratch — all match
exactly. Re-running `analyze_species_ap.py` end-to-end with the documented seed reproduces the markdown
and all six CSVs byte-for-byte, confirming the reported bootstrap CIs and sensitivity numbers are not
cherry-picked or hand-edited. The document directly and successfully remediates the two CRITICAL gaps
identified in the prior `ap_support_experiment_review.md` (**FAIL**) review — per-species uncertainty
now exists via a correctly-scoped `sound_id` cluster bootstrap with honest non-estimability handling,
and `train_pos`'s augmented/canonical ambiguity is now resolved into two clearly separated,
consistently-used columns — and satisfies every CRITICAL item from the companion
`ap_support_data_validation.md` (**PASS**) review's implementation checklist (support definitions
labeled, `test_evaluable`-equivalent filtering implicit in the 68-row table with no NaN, Spearman
primary, sensitivity subsets reused from existing gates). Causal language is explicitly and repeatedly
disclaimed, the prevalence confound is addressed via a computed lift metric, thin-support instability is
surfaced rather than hidden, and the technical-validation implication most likely to be misused
(justifying more annotation effort from this correlation alone) is explicitly foreclosed in the
document's own Limitations section.

The issues found here are refinements to an already rigorous document — a residual soft-causal phrase,
inconsistent multiplicity treatment between the primary and secondary test families, and two
interpretive claims (variance-driven test-support correlation, prevalence-lift's scope) stated with more
confidence than directly demonstrated — none of which invalidate the document's central, correctly
hedged claim that representation is *associated with*, not proven to *cause*, higher per-species AP.

STATUS: PASS
