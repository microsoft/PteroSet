# Experiment Integrity Review — Per-Class AP vs. Representation Analysis (PteroSet Perch v2 Species Probe)

**Scope**: this review evaluates the scientific rigor of using the final Perch v2 species linear-probe
results to draw a **per-class AP vs. representation** conclusion (i.e. "species with more training
examples get higher AP," or any scatter/correlation built from `species_ap.csv`'s `train_pos` /
`val_pos` / `test_pos` columns against its `ap` column). No standalone script or notebook implementing
this specific correlation currently exists in the repository; the narrative in
`docs/implementation/species-linear-probe-v1/results.md` ("performance improves as the minimum test
support increases") and the raw per-species table are the load-bearing artifacts a reader or a future
script would use to make this claim, so this review audits those artifacts and the code that produced
them.

**Reviewed artifacts**:
- `checkpoints/perch/species_v1/species_ap.csv`, `species_diagnostics.csv`, `macro_ap_summary.csv`,
  `no_bird_detection.csv`, `training_manifest.json`
- `data/splits_species_v1/species_distribution.csv`, `split_manifest.json`
- `train_perch_logreg.py`, `prepare_species_splits.py`, `extract_perch_embeddings.py`
- `docs/implementation/species-linear-probe-v1/results.md`
- Prior review `docs/review/phase3_experiment_review.md` (STATUS: PASS) — that review audited overall
  pipeline reproducibility for this same experiment; findings below are additional/narrower and are
  cross-referenced rather than repeated where they overlap.

**Method**: claims below are backed by independent recomputation from the on-disk CSV/JSON artifacts
(shown inline), not by trusting the prose in `results.md`.

---

## Experiment Integrity Report

### Reproducibility
- **Seeds**: `prepare_species_splits.py --seed 42` deterministically drives the sound_id-grouped
  70/15/15 split assignment (repeated-greedy + local search, 50 restarts); `split_manifest.json`
  records `"seed": 42` and `"selected_restart_seed"`. `train_perch_logreg.py` uses only `lbfgs`
  (deterministic convex solver) — no seed needed for the classifier itself. This part is sound and
  consistent with the prior phase3 review's finding.
- **Environment capture**: `training_manifest.json` records `numpy`/`scikit-learn`/`joblib` versions;
  `embedding_manifest.json` records `tensorflow`/`librosa`/`kagglehub`/`pandas` versions and GPU device
  list. No Python interpreter version, OS version, or CUDA/cuDNN version is captured anywhere (already
  flagged as a SUGGESTION in `phase3_experiment_review.md`; repeated here only because it also limits
  reproducibility of the specific AP numbers this analysis depends on).
- **Commit identity**: `git_dirty: true`, `git_commit: null` in both manifests — confirmed live
  (`git status --short` currently shows 10 modified/untracked paths including `train_perch_logreg.py`
  and this experiment's own docs). The design choice not to fabricate a stale SHA on a dirty tree is
  good practice, but it means the exact code state that produced `species_ap.csv` is **not yet
  committed and not tied to a citable revision**. Any figure/claim built on top of "final PteroSet
  Perch results" is provisional until this is committed.
- **Determinism of the specific numbers used for representation**: not independently verified by
  re-running end-to-end in this review (no re-run was performed here); `lbfgs` + single-threaded BLAS
  under `threadpool_limits(limits=1)` makes this a low-risk gap, not a demonstrated one.

### Data Integrity
- **Leakage risk**: none detected for the AP numbers themselves — `sound_id` is disjoint across
  train/val/test (validated by `validate_splits_disjoint` and re-checked structurally in the prior
  review).
- **Test prevalence handling — PARTIAL, confound not neutralized**:
  - `species_ap.csv` correctly reports `test_prevalence` and `prevalence_baseline_ap` (the two are
    identical, i.e. the theoretical AP of a random/no-skill ranker) per species, so the *inputs* needed
    to control for prevalence exist.
  - Test prevalence spans a **115x range** across the 68 species (min `0.000154`, i.e. AMAFAR/COEFLA-scale
    rarity, to max `0.01776`, recomputed directly from the CSV). Because AP's floor is the class
    prevalence, raw `ap` is not comparable across species with such disparate baselines — a species at
    AP=0.4 with baseline 0.0179 is unremarkable, while AP=0.4 with baseline 0.0002 is a ~2000x lift.
  - No per-species **lift/normalized metric** (e.g. `ap - prevalence_baseline_ap`, `ap / prevalence_baseline_ap`,
    or a PR-gain statistic) is precomputed anywhere in the pipeline; only the single macro-level delta
    (`0.4070`, from `macro_ap_summary.csv`) is reported, and that is a pooled average, not a per-species
    figure.
  - Consequence for an "AP vs representation" plot: because prevalence and `train_pos` are themselves
    correlated (more common species have more train and test instances), a naive `ap` vs `train_pos`
    trend will mechanically reproduce part of the "AP scales with prevalence" tautology described above.
    Any legitimate representation analysis must regress/plot the **lift-over-baseline** metric against
    support, not raw AP, or must explicitly report and discuss this confound.
- **Actual vs. canonical train support — CRITICAL, quantified inflation, species-dependent**:
  - `species_ap.csv`'s `train_pos` is computed from `train_emb.npz`, which is extracted from
    `train_split.csv` — the **augmented** split (`prepare_species_splits.py` line ~50: "`train_split.csv`
    is then augmented with every other (overlapping) modeled window belonging to a train-assigned
    sound_id"). `val_pos`/`test_pos`, by contrast, come from `val_split.csv`/`test_split.csv`, which are
    canonical (non-overlapping) only. This asymmetry is disclosed in `results.md` ("Training uses the
    augmented train split; validation and test contain canonical non-overlapping windows only") but its
    quantitative impact on a *representation* metric is not.
  - `split_manifest.json` confirms global inflation: `canonical_base_sizes.train = 29,513` vs.
    `augmented_train_size = 95,536` (+66,023 windows, a **3.24x** global multiplier).
  - Recomputing this ratio **per species** (canonical `train_count` from
    `species_distribution.csv` vs. augmented `train_pos` from `species_ap.csv`) shows the multiplier is
    **not constant across species**:

    | code | canonical train_count | augmented train_pos | ratio |
    |---|---:|---:|---:|
    | AMAAMA | 11 | 79 | 7.18x |
    | LEPRUF | 5 | 34 | 6.80x |
    | MESCAY | 4 | 25 | 6.25x |
    | QUEPUR | 13 | 72 | 5.54x |
    | ... | | | |
    | VOLJAC | 11 | 27 | 2.45x |
    | ORTGAR | 39 | 108 | 2.77x |
    | CYAAFF | 14 | 39 | 2.79x |

    Full range: **2.45x–7.18x**, mean **3.71x**, stdev **0.94x** (n=68, recomputed from the two CSVs
    above; every species has canonical `train_count >= 1`). This is presumably driven by per-species call
    duration and temporal clustering (a long or clustered vocalization is caught by more overlapping
    windows than a short, isolated one), which is exactly the kind of species-level acoustic property
    that could independently affect classifier learnability — i.e., the inflation factor is a plausible
    **confounder**, not just noise.
  - Practical implication: `train_pos` as currently reported is **not an apples-to-apples "representation"
    variable** across species. A species with `train_pos=113` (AKLMEL) is not necessarily 2x as
    "represented" as one with `train_pos=57` if their canonical (true distinct-instance) counts are, say,
    28 and 28 — the ratio depends on unmodeled acoustic/behavioral factors, and using the augmented count
    directly will bias any per-class AP-vs-support trend by an amount that varies unpredictably (up to
    ~3x relative to the multiplier's own spread) across the x-axis.
  - Recommendation: any representation analysis should use `canonical_train_split.csv` /
    `species_distribution.csv`'s `train_count` (the audited, non-inflated count) as the "representation"
    axis, or at minimum report both counts side-by-side and discuss the inflation-ratio spread as a
    limitation of using the augmented count.
- **Preprocessing consistency**: consistent per `results.md` and code inspection — L2 normalization is
  applied independently per split (train/val/test), embeddings are extracted with identical windowing
  parameters (`window_sec=5.0`, `target_sample_rate=32000`, etc., per `embedding_manifest.json`) across
  splits; augmentation (overlap) is confined to train only, matching stated design.

### Training Configuration
- **Checkpointing / manifest completeness**: `training_manifest.json` captures classifier config,
  input hashes, dependency versions, timings, and a resumability `config_hash` — this is solid and
  already validated behaviorally in the prior review (re-running produced a no-op resume).
- **Metric computation — bootstrap/permutation uncertainty is ABSENT**:
  - `macro_average_precision()` and the per-species AP loop in `train_perch_logreg.py`
    (`_species_ap_rows`, using `sklearn.metrics.average_precision_score`) compute a single point
    estimate per species. **No bootstrap resampling, no permutation test, and no analytic confidence
    interval is computed anywhere in the codebase** for either per-species AP or the macro-AP summary
    (`grep` for `bootstrap|permutation|pearsonr|spearmanr|confidence` across `.py`/`.md` returns no
    statistical-uncertainty code; the only "bootstrap" hits are the unrelated CUDA-library bootstrap in
    `docs/design/perch2_species_linear_probe_plan.md`).
  - This is a **critical gap specifically for a per-class AP vs. representation analysis**, because
    per-species AP variance is a direct function of test support, and test support here is *extremely*
    thin for a large fraction of the vocabulary: 24/68 species have `test_pos < 5`, 43/68 have
    `test_pos < 10` (per `results.md`'s own Limitation #2, corroborated by `species_ap.csv`). Concretely,
    recomputed from `species_ap.csv`:
    - **QUEPUR**: `test_pos=1`, `ap=1.0000` — headlined in `results.md`'s "Highest AP" table. With a
      single positive test example, AP is a near-coin-flip statistic (it equals 1.0 iff that one
      positive happens to be ranked above all 6,473 negatives by the model's score for that class);
      it carries almost no information about the classifier's true precision-recall behavior for this
      species, yet it visually anchors one end of any AP-vs-support plot.
    - **ORTGAR**: `test_pos=3`, `ap=0.8667` — similarly thin.
    - **MESCAY**: `test_pos=1`, `ap=0.0007` (the "Lowest AP" table) — same problem in the other
      direction.
    - Without a per-species bootstrap CI (e.g., resample test rows with replacement, stratified by
      class, recompute AP, report the 2.5/97.5 percentile band) or a permutation-based null (shuffle
      scores within each species' test column to estimate the AP achievable by chance given that
      species' exact prevalence), none of the "high-AP" or "low-AP" per-species examples in
      `results.md`, nor any future scatter of `ap` vs. `train_pos`/`test_pos`, can be distinguished from
      sampling noise at low support. A correlation coefficient or trend line fit across all 68 points
      without CIs or support-weighting would be dominated by these unstable, low-support points.
  - `results.md` does already gate on this partially by reporting `macro AP restricted to test_pos>=5`
    (0.4817, 44 species) and `test_pos>=10` (0.5738, 25 species) — a reasonable **sensitivity check** at
    the macro level — but this does not substitute for per-species uncertainty, and is explicitly
    labeled in the C-selection table as "sensitivity summaries, not alternative selection rules," which
    is correct but underscores that no formal per-species CI exists.
- **Optimizer/scheduler sanity**: N/A (logistic regression, no LR schedule); `class_weight="balanced"`
  is applied uniformly, computed from the **augmented** train labels — given the inflation-ratio
  variability documented above, the per-class weights implicitly used at fit time are themselves
  affected by the same non-uniform inflation, which is a second-order channel through which the
  train/canonical mismatch could affect learned decision boundaries (not just the reported "representation"
  number).

### Pitfalls Detected
- Raw per-species AP compared across species with a 115x range in test prevalence, with no lift/
  normalized metric computed — risks conflating "well represented" with "common enough that AP's floor
  is already high."
- `train_pos` used (implicitly, via the per-species table) as a stand-in for "representation" is drawn
  from an overlap-augmented split whose inflation factor vs. the canonical/audited count varies 2.45x–
  7.18x across species — not a consistent unit.
- Headline per-species AP values (both highest and lowest in `results.md`) are drawn predominantly from
  `test_pos` in {1, 2, 3} — single-digit-support point estimates with no uncertainty band, presented
  alongside high-support species (e.g. CRYCIN, `test_pos=57`) without visually or statistically
  distinguishing estimate reliability.
- No bootstrap or permutation procedure exists in `train_perch_logreg.py` or elsewhere in the repo to
  quantify per-species AP uncertainty or to test whether an observed AP-vs-support trend is stronger
  than chance.

### Recommendations
- [ ] CRITICAL: Before publishing or plotting any per-class AP vs. representation relationship, add
  bootstrap confidence intervals per species (resample test rows, recompute AP, report percentile band)
  and/or a permutation-based null, and only interpret/report per-species AP for species whose CI half-width
  is below a pre-registered threshold, or explicitly weight/size-code points by test support in any plot.
- [ ] CRITICAL: Replace or augment `train_pos` in the representation axis with the canonical
  (non-inflated) train count from `species_distribution.csv`/`canonical_train_split.csv`. If the
  augmented count must be used (e.g. because it is what the classifier actually saw), report the
  per-species inflation ratio alongside it and discuss it as a confound rather than treating `train_pos`
  as a clean "representation" measure.
- [ ] WARNING: Compute and report a per-species prevalence-normalized metric (e.g. `ap -
  prevalence_baseline_ap` or `ap / prevalence_baseline_ap`) so that an AP-vs-representation trend is not
  a restatement of the AP-floor-scales-with-prevalence relationship.
- [ ] WARNING: When presenting "highest AP" / "lowest AP" per-species examples (as in `results.md`),
  either exclude species with `test_pos < 5` from headline tables or annotate them explicitly as
  unstable point estimates (this repo already computes `test_pos`, so the gate is cheap to apply).
- [ ] SUGGESTION: Record Python/OS/CUDA-cuDNN versions in `training_manifest.json`/
  `embedding_manifest.json` (already flagged in `phase3_experiment_review.md`; still relevant to full
  reproducibility of the AP numbers this analysis depends on).
- [ ] SUGGESTION: Commit the current working tree (or at least the experiment-producing scripts and
  manifests) so `git_commit` is non-null and the "final PteroSet Perch results" can be cited by a fixed
  revision rather than content hashes alone.
- [ ] SUGGESTION: Add an explicit end-to-end reproducibility test (re-run `train_perch_logreg.py` twice
  from the same embeddings and assert byte-identical `species_ap.csv`) to convert the current
  "should be deterministic because lbfgs is deterministic" argument into a verified guarantee.

---

## Conclusion

The underlying pipeline (`prepare_species_splits.py`, `extract_perch_embeddings.py`,
`train_perch_logreg.py`) is engineered with real rigor: seeded deterministic splits, sha256-verified
manifests, a structural test-data guard that makes it impossible to peek at test data before `C*` is
selected, atomic writes, and a resumability config hash independently verified in
`phase3_experiment_review.md`. That review's STATUS: PASS for overall experiment reproducibility stands.

However, this narrower review — specifically of a **per-class AP vs. representation** analysis — finds
two CRITICAL gaps that directly threaten the validity of that specific claim: (1) no bootstrap or
permutation-based uncertainty quantification exists anywhere for per-species AP, and several headline
per-species AP values (e.g. QUEPUR AP=1.0 at `test_pos=1`) are statistically unstable point estimates
presented without caveats sufficient to prevent misreading in a trend/correlation context; and (2) the
`train_pos` figure available for "representation" is drawn from an overlap-augmented split whose
inflation ratio relative to the canonical/audited train count varies 2.45x–7.18x across species (mean
3.71x, sd 0.94x), making it an inconsistent, confounded proxy for true representation. A third,
non-blocking issue is that raw AP is not prevalence-normalized, risking conflation of "well represented"
with "common enough that the AP floor is already high" (115x range in test prevalence across species).

These are fixable without re-running the expensive embedding/training pipeline (all three fixes are
downstream analysis additions using already-computed artifacts), but until they are addressed, a
per-class AP vs. representation conclusion drawn from the current artifacts should be treated as
illustrative, not statistically validated.

STATUS: FAIL
