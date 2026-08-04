# Phase 3 Experiment Integrity Review — Perch v2 Species Linear Probe

**Reviewed artifact set**: `train_perch_logreg.py`, `docs/design/perch2_species_linear_probe_plan.md`
(STATUS: FINAL), `docs/implementation/species-linear-probe-v1/results.md`, and the actual generated
outputs in `checkpoints/perch/species_v1/`, `data/embeddings/perch_v2/species_v1/`, and
`data/splits_species_v1/`.

**Method**: this review does not take the plan/results documents on faith. Every claim below was
checked either by reading the executing code path directly, by re-running the trainer and test
suite in the `bioacoustics` conda env, or by recomputing the reported numbers from the on-disk
artifacts independently of the scripts that produced them.

**Reproduction commands used**:
- `pytest tests/test_train_perch_logreg.py -q` → 89 passed
- `pytest tests/ -q` → 255 passed, 4 warnings (unrelated librosa/PySoundFile deprecation warnings)
- `python train_perch_logreg.py` (no args, i.e. default config) → resumed as no-op against the
  existing `checkpoints/perch/species_v1/` outputs
- Independent recomputation from `species_ap.csv`/`macro_ap_summary.csv`/`c_selection.csv`/
  `no_bird_detection.csv` of: thin-support counts, top/bottom-5 per-species AP, and sha256 of
  `class_list.json` / `embedding_manifest.json` against the values recorded in
  `training_manifest.json`.

---

## Experiment Integrity Report

### Reproducibility
- **Seeds**: `train_perch_logreg.py` uses only `lbfgs` (a deterministic convex solver) with no
  stochastic component in its own code — no seed is required or claimed, and the module docstring
  states this explicitly. Upstream, `prepare_species_splits.py` seeds the stratified split
  construction (`--seed 42`) and this is verified structurally: `split_manifest.json`'s recorded
  `split_manifest_sha256` is cross-checked byte-for-byte against the live file at trainer start
  (`split_manifest_sha256_verified: true` in the actual manifest), so an accidental split
  regeneration would be caught, not silently trusted.
- **Environment capture**: `training_manifest.json` records `scikit-learn`, `numpy`, `joblib`
  versions (verified: `1.7.2`/`2.2.5`/`1.5.3`, matching the installed env); the upstream
  `embedding_manifest.json` additionally records `tensorflow`/`librosa`/`kagglehub`/`pandas`
  versions, GPU device list, and TF version. No OS version or CUDA/cuDNN version is recorded in
  either manifest — a minor gap, since Phase 2 (`extract_perch_embeddings.py`) is the only GPU-bound
  stage and its manifest is the natural place for it.
- **Commit/content identity**: git is genuinely dirty in this repo right now (`git status
  --porcelain` shows 10 changed/untracked paths), and the manifest correctly records
  `git_commit: null, git_dirty: true` rather than fabricating a stale SHA — this matches the
  documented design decision ("a dirty tree never reports a sha that would overstate
  reproducibility"). In lieu of a commit SHA, `script_source_sha256` (verified against the live
  `train_perch_logreg.py` file), `class_list_sha256`, and `embedding_manifest_sha256` are recorded
  and independently re-verified in this review to match the actual files on disk.
- **Config logging**: the full config (C grid, class_weight, solver, max_iter, n_jobs,
  `limit_train`) plus a `config_hash` combining all of it with the input hashes and the script's own
  source hash is written to `training_manifest.json`. Re-running `python train_perch_logreg.py` with
  no arguments reproduced the identical `config_hash` and correctly resumed as a no-op, which is a
  direct behavioral proof (not just a documentation claim) that the hash captures everything needed
  to detect drift.
- **Determinism**: `lbfgs` is deterministic; `OneVsRestClassifier`'s `n_jobs`-driven parallel backend
  is explicitly excluded from the config hash "because it affects wall-clock only, never results" —
  this is true for `lbfgs` (no shared mutable state across per-species fits) and is a reasonable,
  disclosed exception, not a silent one.

### Data Integrity
- **Leakage risk**: none detected. `sound_id` is disjoint across train/val/test — independently
  re-verified directly from the split CSVs (0 overlapping `sound_id`s in all three pairwise
  comparisons). L2 normalization is applied per split independently and statelessly
  (`sklearn.preprocessing.normalize`, no fitted scaler persisted across splits), so no train
  statistic leaks into val/test. `TestDataGuard` provides an active, code-level structural
  guarantee (not just a convention) that test arrays raise `RuntimeError` if read before `C*` is
  fixed — this is exercised directly by `test_test_data_guard_raises_before_unlock` and by the fact
  that `sweep_c_grid` (the C-selection function) is never passed a reference to test data at all
  (two independent layers, as the code comments claim).
- **Preprocessing consistency**: verified — L2 normalization is applied identically and
  independently to train/val/test (same function, same code path, `assert_unit_norm` checked on all
  three immediately after normalization). Train-only augmentation is real and correctly scoped:
  `train_split.csv` contains 95,536 rows (66,023 non-canonical/augmented + 29,513 canonical),
  `val_split.csv`/`test_split.csv` are 100% `is_canonical == True` (independently re-verified from
  the CSVs), matching the documented and reviewed design.
- **Pipeline correctness**: `sound_id`/`window_id`/`start`/`end` identity fields are round-tripped
  from split CSV through embeddings NPZ through predictions NPZ; `validate_splits_disjoint` and
  `validate_split_npz` are unconditional, fail-loud gates run before any modeling step. Class
  vocabulary hash (`class_list_sha256`) is cross-checked between the class list, the embedding
  manifest, and the trainer at every run, closing off a whole class of silent-vocabulary-drift bugs.

### Training Configuration
- **Checkpointing**: `logreg_species.joblib` bundles the fitted `OneVsRestClassifier` plus
  `model_metadata` (class codes, trainable-column indices, preprocessing description, full model
  config, and input hashes) — sufficient to reproduce inference exactly. There is no
  epoch/step-based training loop here (a single-shot `sklearn` fit), so optimizer/scheduler-state
  checkpointing is not applicable; this is a correct scope match to the algorithm, not a gap.
- **Metric computation**: verified end-to-end by independent recomputation from the on-disk
  artifacts — `species_ap.csv`'s thin-support counts (`test_pos < 5` → 24 species, `< 10` → 43
  species) and top/bottom-5 per-species AP values match `results.md` exactly to displayed precision;
  `macro_ap_summary.csv`, `c_selection.csv`, and `no_bird_detection.csv` values match `results.md`'s
  tables exactly (macro-AP `0.4089`, baseline `0.0020`, thin-support macro-APs `0.4817`/`0.5738`,
  any-bird AUROC `0.9437`/AP `0.7133`, selected `C=0.1` at val macro-AP `0.4379`). The prevalence-only
  AP baseline is computed correctly as the per-species label mean (the textbook random-ranker AP
  baseline), not a placeholder constant.
- **Optimizer/scheduler sanity**: `class_weight="balanced"` is passed straight to each per-species
  `LogisticRegression` inside `OneVsRestClassifier`, so imbalance is corrected per binary sub-problem
  (appropriate for a long-tailed, `no_bird`-dominant multilabel task) rather than globally or not at
  all. `C` selection is by validation macro-AP over `trainable ∩ val_evaluable` species only, with an
  explicit, tested lowest-`C` tie-break rule and NaN-never-wins ordering
  (`select_best_c`/`test_c_selection.py`); in this run no tie actually occurred (`C=0.1`'s val
  macro-AP `0.4379` strictly beats `C=1.0`'s `0.4282` and `C=10.0`'s `0.4302`).

### Pitfalls Detected
- None of the classic silent-failure patterns (metric-device mismatch, in-place autograd corruption,
  forgotten `.detach()`, broadcasting bugs, integer division, dtype mixing, eval-mode omission) apply
  here — this is a single-shot `sklearn` fit with no autograd/eval-mode state, and the closest analog
  (BLAS thread oversubscription silently degrading wall-clock, not results) is explicitly diagnosed,
  measured, and mitigated (`threadpoolctl.threadpool_limits(limits=1)` + threading backend), not
  merely asserted.
- **Non-fabrication of predictions**: non-trainable/non-finite-coefficient species are represented as
  `NaN` in prediction/diagnostic outputs (`predict_proba_reindexed`, `test_proba_full[:,
  ~coef_finite] = np.nan`), never as a fabricated zero probability — directly relevant to the
  no-bird/any-bird metric, since a fabricated zero would silently understate `any_bird_score`.
  `any_bird_score_stable` explicitly documents and treats `NaN` columns as "no evidence" (factor of
  1), which is a defensible, disclosed choice, not a silent one.
- **One documentation-traceability gap** (see Recommendations): `results.md` and the CHANGELOG both
  state "the planned go/no-go criterion required macro-AP to exceed the prevalence-only baseline by
  at least 0.05," but the authoritative FINAL plan
  (`docs/design/perch2_species_linear_probe_plan.md`) contains no such numeric margin anywhere in its
  "Metrics," "Phased milestones and acceptance criteria," or "Key operational failure modes"
  sections — the `+0.05` figure only exists in the explicitly-superseded `round_01`/`round_02`
  exploratory proposals, which the FINAL plan's own preamble says need not be read and whose LOPO
  framing "should not be read as still authoritative." This does not affect the experiment's outcome
  (measured improvement is `+0.4070`, roughly 8× any historically-discussed margin), but the results
  report cites a "planned criterion" that is not actually stated in the plan it is nominally
  reporting against.

## Recommendations
- [ ] SUGGESTION: Record OS version and CUDA/cuDNN version in `embedding_manifest.json` (currently
  captures TensorFlow version and GPU device list, but not driver/toolkit versions) to fully close
  the environment-capture gate for the one GPU-bound stage in this pipeline.
- [ ] SUGGESTION: Either add the `+0.05` absolute macro-AP-over-prevalence-baseline margin to the
  FINAL plan's "Metrics"/"Phased milestones" section as an explicit, pre-registered number (so
  `results.md`'s "planned go/no-go criterion" sentence traces to an actual authoritative source), or
  soften that sentence in `results.md`/CHANGELOG to not imply a number that only exists in superseded
  drafts.
- [ ] SUGGESTION: Consider recording an OS-level `pip freeze`/conda env export (or a lockfile hash)
  alongside the three explicitly-pinned package versions in `training_manifest.json`, to catch
  transitive-dependency drift (e.g., a scikit-learn point release changing `lbfgs` numerics) between
  runs, since none of `train_perch_logreg.py`'s own tests would detect that class of drift.

---

## Verdict

Every mechanism this review's mandate calls out was independently verified against the actual
executing code and the actual generated artifacts, not just against the design documents describing
them:

- **Test-once structural guard**: verified in code (`TestDataGuard`, two independent layers) and by
  its dedicated passing tests.
- **Validation-only C selection**: verified — `sweep_c_grid` never receives test data;
  `C=0.1` is the true argmax of val macro-AP (`0.4379` vs `0.3959`/`0.4282`/`0.4302`), independently
  recomputed from `c_selection.csv`.
- **Train-only refit**: verified — the final classifier is refit on `train_X`/`train_Y` only, and
  `test_guard.unlock()` is called strictly after that refit completes.
- **Imbalance handling**: verified — `class_weight="balanced"` applied per per-species binary
  sub-problem inside `OneVsRestClassifier`.
- **Thin-support disclosure**: verified — 24/68 (`test_pos<5`) and 43/68 (`test_pos<10`) species
  independently recomputed from `species_ap.csv` match `results.md` exactly; the limitation is
  stated explicitly, not buried.
- **No-bird metric**: verified — a separate, correctly-derived `any_bird_score`
  (`1 - Π(1-p_i)`, numerically stabilized) AUROC/AP is reported specifically because `no_bird`
  prevalence is preserved rather than downsampled, matching the documented rationale.
- **Convergence**: verified — backend-independent `n_iter_ >= max_iter` check
  (`count_non_converged`), 0 non-converged across all C-grid fits and the final refit
  (independently confirmed in `c_selection.csv` and `training_manifest.json`).
- **Reproducibility/content hashes**: verified by direct action, not inspection alone — re-running
  `python train_perch_logreg.py` reproduced the same `config_hash` and resumed as a true no-op;
  `class_list_sha256`/`embedding_manifest_sha256` recorded in `training_manifest.json` were
  independently recomputed with `sha256sum` and matched exactly; `git_dirty: true`/`git_commit: null`
  matches the actual repository state at review time.
- **Technical-validation claim**: verified — every project (MAP1, PPA1–PPA4) appears in all three
  splits (a genuinely in-distribution, non-cross-project split), matching the plan's explicit
  disclaimer that this is a technical validation and not a cross-site/cross-project generalization
  result; the results/limitations sections state this framing correctly and do not overclaim.

The one finding (the `+0.05` go/no-go citation not tracing to the authoritative FINAL plan) is a
documentation-traceability issue, not a rigor, leakage, or correctness defect, and does not change
the experiment's outcome or its validity. All 89 focused tests and the full 255-test repository suite
pass in the live environment as re-run for this review.

STATUS: PASS
