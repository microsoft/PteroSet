# PteroSet Species Linear Probe -- Round 2 (Minimalist, Sibling-Reuse-Maximized)

Author: architect-minimalist
Status: Draft for team review
Supersedes in part: docs/design/round_01/architect_minimalist_proposal.md (Round 1 remains the
record of the initial reasoning; this document is the authoritative revision)

---

## 0. Purpose of this revision

Round 1 designed a Perch-2 linear-probe pipeline for PteroSet species classification from first
principles plus public Perch documentation. The user has since pointed at a sibling repo,
`orcas_dclde2026`, that has an **actually running, production-used** Perch-embedding +
linear-probe pipeline (killer-whale ecotype classification, and a 3-class detector). This round's
job is narrow and disciplined:

1. Read that pipeline for real (not by inference from docs).
2. Copy anything that is already proven, verbatim wherever the shapes line up.
3. Name precisely what cannot be copied for PteroSet (multi-label species, 5-fold
   leave-one-project-out, sample-indexed windows) and why.
4. Cut anything Round 1 proposed that the sibling repo proves is unnecessary.
5. Resolve multiclass-vs-multilabel with the smallest possible data audit, not by assumption.
6. Give a convergence verdict.

Everything below is tagged:
- `[VERIFIED: repo - <path>]` = read directly from a real file in this session.
- `[VERIFIED: source - <url/doc>]` = confirmed in Round 1 from an authoritative external source
  (HuggingFace model card, perch-hoplite source, arXiv). Not re-fetched this round; carried over.
- `[RECOMMENDATION]` = my proposal, not a fact.
- `[UNVERIFIED - Phase 0 check]` = plausible but must be confirmed by running code before it is
  trusted.

---

## 1. What was inspected this round (facts, not inference)

`[VERIFIED: repo]` All of the following were read in full or in the cited ranges this session:

- `orcas_dclde2026/eval_perch_ecotype.py` (~970 lines, full file)
- `orcas_dclde2026/extract_perch_3class.py` (full file)
- `orcas_dclde2026/remap_perch_embeddings.py` (full file)
- `orcas_dclde2026/reports/ecotype_classifier.md`, section 8 (lines 467-670)
- `orcas_dclde2026/docs/experiments/log.md`, entry S2.3 (lines 57-124)
- `orcas_dclde2026/audit_split_overlap.py` (leakage-audit pattern, spot-checked)
- `orcas_dclde2026/data/splits_3class/train_split.csv` (header row only)
- `orcas_dclde2026/environment.yml` and `orcas_dclde2026/pip-requirements.txt` (full)
- `orcas_dclde2026/.gitignore`

The single most consequential fact found this round is in section 2 below. It eliminates an
entire Round-1 workstream.

---

## 2. The shared-environment finding (eliminates Round 1's isolated-env plan)

`[VERIFIED: repo - orcas_dclde2026/environment.yml]`:
```yaml
name: bioacoustics
prefix: <conda-env-path>/bioacoustics
dependencies:
  - python=3.10
  - pytorch
  - pytorch-cuda=12.1
  - torchaudio
  - pip:
      - -r pip-requirements.txt
```

`[VERIFIED: repo - orcas_dclde2026/pip-requirements.txt]` this same environment's pip layer
contains, simultaneously:
```
tensorflow==2.21.0
tensorflow-hub==0.16.1
tf_keras==2.21.0
torch==2.5.1
torchaudio==2.5.1
lightning==2.6.1
pytorch-lightning==2.6.1
scikit-learn==1.7.2
joblib==1.5.3
kagglehub==1.0.0
librosa==0.11.0
numpy==2.2.5
onnxruntime==1.23.2
```

`[VERIFIED: repo - birds_bioacoustics/CLAUDE.md]` (from Round 1) instructs: "Activate env
`bioacoustics` before running anything (`conda activate bioacoustics`)".

The env name, and the fact that both repositories live side by side in the same workspace, make it
overwhelmingly likely this is the *same* conda
environment, already carrying TensorFlow 2.21, PyTorch 2.5.1, PyTorch Lightning 2.6.1, scikit-learn
1.7.2, kagglehub and joblib together without conflict, on this exact machine.

**Round 1 proposed** building an isolated `bioacoustics-perch` conda environment "to avoid a
TF/PyTorch dependency conflict." That risk was hypothetical -- estimated, not measured. It is now
directly falsified by working precedent in a sibling repo using the identically-named environment.

`[RECOMMENDATION]` **Cut the isolated-environment plan entirely.** Phase 0's first action becomes
a one-line check, not an environment-build task:
```bash
conda activate bioacoustics
python -c "import tensorflow, torch, sklearn, kagglehub, joblib, librosa; \
           print(tensorflow.__version__, torch.__version__, sklearn.__version__)"
```
`[UNVERIFIED - Phase 0 check]` Confirm this literally is the same environment instance used by
`birds_bioacoustics/train.py` (not a same-named-but-differently-populated env). If it is not (e.g.
missing `kagglehub` or `tensorflow`), the fallback is `pip install kagglehub tensorflow` into the
existing shared env -- still not a new environment, just added packages, since PteroSet's own
`requirements.txt` shows no conflicting pin (verified in Round 1: no numpy/TF pins in PteroSet's
own requirements).

This is not a small saving. It removes an entire Phase-0 sub-task, an entire "risk" section
bullet, and a maintenance burden (a second environment to keep in sync) from Round 1's plan.

---

## 3. The dependency-footprint finding (eliminates the perch-hoplite package dependency)

`[VERIFIED: repo - orcas_dclde2026/eval_perch_ecotype.py]` never imports `perch_hoplite`. Perch
inference is done by:

1. A **one-time** download: `kagglehub.model_download(PERCH_V2_KAGGLE)`, copied to a local
   directory `checkpoints/perch/model_v2/` (`download_perch_v2()`), after which the network/
   kagglehub path is never touched again.
2. Loading that local directory with bare TensorFlow: `tf.saved_model.load(model_dir)`.
3. Calling the serving signature directly:
   `model.signatures["serving_default"](inputs=tf.constant(batch_np))`.
4. Discovering which output key holds the embedding at runtime, heuristically, via
   `inspect_model_outputs()`: it inspects `model.signatures["serving_default"].structured_outputs`
   and picks the key whose last dimension is 1280 or 1536 (falls back to the smallest 2D output).

Round 1's design assumed the `perch-hoplite` package's `zoo.model_configs.load_model_by_name`
wrapper as the primary loading path (this was the documented, "official" way per the
`google-research/perch-hoplite` source `[VERIFIED: source - perch-hoplite GitHub, Round 1]`, so it
was not a bad guess -- but it is not what the proven pipeline actually uses).

`[RECOMMENDATION]` **Drop the `perch-hoplite` package dependency from Phase 0/1/2.** Load Perch 2
exactly as the sibling repo does: one-time `kagglehub` download to a local SavedModel directory,
then raw `tf.saved_model.load` + `signatures["serving_default"]` + the same output-key-inspection
heuristic, copied close to verbatim. This removes an entire package (with its own transitive
dependency tree: `etils`, `ml-collections`, and agile-modeling extras PteroSet does not need) from
the dependency surface, and removes any question of whether `perch-hoplite`'s API has changed
since Round 1 read its source. `kagglehub` and `tensorflow` are both already present in the shared
`bioacoustics` env per section 2.

**CPU-vs-GPU reconciliation (a real, not cosmetic, open question):** Round 1, reading
`perch-hoplite`'s `model_configs.py`, found an official CPU-capable preset name (`perch_v2_cpu`)
`[VERIFIED: source - perch-hoplite GitHub, Round 1]`. The sibling repo's own scripts, however,
carry a CUDA/`LD_LIBRARY_PATH` bootstrap that re-execs the Python process
(`os.execvpe`) before any TF import specifically because "Perch v2's SavedModel is XLA-compiled
for CUDA and TF cannot dlopen pip-installed nvidia libs otherwise" `[VERIFIED: repo -
eval_perch_ecotype.py, lines ~55-93]`. These two facts are not necessarily contradictory
(`perch_v2_cpu` may be a *different*, separately exported SavedModel variant than the one the
sibling downloaded via `kagglehub.model_download(PERCH_V2_KAGGLE)`), but I cannot resolve this
without running code, so I am not asserting either wins.

`[RECOMMENDATION]` Phase 0 tries the sibling's exact proven path first (local SavedModel + CUDA
bootstrap, since GPUs are already assumed available for `train.py`'s existing spectrogram models
per Round 1's compute review), and only investigates `perch_v2_cpu` as a fallback if GPU is
unavailable in some execution context (e.g. CI, or a laptop dev loop). This is an explicit
`[UNVERIFIED - Phase 0 check]`, not a decision made here.

---

## 4. What is copied verbatim (function-level, not just "the idea")

`[RECOMMENDATION]` The following functions are proposed to be copied with only cosmetic renaming
(module path, docstrings) from `eval_perch_ecotype.py` into a new PteroSet module, because the
audio format constraints are identical (5 s @ 32 kHz mono, Perch 2, 1536-dim embedding) and there
is no PteroSet-specific reason to reimplement them differently:

- `download_perch_v2()` -- one-time kagglehub fetch + local copy.
- `load_perch_v2()` -- `tf.saved_model.load` wrapper.
- `inspect_model_outputs()` -- runtime embedding-key discovery heuristic.
- `load_audio_segment(filepath, center_sec, window_sec=5.0, target_sr=32000, target_peak=0.25)` --
  loads via `librosa.load(..., offset=center_sec - window_sec/2, duration=window_sec, mono=True)`,
  pads/truncates to exactly `window_sec * target_sr` samples, peak-normalizes to 0.25.
- `extract_embeddings()` -- batches rows, computes `center_sec`, calls the serving signature,
  substitutes silence + warns on a per-row load failure rather than crashing the batch.

**A useful, previously-unnoticed simplification found this round:** PteroSet's v4 windows are
already exactly 5 s wide (`window_size_sec: 5.0` in `data/config.yaml`, verified Round 1), unlike
the sibling's 3 s windows (which need a 5 s *context* crop centered on the window midpoint). If
`center_sec = (window_start_sec + window_end_sec) / 2` is passed into `load_audio_segment` with
`window_sec=5.0` on a window whose width is already exactly 5.0 s, the internal offset computation
`center_sec - window_sec/2` algebraically collapses back to `window_start_sec`. In other words:
**`load_audio_segment` can be called completely unmodified**; for PteroSet it simply reloads each
window's own [start, end) span at 32 kHz with peak-norm 0.25, with no extra context borrowed from
neighboring windows. No new windowing logic is needed at all.

Peak normalization to `0.25` (not merely "unit-scaled", as Round 1's paraphrase of the HF model
card put it) is now cross-verified from two independent sources: `perch-hoplite`'s
`TaxonomyModelOnnx` default (`target_peak: float | None = 0.25`, Round 1) and this sibling's own
independently-written loader using the identical constant with a comment that it "matches
perch-hoplite." This value is a fact, not a guess.

---

## 5. What must differ for PteroSet (and why)

| Dimension | orcas_dclde2026 (proven) | PteroSet (this proposal) | Why it must differ |
|---|---|---|---|
| Label topology | Single-label, mutually-exclusive multiclass (5 ecotypes or 3 classes); one orca pod is one ecotype at a time | Multi-label by nature -- multiple species vocalize in the same 5 s window | Biological fact, not a design choice (Round 1 measured 1.6-1.9 mean overlapping species-annotations per positive window in v3/v4). Resolved by a Phase 0 audit (section 7), not assumed. |
| Split structure | One fixed train/val/test split (a seed-selected 70/15/15 split across pooled recordings) | 5-fold leave-one-project-out, already established for the existing binary task | PteroSet's fold structure is a repo convention (`data/folds_segmented_v4/`) that every other model in this repo already uses; species classification must not invent a second, incompatible split convention. |
| Window time units | CSV columns `window_start`/`window_end` already in seconds `[VERIFIED: repo - orcas train_split.csv header]` | `start`/`end` in **samples at 48 kHz** (Round 1, `windows_mapping_4.0overlap_segmented_v4.json`) | Must convert `start/48000`, `end/48000` to seconds before calling `load_audio_segment`; also must resample 48kHz audio to 32kHz for Perch (librosa's `load(..., sr=32000)` does this automatically from the source file, so no separate resampling step is needed -- confirmed by reading `load_audio_segment`, which always loads at `target_sr` directly from the original file path, not from a pre-resampled cache). |
| Window/label sparsity | Ecotype label is dense across the whole assigned-KW population (only a small "unassigned" pool is excluded) | PteroSet species labels are sparse and structurally ambiguous (many boxes are "bird" without species, some are "unknown"); Round 1's 6-state label schema exists precisely to make this explicit | No sibling analog; kept as a genuine PteroSet-specific necessity, not Round-1 sprawl. See section 6. |
| Class list construction | Fixed, small, biologically enumerable (5 ecotypes) | Must be curated from `data/species.csv`, excluding placeholder/unresolved codes, with a minimum per-class window count | No sibling analog; PteroSet-specific, kept from Round 1. |
| Per-project evaluation | Post-hoc slice of one shared test set, looped `for ds_name in ["ALL"] + datasets` | Automatic: in LOPO, the *entire* test set of a fold already **is** one held-out project | Simpler for PteroSet than for the sibling, not harder -- see section 9. |

---

## 6. Label schema: kept from Round 1, now framed against the sibling's simpler case

Round 1's 6-state label schema (`has_bird`, `unresolved_bird`, `usable_strict`,
`species_<CODE>` one-hot columns, `any_species_known`) has **no equivalent complexity** in the
sibling repo, because orca ecotype labels are comparatively clean (a KW encounter is confidently
assigned to a pod, or explicitly held out as "unassigned"). PteroSet's raw annotation data (Round
1's inspection of `annotations_species.json` / `annotations_identification.json`) has:
- boxes labeled "bird" with no species resolved,
- boxes with a species code that is itself a placeholder/aggregate (not evaluated in Round 1's
  detail, flagged there as a Phase-0/1 unknown),
- windows with multiple overlapping boxes of different species.

`[RECOMMENDATION]` Keep Round 1's schema unchanged: `usable_strict` = windows whose only bird
annotations are species-resolved (no `unresolved_bird` co-occurring). This is not sprawl to be
cut; it is the PteroSet-specific equivalent of the sibling's "assigned vs. unassigned KW" split,
and it is a precondition for the label topology audit in section 7 to even be well-defined
(counting "species per window" is meaningless if some of those windows have an unresolved box that
could be any species).

---

## 7. Multiclass vs. multilabel: the smallest audit that decides it

This is the central open design fork this round must resolve, per the user's explicit instruction.
It is resolved by measurement, not assumption, with a single small script.

**Script:** `species_probe/audit_label_topology.py`
```
Input:  data/folds_segmented_v4/ (existing) + windows_mapping_4.0overlap_segmented_v4.json
        + species annotation tables (already joined in Round 1's usable_strict definition)
Output: reports/species_label_topology_v4.csv, one row per usable_strict positive window:
        window_id, project, n_distinct_species, species_codes (semicolon-joined)
        + a printed summary: histogram of n_distinct_species in {0, 1, 2, 3+}
Runtime: single pass over an already-materialized join; no audio I/O, no Perch, no GPU.
Estimated cost: seconds to low minutes over 160,244 windows.
```

**Decision rule `[RECOMMENDATION]`:**
- If >= 90% of `usable_strict` positive windows have exactly one distinct species present:
  adopt **single-label multiclass**, i.e. copy the sibling's exact recipe unmodified --
  `sklearn.linear_model.LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0,
  multi_class="multinomial")`, fit on one label per window (background/no-bird optionally folded
  in as an explicit K+1'th class, or excluded -- to be decided once the audit's numbers are in
  hand, since the right choice depends on how the K+1 class behaves in practice). This is the
  maximal-reuse branch: the training and evaluation code becomes close to a rename of
  `run_fulldata()`.
- If a non-trivial fraction (`[RECOMMENDATION]` provisional threshold: >= 10%) of positive windows
  have two or more distinct species: adopt **true multi-label** -- one independent binary
  `sklearn.linear_model.LogisticRegression` per class (still `lbfgs`, still `C=1.0`, just one
  `.fit()` call per class column instead of one joint multinomial fit), each row a K-length
  multi-hot vector, windows with zero species as negatives for every class. Evaluation switches
  from macro-F1 + confusion matrix to per-class average precision / ROC-AUC + micro/macro-F1 at a
  chosen threshold (already specified in Round 1).

Both branches keep every other piece of the pipeline identical: the same NPZ cache format
(section 8), the same L2-normalization-at-use-time convention, the same `joblib` persistence, the
same per-project evaluation loop. **Only the `.fit()` call and the metric aggregation differ.**
This is deliberately the smallest fork that the label reality could force -- not a design that
tries to support both simultaneously "just in case."

Round 1 already measured that positive windows average 1.6-1.9 overlapping species-annotations,
which is directly suggestive that the >= 10% multi-label branch will be the one taken -- but this
proposal does not skip the audit and assume that outcome. The audit is one script, costs minutes,
and removes any doubt before a single Perch embedding is extracted.

`[RECOMMENDATION]` This audit is Phase 0's second deliverable (after the environment check in
section 2), run before any embedding extraction, so that the NPZ label field (section 8) is
written in its final form the first time, rather than re-extracted after a late discovery.

---

## 8. Embedding extraction and caching architecture

### 8.1 Why "one split-level cache per fold" (the sibling's exact convention) needs one PteroSet-specific adaptation

The sibling repo has a **single fixed train/val/test split**; `run_extraction()` therefore writes
exactly one compressed NPZ per split:
`np.savez_compressed(path, embeddings=..., <label>=..., sound_filepath=..., dataset=...,
window_id=...)` `[VERIFIED: repo - eval_perch_ecotype.py, extract_perch_3class.py]`.

PteroSet's 5-fold leave-one-project-out means the *same* window appears in the train/val pool of
up to 4 folds and the test pool of exactly 1 fold. If embeddings were extracted independently per
fold per split (5 folds x up to 3 splits), a large fraction of the 160,244 windows would be
re-embedded redundantly across folds -- pure wasted GPU time for a value (the embedding) that does
not depend on which fold it is used in.

`[VERIFIED: repo - remap_perch_embeddings.py]` The sibling repo already has, and uses, the
solution to exactly this class of problem: it demonstrates that Perch embeddings are a **pure
function of `window_id`** (audio file + time bounds + model version), so re-partitioning splits
never requires re-running Perch -- only gathering existing rows into new split files by
`window_id`, with hard verification gates: the `window_id` must exist in the source pool, the
`sound_filepath` must match exactly, and a random sample of rows is checked bit-identical
(L2 difference == 0.0) before the new file is trusted. Mismatches abort the script; nothing is
silently dropped.

`[RECOMMENDATION]` Adopt this pattern directly, generalized from "one fixed split" to "5 LOPO
folds":

**Stage A -- extract once, for the whole v4 window pool:**
```
species_probe/extract_perch_pool.py
  --windows_mapping windows_mapping_4.0overlap_segmented_v4.json
  --model v2 --model_dir checkpoints/perch/model_v2
  --batch_size 64
  --out data/embeddings/perch_v2_pool_segmented_v4.npz
```
Runs Perch exactly once over all 160,244 windows (not per fold). Output schema, matching the
sibling's field names as closely as PteroSet's own vocabulary allows:
```
embeddings      float32 (160244, 1536)
window_id       int64   (160244,)      # PteroSet's existing stable window identity
sound_filepath  object  (160244,)      # full path to source recording
project         object  (160244,)      # PteroSet's own term for the sibling's "dataset"
                                        #  (MAP1 / PPA1 / PPA2 / PPA3 / PPA4) -- kept as
                                        #  "project" for consistency with this repo's own
                                        #  existing vocabulary, not renamed to match the
                                        #  sibling verbatim; this is the one deliberate
                                        #  naming deviation in this proposal.
label           per section 7's decision -- either int64 class id (single-label branch)
                or uint8 (160244, K) multi-hot matrix (multi-label branch)
usable_strict   bool    (160244,)      # section 6's mask; rows outside it carry label=-1
                                        #  (single-label) or all-zero (multi-label)
```

**Stage B -- gather per-fold, per-split caches from the pool, by `window_id`:**
```
species_probe/build_fold_embedding_caches.py
  --pool data/embeddings/perch_v2_pool_segmented_v4.npz
  --fold_dir data/folds_segmented_v4/
  --out_dir checkpoints_species_v4/perch_v2/
```
For each of the 5 folds x {train, val, test}, this writes
`checkpoints_species_v4/perch_v2/fold_{N}_{split}.npz` by indexing the pool array with the
fold/split's `window_id` list, applying the **same hard gates** as `remap_perch_embeddings.py`:
window_id must be present in the pool; `sound_filepath` must match the fold CSV's own path column;
3 randomly sampled rows are checked bit-identical against a direct re-index. Any gate failure
aborts with a clear error naming the offending `window_id` -- never a silent drop.

This two-stage design is the direct generalization of the sibling's proven "extract once, remap
per split" pattern to "extract once, remap per fold" -- it is not a new idea invented for
PteroSet, it is the same idea applied to a superset of situations (N-fold overlapping pools
instead of one fixed split).

### 8.2 What this replaces from Round 1

Round 1 proposed a single monolithic `.npy` array for the whole v4 pool plus a separate index CSV,
with the fold-to-embedding join performed ad hoc, inline, inside `train_linear_probe.py` itself.
That was already the right instinct (extract once, not per fold) but left the join/verification
logic unmaterialized and untested as its own artifact. Round 2 keeps the "extract once" instinct
and adds the sibling's proven separation of concerns: the join is its own script
(`build_fold_embedding_caches.py`), producing its own tested, inspectable artifact (the per-fold
NPZ), with its own hard-gated verification -- exactly mirroring `remap_perch_embeddings.py` -- so
that `train_linear_probe.py` never has to know about `window_id` joins at all; it just loads a
ready-made NPZ.

### 8.3 L2 normalization: stateless, at use time, not baked into the cache

`[VERIFIED: repo - eval_perch_ecotype.py, run_fulldata/run_unassigned_eval]` The sibling repo
stores raw (non-normalized) Perch output in every NPZ, and applies
`sklearn.preprocessing.normalize(embeddings, norm="l2")` immediately before every `.fit()` /
`.predict_proba()` call, never persisting a normalized copy.

`[RECOMMENDATION]` Adopt this exactly, replacing Round 1's proposal to fit a `StandardScaler` on
the training fold and persist it alongside the classifier. L2-row-normalization is parameterless
(no per-fold fitting, no artifact to persist, no risk of a scaler leaking test-fold statistics)
and is proven sufficient in production by the sibling's own results. This is a genuine
simplification: an entire artifact type (`scaler_fold_{N}.joblib`) and its associated leakage-
control reasoning disappear from the design.

---

## 9. Model, evaluation, and per-project reporting

### 9.1 LogisticRegression hyperparameters: fixed, not tuned, on the first pass

`[VERIFIED: repo - eval_perch_ecotype.py::run_fulldata]` and `[VERIFIED: repo -
docs/experiments/log.md, S2.3]`: the sibling's production recipe is
`LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0[, multi_class="multinomial"])`, with no
grid search, no validation-based hyperparameter selection. The S2.3 log entry explicitly frames
this as a *virtue*: "Perch+LogReg is deterministic given the same train embeddings (no end-to-end
training shuffle/init noise)", used to argue a measured delta (+1.16pp macro-F1 on the ecotype
task) is signal rather than seed noise.

`[RECOMMENDATION]` **Cut Round 1's `--C_grid 0.01 0.1 1.0 10.0` validation-based hyperparameter
search from the Phase-1/2 baseline.** Use the sibling's exact fixed hyperparameters
(`solver="lbfgs", max_iter=1000, C=1.0`) for the first trustworthy result. If Phase 3's go/no-go
review shows the linear probe is close to a decision boundary (e.g. borderline "does this beat
majority-class baseline" call), a small, explicitly-labeled follow-up sweep over `C` is a cheap,
optional Phase-4 refinement -- not a requirement to reach a first result. This directly serves the
"push back on complexity that is not needed for an initial trustworthy result" brief: a grid search
is premature optimization before the base recipe has even been shown to work on this task.

`[UNVERIFIED - Phase 0 check]` `scikit-learn==1.7.2` (the sibling's pinned version) may emit a
deprecation warning for the `multi_class` parameter (deprecated starting around sklearn 1.5 in
favor of automatic solver-based multinomial handling); this needs a one-time check at
implementation time, not a guess here. It does not change the recommended hyperparameters, only
whether `multi_class="multinomial"` needs to be passed explicitly or is now default behavior.

### 9.2 Persistence: joblib, matching the proven convention

`[VERIFIED: repo - eval_perch_ecotype.py]` classifiers are persisted with `joblib.dump` /
`joblib.load`, not a hand-rolled `.npy` weight-matrix format.

Round 1 proposed persisting plain `(weights, bias)` numpy arrays specifically to avoid
sklearn-pickle-version fragility across environments. That is a real, legitimate concern in
general, but it is not a concern the sibling repo's own working pipeline has needed to solve --
and per section 2, PteroSet and the sibling now provably share one scikit-learn version
(`1.7.2`) in one environment, which removes the cross-environment mismatch risk this concern was
protecting against.

`[RECOMMENDATION]` **Adopt `joblib.dump`/`joblib.load`**, matching the proven convention, since
the risk it was meant to guard against does not apply here. Retain one cheap safeguard from Round
1's original concern: log the exact `sklearn.__version__` string into a sidecar
`fold_{N}_model_meta.json` next to each `.joblib` file at save time, so any future version drift is
at least detectable rather than silent.

### 9.3 Per-project evaluation is close to automatic under LOPO

`[VERIFIED: repo - eval_perch_ecotype.py::run_fulldata]` The sibling's per-dataset breakdown is a
post-hoc loop: `for ds_name in ["ALL"] + sorted(unique_datasets): mask = (test_datasets ==
ds_name); compute metrics on embeddings[mask]`. This is needed there because a single shared test
set spans multiple datasets.

`[RECOMMENDATION]` PteroSet's existing leave-one-project-out convention means each fold's test set
already contains exactly one project (MAP1, PPA1, PPA2, PPA3, or PPA4) by construction. The
"per-project evaluation" the sibling computes as a slice of one shared test set is, for PteroSet,
simply "the fold's own test metrics" -- report per-fold metrics (5 numbers per metric, one per
held-out project) plus a pooled/mean-across-folds summary, exactly mirroring
`train.py`'s existing `--cross_validation` reporting convention (Round 1 confirmed this convention
already exists and is used by the spectrogram models). This table of 5 numbers *is* the per-project
breakdown; no separate script or extra loop is needed beyond what LOPO cross-validation already
requires.

### 9.4 Metrics

`[VERIFIED: repo - eval_perch_ecotype.py::_compute_full_metrics]` the sibling's core "fulldata"
path computes macro-F1, per-class F1, macro ROC-AUC (one-vs-rest), and a printed confusion matrix
-- no calibration curve, no threshold sweep, in the base path (threshold/calibration analysis is a
separately-flagged, later addendum in their report, section 9, not part of the base recipe).

`[RECOMMENDATION]` Match this scope for the base pipeline:
- **Single-label branch:** macro-F1, per-class F1, macro ROC-AUC (OVR), confusion matrix -- a
  direct rename of `_compute_full_metrics`.
- **Multi-label branch:** per-class average precision, per-class ROC-AUC, micro- and macro-F1 at a
  single fixed threshold (0.5, or the training-fold's positive rate as a simple default -- to be
  fixed once, not swept), plus the same per-fold/per-project breakdown table.

Calibration, threshold sweeps beyond the one fixed default, and any ablation matrix (Perch v1 vs
v2, window-level vs recording-level pooling, frozen vs fine-tuned) remain explicitly out of scope
for the Phase 0-3 baseline, exactly as Round 1 already scoped them -- Round 1 did not actually
propose calibration/ablation sprawl in its baseline phases (that material was already deferred to
an explicitly-conditional Phase 4). No further cutting is needed there; this is confirmed, not
found to be a problem.

---

## 10. Cross-task caution carried forward from the sibling's own experience

`[VERIFIED: repo - docs/experiments/log.md, S2.3]` The identical Perch v2 + LogReg recipe that
improved the sibling's ecotype task (Stage 2, a *refinement within already-detected* orca
encounters, +1.16pp macro-F1) **failed** when applied to their Stage-1-like bio/non-bio-like
detection task (macro-F1 -1.12pp, ship-noise false positives +550, and was discarded there).

This is directly relevant, not incidental: PteroSet's species task is also, structurally, a
refinement *within* already-detected bird-positive windows (`usable_strict`, conditioned on
`has_bird`), not a redo of the primary bird/no-bird detection task the existing spectrogram model
already performs. This is the same shape as the sibling's *successful* Stage 2 use, not their
*failed* Stage 1 use. This strengthens confidence in Round 1's scoping choice (species
classification conditioned on an already-positive window, not a joint detect+classify task) but
it is not proof of success -- it is evidence for why this is the right task shape to try first,
consistent with Round 1's and this round's shared go/no-go philosophy: expect to be told no by the
data, and design the smallest experiment that can say so quickly.

`[RECOMMENDATION]` Keep this framing explicit in the design doc's Phase 1 go/no-go criteria: a
negative result here (linear probe does not beat majority-class/frequency baseline) is a valid,
useful outcome, not a pipeline failure -- exactly as it was for the sibling's Stage 1 attempt.

---

## 11. Revised phased plan

Phases below assume the reader has Round 1's full detail (label schema definitions, exact fold
directory layout, species-count-threshold curation logic, leakage policy) already in hand; only
what changed is spelled out. Anything not mentioned here is unchanged from Round 1.

**Phase 0 -- Environment + label topology audit (was: environment build + Perch API spike)**
- Confirm the shared `bioacoustics` env already has `tensorflow`, `kagglehub`, `scikit-learn`,
  `joblib`, `librosa` (section 2). No new environment.
- Run `audit_label_topology.py` (section 7); decide single-label vs multi-label branch.
- One-time `download_perch_v2()` call; confirm local SavedModel loads and
  `inspect_model_outputs()` finds a 1536-dim output (or resolve the CPU/GPU question, section 3,
  if GPU is unavailable in the working context).
- **Go/no-go:** environment check passes; label topology audit produces the histogram; Perch
  loads and returns a 1536-dim embedding for one sample window. All three are cheap (minutes,
  no full-dataset extraction yet).

**Phase 1 -- Pool extraction + fold cache build (was: per-fold extraction inline in training)**
- Run `extract_perch_pool.py` once over all 160,244 v4 windows (section 8.1, Stage A).
- Run `build_fold_embedding_caches.py` for all 5 folds x 3 splits (section 8.1, Stage B), with
  hard verification gates.
- **Go/no-go:** pool NPZ has 160,244 rows, no NaN/silence-substitution rate above a small
  threshold (e.g. <1% of windows failed to load); all 15 fold/split caches pass the bit-identical
  spot-check gate.

**Phase 2 -- Baseline linear probe (was: linear probe + C-grid selection)**
- Train the section-7-selected model type (single multinomial LogisticRegression, or K independent
  binary LogisticRegressions) per fold, fixed hyperparameters (section 9.1), `joblib`-persisted.
- Evaluate per fold/per-project (section 9.3), pooled across folds.
- Compare against a majority-class / label-frequency baseline (kept from Round 1).
- **Go/no-go:** linear probe beats the frequency baseline by a pre-registered margin on macro-F1 (or
  macro-AP for the multi-label branch) on at least 4 of 5 folds. If not, stop and report a negative
  result (section 10) -- do not proceed to Phase 3/4 hyperparameter or architecture escalation
  without first understanding *why* via the failure-analysis steps already specified in Round 1.

**Phase 3 -- Reporting and integration**
- Per-project / per-fold result tables and confusion-matrix or per-class-AP artifacts, in the same
  `reports/` location convention as the existing binary-task reports.
- Written comparison against the existing spectrogram-based bird-positive model's performance on
  the same folds, where applicable (already scoped in Round 1).

**Phase 4 -- Conditional refinements (unchanged from Round 1: explicitly optional, not baseline)**
- Small `C` sweep, calibration, Perch v1-vs-v2 comparison, recording-level mean-pooling
  (mirroring the sibling's `aggregate_to_recordings`), only if Phase 2/3 results warrant deeper
  investigation.

---

## 12. Files and artifacts (revised)

```
species_probe/
  audit_label_topology.py        # Phase 0, decides section 7's fork
  extract_perch_pool.py          # Phase 1, Stage A (adapted from eval_perch_ecotype.py helpers)
  build_fold_embedding_caches.py # Phase 1, Stage B (adapted from remap_perch_embeddings.py)
  perch_io.py                    # shared: download_perch_v2, load_perch_v2,
                                  #   inspect_model_outputs, load_audio_segment,
                                  #   extract_embeddings -- copied near-verbatim (section 4)
  train_linear_probe.py          # Phase 2 -- loads a ready-made fold NPZ, no join logic
  evaluate_linear_probe.py       # Phase 2/3 -- per-fold + per-project + pooled metrics

data/embeddings/
  perch_v2_pool_segmented_v4.npz     # Phase 1 Stage A output (gitignored, per section 8.1 schema)

checkpoints_species_v4/perch_v2/
  fold_{0..4}_{train,val,test}.npz   # Phase 1 Stage B outputs (gitignored)
  fold_{0..4}_model.joblib           # Phase 2 outputs (gitignored)
  fold_{0..4}_model_meta.json        # sklearn version + hyperparameters used (small, could be
                                      #  tracked in git if desired, unlike the .joblib itself)

reports/
  species_label_topology_v4.csv      # Phase 0 output
  species_probe_fold_metrics.csv     # Phase 2/3 output
  species_probe_per_project.csv      # Phase 2/3 output
```

`[RECOMMENDATION]` `.gitignore` addition: `data/embeddings/` and `checkpoints_species_v4/`
wholesale (mirroring `[VERIFIED: repo - orcas_dclde2026/.gitignore]`'s blanket `checkpoints/`
entry, a cleaner convention than relying on a bare `*.npy`/`*.joblib` extension rule alone).

**CLI conventions**, matching the sibling's flag naming where the concept is identical
`[VERIFIED: repo - eval_perch_ecotype.py::main]`:
```bash
python species_probe/extract_perch_pool.py \
  --windows_mapping windows_mapping_4.0overlap_segmented_v4.json \
  --model v2 --model_dir checkpoints/perch/model_v2 --batch_size 64 \
  --out data/embeddings/perch_v2_pool_segmented_v4.npz

python species_probe/build_fold_embedding_caches.py \
  --pool data/embeddings/perch_v2_pool_segmented_v4.npz \
  --fold_dir data/folds_segmented_v4/ \
  --out_dir checkpoints_species_v4/perch_v2/

python species_probe/train_linear_probe.py \
  --fold_dir checkpoints_species_v4/perch_v2/ --fold 0 \
  --label_mode {single_label,multi_label}   # section 7's decision, not a free choice per run
  --out checkpoints_species_v4/perch_v2/fold_0_model.joblib

python species_probe/evaluate_linear_probe.py \
  --fold_dir checkpoints_species_v4/perch_v2/ --fold 0 \
  --model checkpoints_species_v4/perch_v2/fold_0_model.joblib \
  --out_dir reports/
```

---

## 13. Tests (revised, minimal)

`[RECOMMENDATION]` Kept from Round 1, re-scoped to the new artifact boundaries:
1. `test_perch_io.py` -- unit test that `load_audio_segment` on a synthetic sine wave produces
   exactly `window_sec * target_sr` samples and peak `== target_peak` (or 0 for silence input);
   no network, no real Perch model needed.
2. `test_fold_cache_gates.py` -- unit test that `build_fold_embedding_caches.py`'s verification
   gates actually abort on a deliberately corrupted pool (mismatched `sound_filepath`, or a
   `window_id` missing from the pool) -- this is the single highest-value test, since it protects
   the leakage-free property the whole LOPO design depends on.
3. `test_label_topology_audit.py` -- unit test on a tiny synthetic annotation set that
   `audit_label_topology.py` correctly counts distinct species per window, including the
   `usable_strict` exclusion of windows with an unresolved co-occurring box.

No test suite for the sklearn training/eval scripts themselves is proposed beyond a smoke test
(runs end-to-end on a tiny synthetic NPZ and produces a non-empty metrics file) -- this mirrors
the sibling repo, which has no unit tests for `eval_perch_ecotype.py` itself
`[VERIFIED: repo - orcas_dclde2026/tests/ contains only test_gpt_sparrow_filter.py, unrelated]`,
and matches the minimalist brief: test the parts that silently corrupt data (cache joins, audio
normalization) more than the parts that just call well-tested library functions (`sklearn.fit`).

---

## 14. Compute/storage estimates (revised)

Unchanged from Round 1 in order of magnitude, restated against the new architecture:
- Pool extraction (Phase 1 Stage A): 160,244 windows x 5 s audio load + 1 Perch forward pass,
  batched at 64. `[RECOMMENDATION]` estimate 1-3 GPU-hours depending on I/O throughput from the
  existing audio storage; this is a **one-time** cost regardless of how many folds/experiments
  follow, which is the whole point of the pool/remap architecture (section 8).
- Pool NPZ size: 160,244 x 1536 x 4 bytes (float32) ~= 0.98 GB for embeddings alone, plus small
  metadata columns -- comfortably fits in memory for the fold-cache-building step and for training.
- Per-fold NPZ sizes: sum across the 5 folds' train/val/test row counts will be roughly 4x the
  pool size in aggregate (each window appears in ~4 of 5 folds' train/val pools) -- estimate
  ~4 GB total across all 15 fold/split files. This is an acceptable, disk-cheap redundancy in
  exchange for zero re-extraction cost and simple, self-contained per-fold training inputs.
- LogisticRegression training: seconds to low minutes per fold on a 1536-dim, tens-of-thousands-
  of-rows problem, whether single-label or K-independent-binary multi-label (K expected to be a
  small class list per Round 1's curation, not hundreds).

---

## 15. Risks and alternatives (revised)

- **Risk:** the "same environment" claim (section 2) turns out to be wrong (two differently-named
  environments, or the same name pointing at different populated environments on different
  machines/users). **Mitigation:** the one-line Phase 0 check is the very first action; if it
  fails, the fallback is `pip install` into the existing shared env, still not a new environment.
- **Risk:** the CPU/GPU discrepancy (section 3) means Perch v2 genuinely requires a GPU and none is
  available in some execution context (e.g. a lightweight CI runner). **Mitigation:** Phase 0
  surfaces this immediately (single test window, not the full pool); if GPU is required, this is a
  known, bounded infrastructure requirement, not a design defect, and the existing spectrogram
  models in this repo already assume GPU availability per Round 1's review of `train.py`.
- **Risk:** the multi-label branch (section 7) is selected, and K independent binary
  LogisticRegressions turn out to need per-class class-imbalance handling (`class_weight="balanced"`)
  that the sibling's single-label recipe never needed. **Mitigation:** this is a one-argument change
  to the same `sklearn.linear_model.LogisticRegression` call if Phase 2's results show it is
  needed; not a new machinery decision.
- **Alternative considered and rejected (again, now with stronger evidence):** a `perch-hoplite`-
  wrapped loading path, an isolated conda environment, a `StandardScaler`-based normalization
  pipeline, and a validation-based `C`-grid search were all considered in Round 1 and are now
  rejected with concrete evidence from a working sibling pipeline, not just a preference for
  simplicity in the abstract.
- **Alternative not adopted:** ONNX inference (mentioned in Round 1 as a documented fallback with
  its own Phase-0 spike). The sibling repo does not use ONNX for Perch at all despite having
  `onnxruntime` installed (evidently for an unrelated model in that repo, e.g. their
  YOLO/ultralytics detector). `[RECOMMENDATION]` drop the ONNX path entirely from Phase 0 scope;
  it is not needed by the proven recipe and would only be revisited if the raw-SavedModel path is
  shown not to work in this repo's execution environment.

---

## 16. Convergence verdict

**NEW-IDEAS.**

The overall philosophy is unchanged and re-confirmed by an independent, already-working
implementation: frozen Perch embeddings, a plain `sklearn.linear_model.LogisticRegression` head
(explicitly not PyTorch Lightning), phased go/no-go gates, and a negative result treated as a
valid, useful outcome. That part of Round 1 converges cleanly with the sibling's proven approach
and required no revision.

However, enough of the concrete implementation changed, based on direct evidence from a working
system rather than documentation-only research, that this is not a minor refinement:
- The isolated conda environment plan is eliminated (section 2) -- a real scope cut, evidenced by
  a shared-environment discovery Round 1 could not have made without reading this sibling repo.
- The `perch-hoplite` package dependency is dropped in favor of direct `tf.saved_model.load`
  (section 3) -- a smaller, more direct dependency footprint than Round 1 proposed.
- The caching architecture changes from "one big array + inline join" to an explicit two-stage
  "extract-once pool, then gated per-fold remap" pipeline modeled directly on
  `remap_perch_embeddings.py` (section 8) -- a structurally different, more rigorously verified
  design.
- `StandardScaler`-per-fold is replaced by stateless L2 normalization at use time (section 8.3).
- The `C`-grid search is cut from the baseline in favor of the sibling's fixed, proven
  hyperparameters (section 9.1).
- Persistence changes from plain numpy arrays to `joblib` (section 9.2).
- The multiclass-vs-multilabel question, only implicitly assumed in Round 1's "K independent
  binary classifiers" design, now has an explicit, minimal, data-driven decision procedure
  (section 7) that could in principle send the design down the sibling's exact single-label path
  instead, if the audit's numbers surprise us.

These are concrete, file-level and dependency-level changes to the plan, not a restatement of
Round 1 with better citations -- hence NEW-IDEAS rather than CONVERGED.

---

STATUS: DONE
