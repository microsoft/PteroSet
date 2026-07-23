# Linear Probing on Perch 2 Embeddings for Species Classification in PteroSet — Round 2

**Author role**: Chief Architect (architect-pipeline), Design Round 2
**Status**: Revision of Round 1 (`docs/design/round_01/architect_pipeline_proposal.md`)
**Trigger**: User directive to align with the sibling repo `../orcas_dclde2026`, treating
`eval_perch_ecotype.py`, `extract_perch_3class.py`, and `remap_perch_embeddings.py` as the reference
implementation, and to reuse/copy those patterns rather than invent a parallel framework.

## Executive summary

Round 1 designed a plausible but self-invented pipeline: `perch_hoplite` package, per-window `.npy` cache with a SHA-256 manifest, PyTorch Lightning linear-probe training. Round 2 replaces that with the **already-running, empirically-proven** pattern from `orcas_dclde2026`: a raw `tf.saved_model.load()` on a locally-vendored SavedModel, a CUDA-library re-exec bootstrap, per-split `.npz` embedding caches, a `remap`-style window-id gather with hard integrity gates, and `sklearn` LogisticRegression + `joblib`. This is a **substantial pivot in implementation substrate**, not a refinement — see §9 for the convergence verdict.

What does **not** change from Round 1: the label-provenance/multilabel/leakage-control design (§4 of Round 1) is still correct and is carried forward essentially unmodified, because it is orthogonal to which library extracts embeddings or which classifier framework fits the head.

---

## 1. Reference implementation — verified facts from `orcas_dclde2026`

All of the following were read directly from the sibling repo's source in this session (not inferred).

| Fact | Detail | Source |
|---|---|---|
| No `perch_hoplite` dependency | Neither `pip-requirements.txt` nor any of the three reference scripts import `perch_hoplite`. Perch is loaded as a **raw TensorFlow SavedModel** | `[VERIFIED]` `pip-requirements.txt`; `eval_perch_ecotype.py` |
| Pinned versions in production use | `tensorflow==2.21.0` (stable, not the `2.20.0rc0` release-candidate the HuggingFace card mentioned), `tensorflow-hub==0.16.1`, `kagglehub==1.0.0`, `kagglesdk==0.1.18`, `scikit-learn==1.7.2`, `joblib==1.5.3`, `numpy==2.2.5` | `[VERIFIED]` `orcas_dclde2026/pip-requirements.txt` |
| Download-once, vendor locally | `download_perch_v2()` calls `kagglehub.model_download(...)` **once**, then copies the SavedModel tree (`saved_model.pb`, `variables/`, `assets/`) into a repo-local dir (`checkpoints/perch/model_v2/`, confirmed to contain a standard SavedModel layout on disk). All subsequent loads are `tf.saved_model.load(local_dir)` — zero further network/Kaggle dependency after the first download | `[VERIFIED]` `eval_perch_ecotype.py::download_perch_v2/load_perch_v2`; directory listing of `orcas_dclde2026/checkpoints/perch/model_v2/` |
| GPU/CUDA is empirically required | A code comment states: *"Perch v2's SavedModel is XLA-compiled with `platforms=[CUDA]`, so it refuses to run on CPU."* A `_ensure_gpu_env`/`_ensure_cuda_libs_in_ldlibpath` bootstrap re-execs the Python process (`os.execvpe`) with `LD_LIBRARY_PATH` pointed at the pip-installed `nvidia-*` package libs **before any TF import**, because the dynamic linker caches search paths at process start and setting `os.environ` after import has no effect | `[VERIFIED — as coded and commented in this repo]`. This is evidence about the **`perch_v2` (GPU-targeted) Kaggle slug specifically** (`google/bird-vocalization-classifier/tensorFlow2/perch_v2`, confirmed by the `PERCH_V2_KAGGLE` constant) — a **different artifact** from `perch_hoplite`'s separate `perch_v2_cpu` slug that Round 1 found in `perch_hoplite/zoo/kaggle_hub.py`. Neither repo has empirically exercised `perch_v2_cpu`; do not conflate the two. |
| Model output introspection | `inspect_model_outputs()` calls `model.signatures["serving_default"]`, prints all `structured_outputs`, and picks the embedding key heuristically (rank-2 output with last-dim in `{1280, 1536}`, or the smaller of two large outputs) because a raw SavedModel's output *names* (`output_0`, `output_1`, ...) are not guaranteed stable across model versions | `[VERIFIED]` `eval_perch_ecotype.py::inspect_model_outputs` |
| Forward pass | `model.signatures["serving_default"](inputs=tf.constant(batch_np))`, batched, `out[emb_key].numpy()` | `[VERIFIED]` `eval_perch_ecotype.py::extract_embeddings` |
| Audio segment extraction | `load_audio_segment(filepath, center_sec, window_sec=5.0, target_sr=32000)`: computes `offset = max(0, center_sec - window_sec/2)`, then a **single** `librosa.load(filepath, sr=target_sr, offset=offset, duration=window_sec, mono=True)` call does partial-file decode **and** resample in one step (no separate whole-file resample pass); pads/truncates to exactly `window_sec*target_sr` samples; applies manual peak normalization to `target_peak=0.25` (matching `perch-hoplite`'s documented default, confirmed independently in Round 1) | `[VERIFIED]` `eval_perch_ecotype.py::load_audio_segment` |
| Centered-window convention | `center_sec = (window_start + window_end) / 2.0`, then the loader re-derives a clean, fixed-duration 5 s window around that center — robust to source windows that are not already exactly 5 s (true for orcas' 3 s windows; a no-op offset for PteroSet's already-5 s windows, but the *pattern* is what we copy) | `[VERIFIED]` `eval_perch_ecotype.py::extract_embeddings` (`center_sec` computation) |
| `window_start`/`window_end` units | **Seconds**, not sample indices (`center_sec = (row.window_start + row.window_end) / 2.0` is a plain float average) — differs from PteroSet's fold CSVs, whose `start`/`end` are **sample indices at 48 kHz**. Must convert at the pool-CSV construction step (§5) | `[VERIFIED]` `eval_perch_ecotype.py::extract_embeddings` |
| Storage format | **One `.npz` per split**, not one file per window: `checkpoints/perch/ecotype_emb_{v1,v2}_{train,val,test}.npz` and `checkpoints/perch/3class_emb_{v1,v2}_{train,val,test}.npz`, each containing parallel arrays `embeddings (N,D)`, a label array, `sound_filepath (N,)`, `dataset (N,)`, `window_id (N,)` | `[VERIFIED]` `eval_perch_ecotype.py::run_extraction`; `extract_perch_3class.py`; directory listing |
| Resumability granularity | **Split-level, not window-level.** `extract_perch_3class.py` checks: if the output `.npz` exists and `len(existing["embeddings"]) == len(df)` (row-count match against the source CSV), skip the whole split; otherwise re-extract the **entire** split from scratch. No per-window incremental resume, no content hash, no atomic tmp-then-rename | `[VERIFIED]` `extract_perch_3class.py::main` |
| Cross-partition reuse without re-extraction | `remap_perch_embeddings.py`: embeddings are a **pure function of `window_id`** (audio + bounds + model), so when a window pool is **re-partitioned** into a new train/val/test split, the old embeddings are gathered into the new partition **by `window_id`** instead of re-running Perch (`"the extraction takes hours; this takes seconds and is bit-identical"`). Two **hard gates** before trusting the gather: (1) every `window_id` in the new split must exist in the old pool, or `SystemExit` — no silent drops; (2) the old pool's `sound_filepath` for each matched `window_id` must equal the new split's `sound_filepath` for that row, or `SystemExit` — catches identity drift. After gathering, a **self-verification** step re-checks 3 random rows for exact (`L2 == 0.0`) equality against the source pool | `[VERIFIED]` `remap_perch_embeddings.py::load_old_pool/remap_model` |
| Classifier | `sklearn.linear_model.LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, multi_class="multinomial")` fit directly on L2-normalized embeddings (`sklearn.preprocessing.normalize(..., norm="l2")`); **no PyTorch, no Lightning, no training loop, no epochs** for the frozen-embedding head | `[VERIFIED]` `eval_perch_ecotype.py::run_fulldata/run_fewshot` |
| Persistence | Fitted classifier saved with `joblib.dump(clf, "logreg_ecotype_{version}.joblib")` | `[VERIFIED]` `eval_perch_ecotype.py::run_fulldata` |
| Metrics & outputs | `sklearn.metrics.{f1_score, roc_auc_score(multi_class="ovr", average="macro"), confusion_matrix}`; results written as flat CSVs (`fulldata_results_{v}.csv`, `fulldata_per_dataset_{v}.csv`, `fewshot_results_{v}.csv`) — plain `pandas.DataFrame.to_csv`, no other artifact format | `[VERIFIED]` `eval_perch_ecotype.py::_compute_full_metrics/run_fulldata` |
| Recording-level aggregation | `aggregate_to_recordings()`: mean-pool all window embeddings sharing a `sound_filepath`, L2-normalize, majority-vote the (single, mutually-exclusive) label — used because orcas' ecotype/3-class labels are one-per-recording-ish and both window- and recording-level results are scientifically interesting to report | `[VERIFIED]` `eval_perch_ecotype.py::aggregate_to_recordings`; real numbers reported in `reports/ecotype_classifier.md` §8.4–8.5 |
| Empirical performance context (different domain, still useful) | Full-data window-level: Perch v2 + LogReg reached macro ROC-AUC 0.993 / macro F1 0.890 on a 5-class orca-ecotype task with 31,859 test windows, extraction "~20 min each on H100" for the full ecotype pool | `[VERIFIED]` `reports/ecotype_classifier.md` §8.4, §12 comment `# ~20 min each on H100` — cited as an order-of-magnitude throughput proxy only; PteroSet's own Phase 0 benchmark is still required (see §8) |
| Real dependency-pin evidence | Config/env conventions live as Python module-level constants (`EMB_DIR`, `PERCH_V2_LOCAL`, `PERCH_SR`, etc.), **not** a project-wide `config.yaml` — the reference repo does not use PteroSet's `--config data/config.yaml` convention at all for this sub-pipeline | `[VERIFIED]` `eval_perch_ecotype.py` module-level constants |

---

## 2. Side-by-side: Round 1 vs. reference vs. Round 2 decision

| Concern | Round 1 (self-invented) | Orcas reference (proven) | Round 2 decision |
|---|---|---|---|
| Perch loading | `perch_hoplite.zoo.model_configs.load_model_by_name()` | Raw `tf.saved_model.load()` on a locally-vendored SavedModel, downloaded once via `kagglehub` then copied | **Adopt reference.** Drop `perch_hoplite` as a dependency entirely. |
| CUDA/GPU bootstrap | Not designed; assumed `perch_v2_cpu` might "just work," flagged as needing empirical confirmation | Concrete, working `os.execvpe` re-exec shim setting `LD_LIBRARY_PATH` before TF import; empirically required because `perch_v2`'s SavedModel is CUDA-only | **Adopt reference verbatim** (copied function), targeting the GPU `perch_v2` slug rather than gambling on the unverified `perch_v2_cpu` slug. |
| Embedding storage | One `.npy` per window (160,244 files) + JSONL manifest with a SHA-256 `cache_key` per window | One `.npz` per split, containing all rows for that split as parallel arrays | **Adopt reference.** Drop per-window `.npy` files and the JSONL manifest. |
| Alternative storage considered | Parquet-sharded alternative floated in Round 1 §7 as a "future migration" option | Not used anywhere in the reference repo | **Remove from the design entirely** — not just deferred, actively removed. No evidence it is needed at this scale (orcas' pool is 216,940 windows, larger than PteroSet's 160,244, and a single `.npz` per split handles it fine). |
| Resumability | Per-window SHA-256 cache-key check, atomic tmp-then-rename `.npy` writes, `--shard-index/--num-shards` parallelism, `--verify` fsck pass | Split-level row-count check (`len(existing) == len(df)`); no atomic-write ceremony, no sharding, no separate verify mode | **Adopt reference's split-level granularity as the primary mechanism.** Keep one narrow addition (§6) justified by this exact repo's own documented history of a stale-cache bug (v2→v3 PPA4 regression), but scoped down from "SHA-256 of every config field" to a single cheap `source_json_sha256` check per pool file — see §6 for the precise, minimal version. |
| Cross-fold/cross-split reuse | Not designed — Round 1 assumed a single global extraction pass with no explicit mechanism for reusing it across differently-partitioned splits | `remap_perch_embeddings.py`: gather-by-`window_id` with hard existence + identity gates + random-sample self-verification | **Adopt directly, generalized from a 2-partition remap to an N-fold materializer** (§5) — this is the single most valuable pattern for PteroSet, because the 5 leave-one-project-out folds are exactly "many different partitions of one shared window pool," which is precisely the scenario this script was built for. |
| Classifier / training | `linear_probe/model.py`: `LinearProbeClassifier(pl.LightningModule)`, `nn.Linear` + `BCEWithLogitsLoss`, `torchmetrics`, PyTorch `Dataset`/`DataLoader`, epochs, a Lightning `Trainer` | `sklearn.linear_model.LogisticRegression` fit directly on an in-memory NumPy array; `joblib` persistence | **Adopt reference's framework, adapted for multilabel** (§7 — no multilabel precedent exists in the reference repo, so this part is a genuine, explicitly-flagged extension, not a copy). Drop the entire `linear_probe/` PyTorch Lightning package. |
| Module layout | New package `embeddings/` (`perch_config.py`, `perch_extractor.py`, `manifest.py`, `species_labels.py`) + new package `linear_probe/` (`dataset.py`, `model.py`) + 3 new top-level CLIs | Flat top-level scripts; one script (`eval_perch_ecotype.py`) holds all reusable functions and is directly imported by a sibling script (`extract_perch_3class.py`) — no package/`__init__.py` layering | **Adopt reference's flat, dual-purpose-file convention** (§5). Remove the `embeddings/` and `linear_probe/` package scaffolding from Round 1; replace with 3 flat top-level scripts, one of which (`extract_perch_pool.py`) is both a runnable CLI and an importable module, exactly mirroring `eval_perch_ecotype.py`. |
| Config integration | New `species:`/`embedding:`/`linear_probe:` blocks added to `data/config.yaml`, consumed via `--config` | Module-level Python constants, no project config file used for this sub-pipeline | **Partial adoption, one deliberate deviation** (§5): keep the reference's low-ceremony module-level constants as defaults (do not build a `perch_hoplite`-style `ConfigDict`), but still accept PteroSet's existing `--config data/config.yaml` for the handful of values PteroSet's own scripts already externalize this way (`species.excluded_codes`, `species.min_class_support`), since Round 1 already established that block and `train.py`/`prepare_dataset.py` already share this convention. This is a repo-consistency argument, not a rejection of the reference's simplicity. |
| Species label derivation | `species_labels.py`: multi-hot target construction from `annotations_species.json`, `excluded_codes`, `min_class_support`, `OTHER` bucket | No equivalent in the reference repo (orcas' labels are pre-baked single-label columns already in its split CSVs) | **Carried forward unchanged from Round 1.** This logic is orthogonal to the Perch backend/storage pivot; see §4. |
| Leakage control | Reuse existing, unmodified `folds_segmented_v4/*/{train,val,test}_split.csv`; fail-loud project-mismatch assertion at join time | N/A (orcas has its own separate split-construction scripts, not part of the cited reference trio) | **Carried forward unchanged from Round 1**, now implemented as the hard gates inside the fold-materializer (§5), which is a strictly *stronger* version of the join-time assertion Round 1 proposed. |
| Recording-level aggregation | Not proposed | `aggregate_to_recordings()`: mean-pool + majority-vote per recording | **Explicitly not adopted for Round 1 scope** (§7) — PteroSet's species task is dense multi-event per recording (dawn chorus), so a single majority-vote label per 8-minute file is not a meaningful target the way it is for orcas' one-ecotype-per-encounter recordings. Window-level only, matching the existing binary detector's evaluation granularity. Flagged as a possible Round 3 diagnostic, not a removal-by-oversight. |

---

## 3. What is explicitly removed from Round 1

1. `perch_hoplite` as a dependency, and every API surface built on it (`model_configs.load_model_by_name`, `PresetInfo`, `zoo_interface.EmbeddingModel`).
2. The `embeddings/` package (`perch_config.py`, `perch_extractor.py`, `manifest.py`) — replaced by flat scripts.
3. Per-window `.npy` files and the per-window JSONL manifest (`manifest.jsonl`, `manifest_meta.json`) with SHA-256 `cache_key`.
4. The Parquet-sharded storage alternative floated in Round 1 §7.
5. `--shard-index/--num-shards` parallel-extraction CLI flags and the atomic tmp-then-rename per-file write ceremony (unnecessary at split-level `.npz` granularity — a whole-split re-extraction is cheap and the reference's simpler row-count check is sufficient).
6. The `linear_probe/` PyTorch Lightning package (`dataset.py`, `model.py`, `LinearProbeClassifier(pl.LightningModule)`, `torchmetrics`).
7. `checkpoints_perch_v1/`, `outputs_perch_v1/` as Lightning-shaped artifact families — replaced by `joblib` classifiers + flat CSVs living under `checkpoints/perch/` (mirroring the reference's own directory name, adapted to PteroSet's existing `checkpoints_v{N}` sibling convention — see §5).

What is **not** removed (carried forward from Round 1 unchanged): the label-provenance/multilabel/unknown-background design (§4 of Round 1), the decision to reuse the existing `folds_segmented_v4/*` CSVs verbatim with no new split logic, the `excluded_codes`/`min_class_support`/`OTHER`-bucket class-list construction, and the phased-milestone structure (adjusted in §8 for the new substrate).

---

## 4. Label provenance, multilabel, leakage — unchanged from Round 1

Round 1 §4 stands as written: species multi-hot targets derived from `annotations_species.json` via the same any-overlap rule as the existing binary `label` field, `min_overlap_frac` configurable (default 0.0), coarse/placeholder codes (`PSITTACIDAE`, `PSITTACIFORMES`, `RHACAR`, `TYRANN_SP1`, `PSITTA`, `PICIDA_1`) excluded via config not silently dropped, rare species collapsed into an explicit `OTHER` bucket below `min_class_support`, "bird detected but no species-level annotation" windows tagged `has_species_label=false` and excluded from train/val/test tensors by default, and leakage control inherited for free by joining onto the existing, unmodified fold CSVs by `window_id`.

The only change: in Round 1 this join happened inside a PyTorch `Dataset.__init__`; in Round 2 it happens inside the fold-materializer script (§5, Step 3), which bakes the resulting multi-hot matrix directly into each fold's `.npz` — matching the reference's convention of a **self-contained** per-split cache (embeddings + labels + metadata together, no join needed at train time).

---

## 5. Revised pipeline — modules, classes, CLI contracts

Flat top-level scripts, mirroring the reference's file layout exactly. No new packages.

```
extract_perch_pool.py        # CUDA bootstrap, download/load Perch v2, forward-pass loop,
                              # extraction over the FULL window pool. Both a CLI and an
                              # importable module (mirrors eval_perch_ecotype.py's dual role).
build_fold_embeddings.py      # generalizes remap_perch_embeddings.py: gather-by-window_id
                              # from the pool .npz into 5 folds x {train,val,test}, with
                              # hard gates + self-verification.
train_perch_logreg.py         # generalizes eval_perch_ecotype.py --fulldata: multilabel
                              # OneVsRestClassifier(LogisticRegression), looped over 5 folds.
species_labels.py              # UNCHANGED from Round 1 — label derivation is independent
                              # of the Perch backend pivot.
build_embedding_pool_csv.py    # new, small (~30 lines): resolves windows_mapping_*.json +
                              # annotations_identification.json into one flat pool CSV with
                              # the columns extract_perch_pool.py expects.
```

### `build_embedding_pool_csv.py` (new, minimal)

Purpose: bridge PteroSet's existing artifacts (`windows_mapping_4.0overlap_segmented_v4.json`, sample-index geometry) into the column shape the reference's audio loader expects (`sound_filepath`, `window_start`/`window_end` **in seconds**, `window_id`, `dataset`). This is the one place Round 2 must diverge from a pure copy, because PteroSet's on-disk window representation is JSON+samples while orcas' is CSV+seconds.

```
python build_embedding_pool_csv.py --config data/config.yaml \
    --windows_json data/windows_mapping_4.0overlap_segmented_v4.json \
    --annotations data/annotations_identification.json \
    --out data/embeddings_pool/segmented_v4_pool.csv
```
Output columns: `window_id, sound_id, sound_filepath, dataset, window_start, window_end` (`window_start = start / sample_rate`, `window_end = end / sample_rate`, seconds, float) — a straight port of the existing `sound["file_name_path"]` + `start`/`end`/`sample_rate` join already performed inside `prepare_dataset.py::run_spectrograms`/`run_splits`, reused here rather than reinvented.

### `extract_perch_pool.py` (copied + adapted from `eval_perch_ecotype.py`)

Functions carried over near-verbatim (attribution comment pointing at the source file/repo, per the "copy/adapt" option chosen in §10):
- `_ensure_gpu_env()` / CUDA `LD_LIBRARY_PATH` re-exec bootstrap — **copied as-is**.
- `download_perch_v2(dest_dir)` — **copied as-is**, targeting the same `PERCH_V2_KAGGLE = "google/bird-vocalization-classifier/tensorFlow2/perch_v2"` slug (the GPU-empirically-working one, not the untested `perch_v2_cpu`).
- `load_perch_v2(model_dir)`, `inspect_model_outputs(model, version)` — **copied as-is**.
- `load_audio_segment(filepath, center_sec, window_sec=5.0, target_sr=32000)` — **copied as-is** (already correct for PteroSet's audio: 48 kHz source → 32 kHz target via the same single `librosa.load(sr=32000, offset=..., duration=5.0)` call).
- `extract_embeddings(model, emb_key, df, batch_size)` — **copied as-is**, since it only depends on `df.window_start`/`df.window_end`/`df.sound_filepath`, all present in the new pool CSV.

New, PteroSet-specific glue (the only genuinely new code in this file):
```python
def run_pool_extraction(model_version, pool_csv_path, out_path, batch_size,
                          model_dir_v1, model_dir_v2):
    """Single-partition analogue of run_extraction(): one CSV in, one .npz out.
    Row-count-match skip check identical to extract_perch_3class.py's."""
```
CLI:
```
python extract_perch_pool.py --download_v2                     # once
python extract_perch_pool.py --extract --model v2 \
    --pool_csv data/embeddings_pool/segmented_v4_pool.csv \
    --out data/embeddings/perch_v2/pool_emb_v2.npz \
    [--batch_size 32]
```
Output `.npz` fields (mirrors the reference schema, PteroSet-adapted): `embeddings (160244, 1536) float32`, `window_id (160244,) int64`, `sound_filepath (160244,) object`, `dataset (160244,) object` (project code), `sound_id (160244,) int64`, `window_start (160244,) float64`, `window_end (160244,) float64`. A single `.npz` for the full pool, matching the proven scale precedent (orcas' 216,940-window pool, larger than PteroSet's 160,244).

### `build_fold_embeddings.py` (generalized from `remap_perch_embeddings.py`)

Framing note: the reference script solves "one pool, an **old** partition superseded by a **new** partition" (a 1-to-1 remap). PteroSet's actual need is "one pool, **five independent** partitions of it" (the 5 LOPO folds) — same underlying mechanism (gather-by-`window_id`, hard gates, self-verify), generalized from a single remap call to a loop over `5 folds × 3 splits = 15` materializations, each independent.

```python
NPZ_KEYS_IN  = ("embeddings", "window_id", "sound_filepath", "dataset",
                "sound_id", "window_start", "window_end")

def load_pool(pool_npz_path) -> tuple[np.ndarray, dict[int, int], np.ndarray, np.ndarray]:
    """Loads the pool .npz once; returns (emb_pool, window_id -> row index,
    sound_id array, (start,end) array) for hard-gate checks."""

def materialize_fold_split(pool, fold_csv_path, species_labels_path,
                            class_list_path, out_path):
    """
    Hard gates (strictly matching remap_perch_embeddings.py's pattern, PLUS one
    PteroSet-specific strengthening):
      1. every window_id in fold_csv_path exists in the pool -> else SystemExit
         (report up to 5 missing ids, exact count).
      2. pool.sound_id[row] == fold_csv.sound_id  AND
         pool.window_start/end[row] matches fold_csv.start/end (converted to
         the same units) for every matched window_id -> else SystemExit.
         [STRENGTHENED vs. reference: the reference only checks sound_filepath;
         PteroSet additionally checks the (sound_id, start, end) triple, since
         filename alone is shared by ~50 windows per file and is a weaker key.]
      3. self-verification: for a random sample of min(20, N) rows per split,
         re-fetch from the pool by window_id and assert L2(emb_new - emb_pool) == 0.0.
    Writes: embeddings, window_id, dataset (=project), sound_id, start, end,
            label (existing binary), species_multihot (N, C) uint8 -- joined
            from species_labels_path + class_list_path by window_id.
    """
```
CLI:
```
python build_fold_embeddings.py \
    --pool_npz data/embeddings/perch_v2/pool_emb_v2.npz \
    --fold_dir data/folds_segmented_v4 \
    --species_labels data/species_labels/segmented_v4/species_labels_v1.json \
    --class_list data/species_labels/segmented_v4/class_list.json \
    --out_dir data/embeddings/perch_v2/folds_segmented_v4
```
Output layout: `data/embeddings/perch_v2/folds_segmented_v4/fold_{i}_{PROJECT}/{train,val,test}_emb.npz` — parallel to, and named after, PteroSet's own existing `folds_segmented_v4/fold_{i}_{PROJECT}_segmented/` directories, so the correspondence between a fold's spectrogram-pipeline split and its embedding-pipeline split is obvious by directory-name inspection alone. Runtime: "seconds," per the reference's own claim, since this is pure in-memory array gathering, no audio I/O.

### `train_perch_logreg.py` (generalized from `eval_perch_ecotype.py --fulldata`)

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import joblib

def fit_and_eval_fold(fold_dir, out_dir, C=1.0, max_iter=1000):
    """Window-level only (no recording-level aggregation -- see rationale in
    the comparison table, §2). For each fold:
      1. Load {train,val,test}_emb.npz.
      2. L2-normalize embeddings (normalize(..., norm='l2')) -- same call as
         the reference.
      3. clf = OneVsRestClassifier(LogisticRegression(solver='lbfgs',
             max_iter=max_iter, C=C))
         clf.fit(X_train, Y_train_multihot)
         [VERIFIED via scikit-learn documentation: for a multilabel-indicator
         y of shape (N, C), OneVsRestClassifier.predict_proba(X) returns an
         (N, C) array of per-class probabilities -- the direct multilabel
         analogue of the reference's multiclass predict_proba call.]
      4. Per-class roc_auc_score / average_precision_score (average=None),
         plus macro/micro aggregates; per-class F1 at a val-calibrated
         threshold (mirrors config.yaml's existing conf_threshold pattern
         and Round 1's `threshold_calibration: val_f1`).
      5. joblib.dump(clf, f'checkpoints/perch/logreg_species_fold{i}.joblib')
      6. Append one row per (fold, class) to a flat results DataFrame.
    """
```
CLI:
```
python train_perch_logreg.py \
    --embeddings_dir data/embeddings/perch_v2/folds_segmented_v4 \
    --class_list data/species_labels/segmented_v4/class_list.json \
    --out_dir checkpoints/perch \
    [--fold 0]                      # single fold, or all 5 if omitted
```
Outputs (flat CSVs, matching the reference's naming convention, adapted): `checkpoints/perch/fulldata_results_species_fold{i}.csv` (per-class AUROC/AP/F1 for that fold), `checkpoints/perch/fulldata_results_species_cv_summary.csv` (aggregated across all 5 folds — the multi-fold analogue the reference doesn't need, since it only ever has one train/test split; this generalization is required and new, flagged as such).

### Config touchpoint (the one deliberate deviation from pure copy, per §2)

`data/config.yaml` keeps the `species:` block from Round 1 unchanged (`excluded_codes`, `min_class_support`, `min_overlap_frac`, `rare_class_bucket`). Perch/embedding parameters (`sample_rate`, `window_size_s`, `target_peak`, `batch_size`, model dir paths) are **module-level constants** in `extract_perch_pool.py`, matching the reference's style, with CLI flags to override — not a new `embedding:` YAML block. This removes the `embedding:`/`linear_probe:` YAML blocks Round 1 proposed, replacing them with the reference's simpler constants+CLI-flags convention, while keeping the one YAML block (`species:`) that has no reference-repo analogue and genuinely belongs in PteroSet's existing config-driven label logic.

---

## 6. Resumability & integrity — scoped down from Round 1

Round 1's per-window SHA-256 cache-key/manifest system is removed (§3). In its place, Round 2 adopts the reference's row-count check **as the primary mechanism**, plus exactly one narrow addition justified by this specific repo's own history:

- **Pool extraction** (`extract_perch_pool.py`): skip-if-`len(existing["embeddings"]) == len(pool_csv)`, matching `extract_perch_3class.py` verbatim.
- **One addition**: also store `source_windows_json_sha256` (a single hash of `windows_mapping_4.0overlap_segmented_v4.json`, computed once) as a scalar field in the pool `.npz`. On re-run, if this hash differs from the current file's hash, force full re-extraction even if the row count happens to match — this directly targets the exact failure mode that caused the documented v2→v3 PPA4 label regression in *this* repo (a row-count-equal cache silently serving stale content after an upstream annotation change). This is a one-line check, not the full per-window manifest system Round 1 proposed — a minimal, targeted mitigation rather than a parallel framework.
- **Fold materialization** (`build_fold_embeddings.py`): the hard gates and self-verification described in §5 **are** the integrity mechanism here — stronger than anything Round 1 proposed for this step, and copied directly from a script whose entire purpose is exactly this kind of integrity-checked reuse.
- **No atomic tmp-then-rename, no sharding, no `--verify` fsck CLI mode.** At split/fold `.npz` granularity (15 small files, each written in one `np.savez_compressed` call), a crash mid-write simply leaves one incomplete file that the next run's row-count check will detect and regenerate — the reference's implicit safety margin, adopted as-is rather than re-engineered.

---

## 7. Multilabel adaptation — explicitly new, not copied

No file in the cited reference trio performs multilabel classification (ecotype and 3-class are both single-label, mutually exclusive). This is the one part of Round 2 that is a genuine extension rather than a direct port, and is flagged as such for Inquisitor scrutiny:

- Target: multi-hot `(N, C)` matrix (species co-occurrence within a window, e.g. dawn chorus), from Round 1's `species_labels.py` (unchanged).
- Classifier: `sklearn.multiclass.OneVsRestClassifier(LogisticRegression(...))` — verified against scikit-learn's own documentation that `predict_proba` on a multilabel-indicator target returns an `(N, C)` probability matrix, one column per class, which is the direct generalization of the reference's `multi_class="multinomial"` single-label call. `[VERIFIED via scikit-learn documentation, located via search and cross-checked against a mirrored copy of the same page]`.
- Metrics: per-class `roc_auc_score`/`average_precision_score` with `average=None` (independent per-class scores, since classes are not mutually exclusive — the reference's `multi_class="ovr", average="macro"` call is for softmax-style single-label multiclass and is not directly reusable for a true multilabel target), plus macro/micro aggregates.
- Recording-level aggregation (`aggregate_to_recordings()`) is **not** adopted for this task (see comparison table, §2): PteroSet windows within one 8-minute recording can each carry a *different* species combination, so a single majority-vote label per recording is not a coherent target the way it is for orcas' one-ecotype-per-encounter recordings. Window-level evaluation only, matching the granularity of the existing binary detector's `plot_cv_results.py`.

---

## 8. Compute & storage — revised

| Quantity | Round 1 estimate | Round 2 (npz-based) | Basis |
|---|---|---|---|
| Pool embeddings, full 160,244 windows | ~984 MB across 160,244 `.npy` files (float32) | **Same total bytes, one file**: 160,244 × 1536 × 4 B ≈ 984 MB inside a single `pool_emb_v2.npz` (compressed, likely smaller on disk) | `[COMPUTED]`, format changed not the underlying math |
| Per-fold materialized `.npz` | N/A (Round 1 joined at train time, no extra copy) | 5 folds × 3 splits, each a **subset** of the pool (train/val overlap-inclusive, test non-overlapping) — sums to somewhat more than 160,244 rows total across all 15 files because train/val windows are reused across multiple folds' pools; still well under 2× the pool size in aggregate | `[COMPUTED]`, negligible in absolute terms (low hundreds of MB even at 2×) |
| Throughput | Unbenchmarked, flagged for Phase 0 | Still unbenchmarked for PteroSet specifically, but now anchored to a real proxy: the reference repo's own comment logs "~20 min each on H100" for its 216,940-window ecotype pool (Perch v2). PteroSet's 160,244-window pool is ~74% of that size | `[VERIFIED proxy number]` for orcas' domain; `[ASSUMPTION — NEEDS EMPIRICAL CONFIRMATION]` that PteroSet's audio I/O characteristics (shorter 8-min files vs. orcas' recordings) transfer similarly — still requires PteroSet's own Phase 0 benchmark, now with a much better prior than Round 1's blind guess |
| Fold-materialization cost | N/A | "Seconds" per the reference's own description — pure in-memory NumPy gather, no audio decode, no TF forward pass | `[VERIFIED — as claimed for the reference's equivalent operation]` |

---

## 9. Convergence verdict

**NEW-IDEAS.**

Rationale: this round did not merely refine Round 1's plan — it replaced the implementation substrate wholesale (dependency (`perch_hoplite` → raw TF SavedModel), storage granularity (per-window `.npy`+manifest → per-split `.npz`), classifier framework (PyTorch Lightning → sklearn), integrity mechanism (SHA-256 cache-key → row-count + one targeted hash), and module layout (nested packages → flat dual-purpose scripts), on the strength of a **working, already-deployed sibling implementation** rather than first-principles design. That is new, load-bearing information that should get another pass from the Inquisitor before implementation starts — specifically on: (a) the GPU-only empirical finding and whether PteroSet's infrastructure can actually run the CUDA bootstrap reliably; (b) the multilabel `OneVsRestClassifier` adaptation, which has no precedent in either repo and is the one place this design is extrapolating rather than copying; (c) the strengthened `(sound_id, start, end)` hard gate in `build_fold_embeddings.py`, which is a deliberate deviation from the reference's `sound_filepath`-only check and should be checked for correctness against PteroSet's actual CSV column types (e.g. int vs. string `sound_id` across pandas dtypes).

What *did* converge and should not be revisited: the label-provenance/multilabel-target/leakage-control design (Round 1 §4, carried forward unchanged in §4 above), and the decision to reuse `folds_segmented_v4/*` verbatim with zero modification to the existing binary-detector pipeline.

---

## 10. Smallest shared-refactor vs. lower-risk copy/adapt

**Option A — smallest cross-repo refactor.** Extract the domain-agnostic functions (`_ensure_gpu_env`/CUDA bootstrap, `download_perch_v2`, `load_perch_v2`, `inspect_model_outputs`, `load_audio_segment`, `extract_embeddings` batch loop — none of which reference orca ecotypes, bird species, or either repo's directory layout) into a small local-path package, e.g. `perch_embed_utils/`, installed via `pip install -e ../shared/perch_embed_utils` from both repos' environments. Any future CUDA/TF-version fix is applied once. Cost: introduces a third artifact to version and release-coordinate, and a cross-repo dependency edge that must be managed carefully so a change made for one project's benefit cannot silently break the other's pinned behavior (e.g. bumping `tensorflow` in the shared package to fix an orcas CUDA issue could change PteroSet's embeddings without PteroSet's team requesting it, re-creating exactly the kind of stale/drifted-cache risk this design otherwise defends against in §6).

**Option B — copy/adapt, no shared package (recommended for Round 2).** Copy the same ~5 functions verbatim into `extract_perch_pool.py`, with a header comment attributing the source (`orcas_dclde2026/eval_perch_ecotype.py`) and the copy date, adapting only the path constants (`PERCH_V2_LOCAL`, `EMB_DIR`) to PteroSet's conventions. Zero cross-repo coupling: each repo's environment, TensorFlow/CUDA pin, and bootstrap logic can evolve independently. Cost: manual drift — a bugfix discovered in one repo (e.g. a new CUDA library path needed for a newer driver) must be manually ported to the other, and nothing enforces that it is.

**Recommendation**: adopt **Option B** for Round 2. This is a first experiment, not a committed shared platform, and the user's stated priority is explicitly "without coupling repos." Concrete trigger for promoting to Option A later: the moment a **third** project needs the same Perch-loading logic, or the moment a CUDA/TF driver upgrade requires the *same* fix to be applied in both repos simultaneously (at which point manual drift has already become a real, not hypothetical, cost) — either event should prompt revisiting this decision, not before.

---

## 11. Phased milestones — revised for the new substrate

**Phase 0 — Environment + smoke test (1 day, shorter than Round 1's estimate thanks to the reference's proven recipe)**
- Vendor `checkpoints/perch/model_v2/` locally via a one-time `download_perch_v2()` run (copied function), using the exact pinned versions from `orcas_dclde2026/pip-requirements.txt` (`tensorflow==2.21.0`, `kagglehub==1.0.0`, `tensorflow-hub==0.16.1`, `scikit-learn==1.7.2`, `joblib==1.5.3`) as PteroSet's `requirements-embeddings.txt`, rather than Round 1's untested `~=2.20.0rc0` guess.
- Run the copied CUDA bootstrap on PteroSet's actual GPU host; confirm `tf.saved_model.load()` + one forward pass succeeds without CPU fallback.
- Confirm `inspect_model_outputs()` resolves an embedding key of dim 1536 on the vendored model.
- Acceptance: one real PteroSet 5 s window embedded end-to-end; shape/dtype documented; zero changes to any existing repo file.

**Phase 1 — Pool CSV + full extraction (2–3 days)**
- `build_embedding_pool_csv.py` against `windows_mapping_4.0overlap_segmented_v4.json` + `annotations_identification.json`.
- `extract_perch_pool.py --extract` over the full 160,244-window pool.
- Acceptance: `pool_emb_v2.npz` row count == pool CSV row count; `source_windows_json_sha256` recorded; re-running is a no-op skip.

**Phase 2 — Species labels + fold materialization (1–2 days)**
- `species_labels.py` (unchanged from Round 1) → `species_labels_v1.json`, `class_list.json`, `class_support.csv`.
- `build_fold_embeddings.py` over all 5 folds × 3 splits.
- Acceptance: all 3 hard gates pass for all 15 materializations with zero `SystemExit`s; self-verification passes on every sampled row; per-fold `test_split.csv` project matches the fold's held-out project (inherited leakage guard).

**Phase 3 — LogReg training + CV (1–2 days)**
- `train_perch_logreg.py` across all 5 folds; trivial marginal-frequency baseline for sanity comparison.
- Acceptance: `fulldata_results_species_cv_summary.csv` produced; per-class AUROC/AP reported alongside per-class support; linear probe beats the trivial baseline on macro-AUROC.

**Phase 4 — Documentation (~1 day)**
- Documenter writes `docs/implementation/species-linear-probe-v1/results.md`, modeled directly on `orcas_dclde2026/reports/ecotype_classifier.md`'s §8 structure (methodology / models compared / results tables / discussion / artifacts), since that report is itself a proven template for exactly this kind of frozen-embedding-plus-LogReg comparison write-up.

**Phase 5 — explicitly deferred**: fine-tuning, alternative encoders (BirdNET, SurfPerch — still reachable later, now via the same raw-SavedModel-loading pattern rather than `perch_hoplite`'s zoo abstraction, which is no longer in the dependency set), Option A shared-package promotion (§10), recording-level diagnostics for species co-occurrence.

---

## Appendix — Source log for this round

- `orcas_dclde2026/eval_perch_ecotype.py`, `orcas_dclde2026/extract_perch_3class.py`, `orcas_dclde2026/remap_perch_embeddings.py`, `orcas_dclde2026/pip-requirements.txt`, `orcas_dclde2026/CLAUDE.md`, `orcas_dclde2026/reports/ecotype_classifier.md` (§8), `orcas_dclde2026/checkpoints/perch/` directory listing and `model_v2/` SavedModel layout — all read directly in this session.
- scikit-learn `OneVsRestClassifier` multilabel `predict_proba` output shape — verified via a documentation search that cited and cross-checked `scikit-learn.org`'s own generated API reference page.
- Round 1 facts (Perch 2 API via `perch_hoplite`, PteroSet repo conventions) carried forward from `docs/design/round_01/architect_pipeline_proposal.md`, not re-verified in this round except where explicitly superseded above.

STATUS: DONE
