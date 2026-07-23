# Linear Probing with Google Perch 2 Embeddings for Species Classification on PteroSet

**Author**: Chief Architect (architect-experiment, Round 1)
**Status**: Proposed — awaiting Inquisitor stress-test and Documenter polish before Round 2
**Scope**: Design + experiment plan only. No code, configs, or data artifacts are created by this document.

**How to read this document**: every factual claim is tagged.
- `[VERIFIED: repo]` — confirmed by reading files in this repository (path given).
- `[VERIFIED: source]` — confirmed by fetching an authoritative external source (URL given), not from training-data memory.
- `[RECOMMENDATION]` — an architectural choice I am proposing; alternatives and rationale are given.
- `[OPEN QUESTION]` — something that must be resolved empirically before or during Phase 0/1; do not treat as settled.

A companion "Delegated Investigation Log" (§13) documents four focused sub-investigations I ran directly, since this environment does not expose a live multi-agent task-spawning tool (only read/search/fetch/write primitives). I am flagging that limitation explicitly rather than silently pretending sub-agents were spawned.

---

## 1. Why

The repository's current species-relevant asset is `annotations_species.json`, which is *unused* by the training pipeline — `train.py`/`prepare_dataset.py` only consume `annotations_identification.json` (binary "is there a bird vocalization" label) `[VERIFIED: repo — data/config.yaml:18, prepare_dataset.py run_windows/run_segment_windows]`. Species-level classification is a genuinely new objective, not a parameter change to the existing binary detector.

Two facts make **linear probing on frozen Perch 2 embeddings** the right *first* experiment, not merely a convenient one:

1. **Scientific validity of a first claim.** PteroSet has only 6,702 species-determined annotations across 168 species codes `[VERIFIED: repo — grep "anno_id" in annotations_species.json = 6702; species.csv row count = 168]`, spread over 5 recording projects with leave-one-project-out (LOPO) evaluation. Training a full CNN from scratch on this scale, per-species, per-fold, would produce noisy, low-power estimates before we even know whether the *labels* are trustworthy. A frozen-embedding linear probe (a) is a convex, well-understood, low-variance estimator suitable for small-N per-class settings, (b) isolates "do these labels + this feature space support a decision boundary" from "did we tune a big network correctly," and (c) is exactly the workflow Perch 2's authors recommend and validate their embeddings for `[VERIFIED: source — https://huggingface.co/cgeorgiaw/Perch: "The embeddings were trained with the goal of being linearly separable. For most cases training a simple linear classifier on top of the model's outputs should work well."]`.
2. **The existing multiclass path is architecturally the wrong shape for this task.** `train.py`'s `--num_classes` path is framed as *mutually exclusive* classification ("Binary" vs "Multiclass (N classes)", single `label` column, softmax-style multiclass) `[VERIFIED: repo — train.py:4,266, folds_segmented_v4/.../train_split.csv has a single scalar `label` column]`. Species co-occur within a 5 s window (I confirmed two temporally close `CYAVIO` events in the same recording `[VERIFIED: repo — annotations_species.json:6913-6933]`), so species classification is **multilabel**, not multiclass. Reusing the existing single-label path would silently misrepresent the task.

Consequently: this is a **new, parallel pipeline** that reuses PteroSet's existing window/fold artifacts for leakage control, but does not touch `train.py`'s classification head or loss.

## 2. What (scope)

**In scope (this round):**
- Derive a window-level, multilabel species ground truth from `annotations_species.json`, reusing the exact window definitions in `windows_mapping_4.0overlap_segmented_v4.json`.
- Audit `species.csv` / `annotations_species.json` for taxonomic and provenance defects (found several during this design pass — see §4).
- Extract Perch 2 embeddings for all 160,244 segmented_v4 windows, store them as a frozen, versioned artifact.
- Train linear (convex) multilabel probes per LOPO fold, reusing `data/folds_segmented_v4/` splits verbatim.
- Calibrate per-species probabilities and choose thresholds on validation data only.
- Report primary/secondary multilabel metrics with cross-fold statistical summaries and uncertainty.
- Run a bounded ablation matrix (pooling, model version, label-noise policy, evaluation population, minimum support).
- Establish a **zero-shot Perch 2 classifier baseline** (no training at all) as a floor/reference.

**Explicitly out of scope (future rounds):**
- Fine-tuning Perch 2 weights (contradicts "linear probing" objective; also the CPU/GPU model variants are SavedModels, not obviously fine-tunable without additional tooling — `[OPEN QUESTION]`).
- Modifying the existing binary detector (`train.py`, ResNet/spectrogram path) or its checkpoints.
- Any change to how `windows_mapping_*segmented_v4.json` or `folds_segmented_v4/` are generated — they are treated as **read-only, versioned inputs**.
- Deployment/cascade design (binary detector → species head) — noted as a follow-on decision, not built here.

## 3. How — Perch 2 facts, verified

I fetched primary sources directly rather than relying on memory, per the anti-hallucination requirement.

| Fact | Value | Source |
|---|---|---|
| Package | `perch-hoplite` (successor tooling; the `google-research/perch` repo itself warns `pip install` is unlikely to work there) | `[VERIFIED: source]` https://github.com/google-research/perch ("this repository is for sharing open code artifacts... For inference and practical tooling, please try the Perch-Hoplite repository") |
| Install | `pip install perch-hoplite`, or with TF: `pip install 'perch-hoplite[tf]'` (CPU) / `pip install 'perch-hoplite[tf-cuda]'` (GPU) | `[VERIFIED: source]` https://github.com/google-research/perch-hoplite |
| Model load API | `perch_hoplite.zoo.model_configs.load_model_by_name('perch_v2')` → `EmbeddingModel` | `[VERIFIED: source]` https://raw.githubusercontent.com/google-research/perch-hoplite/main/perch_hoplite/zoo/model_configs.py |
| Preset resolution | `'perch_v2'` auto-dispatches to `'perch_v2_gpu'` if `has_gpu_tf()` else `'perch_v2_cpu'`; both map to class `TaxonomyModelTF`; a `'perch_v2_onnx'` preset also exists (`PerchV2OnnxModel`) | `[VERIFIED: source]`, same file |
| Input contract | `model.embed(audio_array)` where `audio_array` is `[Time]`, **unit-scaled** (peak-normalized) waveform | `[VERIFIED: source]` https://raw.githubusercontent.com/google-research/perch-hoplite/main/perch_hoplite/zoo/zoo_interface.py |
| Window/hop/sample rate | `window_size_s=5.0`, `hop_size_s=5.0`, `sample_rate=32000` (i.e. 160,000 samples/window) for `perch_v2` | `[VERIFIED: source]` `model_configs.py`, `taxonomy_model_tf.py` |
| Preprocessing detail | `target_peak=0.25` default peak-normalization applied inside `TaxonomyModelTF` before inference | `[VERIFIED: source]` https://raw.githubusercontent.com/google-research/perch-hoplite/main/perch_hoplite/zoo/taxonomy_model_tf.py |
| Embedding dim | `1536` for `perch_v2`/`perch_v2_gpu`/`perch_v2_cpu`; `1280` for the older `perch_8` (`bird-vocalization-classifier` TFHub v8) and `surfperch`; also unpooled `(5, 3, 1536)` "spatial" embedding available before pooling | `[VERIFIED: source]` `model_configs.py`; HF card corroborates 1536 pooled / (5,3,1536) unpooled |
| Output object | `InferenceOutputs(embeddings, logits, ...)`; `embeddings` shape `[Frames, Channels, Features]` (or `[B,F,C,D]` batched); `pooled_embeddings(time_pooling, channel_pooling)` helper with `POOLING_METHODS = ['first','mean','max','mid','flatten','squeeze']` | `[VERIFIED: source]` `zoo_interface.py` |
| Model weights source | Downloaded via `kaggle_hub` from `kaggle_hub.PERCH_V2_SLUG` / `PERCH_V2_CPU_SLUG` — i.e. requires `kagglehub` and (likely) a Kaggle account/API token or license click-through | `[VERIFIED: source]` `taxonomy_model_tf.py` (`from_tfhub` calls `kaggle_hub.load(...)`); exact auth requirement is `[OPEN QUESTION]` — must be smoke-tested in Phase 0 on the target compute environment. |
| Code license | Apache License 2.0 (perch-hoplite repository, all files carry `Licensed under the Apache License, Version 2.0`) | `[VERIFIED: source]` https://raw.githubusercontent.com/google-research/perch-hoplite/main/LICENSE and per-file headers |
| Model weights license | Reported as Apache-2.0 by a web search summarizing the Kaggle model card, but I could **not** directly render kaggle.com in this session (client-rendered page; `web_fetch` failed) `[OPEN QUESTION — re-verify by opening the Kaggle model card manually before publication]`. Code license (Apache-2.0) and weight license are legally distinct; do not conflate them in the paper's methods/license section. |
| CPU vs GPU discrepancy | The Hugging Face card states "This version of the model requires TensorFlow 2.20.rc0 and a GPU. A CPU variant will be added soon" `[VERIFIED: source]`, yet the `perch-hoplite` source ships a working `perch_v2_cpu` preset today `[VERIFIED: source]`. These two statements are inconsistent in time (HF card likely stale) — **do not trust the HF card's GPU-only claim**; verify empirically which preset actually runs on this repo's compute in Phase 0. |
| Model architecture | EfficientNet-B3 embedding backbone (~12M params) + classification head (~91M params, ~15,000 classes, ~10,000 of which are bird species) | `[VERIFIED: source]` HF card |
| Reliability caveat (from the model authors themselves) | "the output logits for species are uncalibrated and possibly unreliable for rare species, and we recommend that you use your own data to tune detection thresholds" | `[VERIFIED: source]` HF card — this is direct authority-level justification for our calibration/thresholding plan in §8, and for treating Perch's own logits only as a *baseline*, not ground truth. |

**Fit with PteroSet, verified:** PteroSet windows are already 5.0 s at a fixed `window_size_sec: 5.0` `[VERIFIED: repo — data/config.yaml:23]`, and every surviving segmented_v4 window spans exactly `end-start=240000` samples at the recorded 48 kHz rate `[VERIFIED: repo — windows_mapping_4.0overlap_segmented_v4.json sample entries]`, i.e. exactly 5.0 s. This is a very convenient 1:1 alignment with Perch 2's native 5.0 s / 5.0 s hop window — **one PteroSet window → one Perch embedding**, no internal re-framing needed. `[OPEN QUESTION — Phase 0 assertion]`: confirm this holds for *all* 160,244 windows, not just the sampled ones, since a single truncated boundary window would silently break the 1:1 assumption.

## 4. Label provenance and dataset audit — findings so far

I ran a first-pass audit while orienting (this must be completed formally in Phase 0, §9, before any modeling). Findings, all `[VERIFIED: repo]`:

1. **Two annotation files, different populations.** `annotations_identification.json` has a single category, `AVEVOC` (`supercategory: BIO`) — i.e. it is already pre-filtered to bird-vocalization events only (`data/annotations_identification.json:90-96`). It contains **15,372** annotation objects (`grep "anno_id"` count). `annotations_species.json` contains **6,702** annotation objects, each carrying a species code as `category` (`data/annotations_species.json:6900+`).
2. **Discrepancy vs. the dataset's own documented totals.** The dataset's `info.description` (embedded in both JSON files) states "producing 14,205 bird-event annotations at the taxonomic-group level; 6,703 of these additionally include species-level determinations" `[VERIFIED: repo]`. The locally regenerated files show **15,372** identification-level and **6,702** species-level annotations — a **+1,167 (≈8%) discrepancy** on the identification count and an off-by-one on the species count. This is exactly the class of regression CLAUDE.md documents happening between v1–v4 of the windows JSON (PPA4 annotation fixes/regressions). **This must be reconciled and dated against a specific `annotations_*.json` content hash before any species-label artifact is built** — do not silently trust either number.
3. **Species-code taxonomic rank is inconsistent within one flat namespace.** `species.csv` includes not only true species-rank codes but also coarser placeholders used when the annotator could not resolve to species: `PSITTACIDAE` (family, species field `"–"`), `PSITTACIFORMES` (order, `"–"`), `RHACAR` (`"–"`, unresolved), `PICIDA_1` (`"Picidae"`, family), `TYRANN_SP1` (`"Tyrannidae sp 1"`), `PSITTA` (`"Psittacidae sp."`) `[VERIFIED: repo — data/species.csv grep]`. Treating these as classes co-equal with true species (e.g. `CYAVIO` = *Cyanocorax violaceus*) would mix taxonomic ranks in one classification namespace, which is not scientifically defensible and will look like "extra confusable classes" in the confusion matrix without an obvious cause.
4. **Duplicate species codes for the same taxon** (verified by exact string match on the `species` column): `RAMTUC` and `RHATUC` both map to *Ramphastos tucanus*; `ATRPIL` and `ATAPIL` both map to *Atalotriccus pilaris* `[VERIFIED: repo — data/species.csv]`. If not merged, the probe will be asked to separate two classes that are the same biological entity, injecting pure annotator-inconsistency noise into both training and evaluation.
5. **Long-tail is expected but not yet quantified.** 6,702 annotations over up to 168 codes (fewer after dedup/rank-filtering) implies a mean of ~40 events/species, but with almost certainly a heavy-tailed distribution (a handful of common taxa, many singleton/near-singleton codes). The exact per-species, per-project counts are **not yet computed** — this is a mandatory Phase 0 deliverable, not something to estimate manually here.
6. **Window ↔ audio file resolution is indirect and must not be trusted literally from the JSON.** `annotations_*.json`'s `sounds[].file_name_path` points at `data/audios_192khz/...` and records `sample_rate: 192000` `[VERIFIED: repo — data_reader.py:21, sample sound entries]`. But `windows_mapping_..._segmented_v4.json` records `"sample_rate": 48000` per window, and the existing evaluation script defaults to reading audio from `data/audios_48khz/` by basename, not from the JSON's literal path `[VERIFIED: repo — plot_cv_results.py:557 default="data/audios_48khz"]`. The embedding-extraction script (§6) must resolve audio the same way the existing pipeline does — by `sound_filename` basename into `data/audios_48khz/` — not by following `file_name_path` verbatim, or it will silently read the wrong (192 kHz, un-time-aligned-by-name) files.
7. **Existing binary label semantics.** A window's binary `label` is derived by interval overlap between the window's `[start, end)` (converted to seconds) and *any* identification-level annotation on that `sound_id`: `any(a_min < we_sec and a_max > ws_sec for a_min, a_max in ...)` `[VERIFIED: repo — prepare_dataset.py run_segment_windows]`. The species-label derivation (§5) reuses this exact overlap predicate, applied per species-level annotation, to guarantee consistency with the already-validated binary label.

**Decision gate (blocking):** Phase 0 must (a) resolve items 2–5 into a canonical `species_taxonomy_crosswalk.csv`, and (b) publish a dated audit report, before any embeddings or labels are generated for modeling. See §9 and §12.

## 5. Label derivation design (window granularity, multilabel, unknown/background)

**Granularity:** window-level (same physical unit as the existing binary detector — `window_id` in `windows_mapping_4.0overlap_segmented_v4.json`). No new windowing is introduced; species labels are a new field/sidecar keyed by `window_id`.

**Multilabel target construction**, per window `w` with `[start_sec, end_sec)`:
1. Compute the set of species-level annotations (post-canonicalization, post-rank-filtering — see crosswalk) on `w.sound_id` whose `[t_min, t_max)` overlaps `[start_sec, end_sec)` using the identical predicate already used for the binary label (`a_min < we_sec and a_max > ws_sec`).
2. Compute the set of *identification*-level annotations (`AVEVOC`) on `w.sound_id` overlapping the same interval — this reproduces the existing binary `label`.
3. Classify each window into exactly one of three provenance states:
   - **`clean_negative`**: binary `label == 0` (no bird event at all). Multilabel target = all-zero vector. This is a true negative for every species.
   - **`species_positive`**: binary `label == 1` **and** ≥1 canonical species annotation overlaps the window. Multilabel target = 1 for each overlapping canonical species, 0 for all others *that are eligible for this fold* (see below on absent-in-project handling — this is not the same as "confirmed absent").
   - **`ambiguous_unresolved`**: binary `label == 1` **but** no species-level annotation overlaps the window (either the overlapping identification event was never taken to species, or a species annotation exists on the file but does not itself overlap this particular window). **Default policy: exclude these windows from species-model training and from the primary evaluation population.** Marking them as all-zero (implying "no species present") would be false — a bird is present but its identity is simply unknown at this granularity, and doing so would teach the classifier a systematic false negative signal correlated with exactly the recordings/species hardest to identify. This exclusion-by-default is itself an ablation axis (§10, ablation A5), not a silent, unexamined choice.
4. Emit, per window: `window_id`, `dataset/project`, `provenance_state`, `species_present: List[canonical_code]`, `n_species_annotations_overlapping`, and (for auditability) the raw pre-canonicalization codes.

**Multilabel, not multiclass:** the target is a binary vector of length `K` (K = number of canonical, species-rank codes surviving the crosswalk), not a single class index. This directly rules out reusing `train.py`'s existing "Multiclass" path (softmax/argmax semantics) — confirmed as a real, not hypothetical, mismatch in §1.

**Unknown/background treatment summary:**

| State | Windows | Species vector | Used in training? | Used in primary eval? |
|---|---|---|---|---|
| `clean_negative` | binary label 0 | all-zero | Yes (as negatives for every eligible species) | Yes |
| `species_positive` | binary label 1, species resolved | 1s for present species | Yes | Yes |
| `ambiguous_unresolved` | binary label 1, species unresolved | undefined — excluded | No (default) | No (default; reported separately as a coverage statistic) |

**Absent-in-held-out-project species:** under LOPO, a species observed in training projects may have **zero** occurrences in the held-out test project. Per-fold, per-species metrics for such species are **undefined (support = 0)**, not zero — they must be explicitly excluded from that fold's macro-average and reported as "N/A (no positives in held-out project)" rather than silently contributing a 0 or being dropped without a trace. See §8, §11 (failure modes).

## 6. Data-flow diagram

```
                              ┌─────────────────────────────────────────┐
                              │   data/annotations_species.json         │
                              │   data/annotations_identification.json  │  (read-only, provenance-audited)
                              │   data/species.csv                      │
                              └───────────────────┬───────────────────--┘
                                                   │
                                        [Phase 0]  ▼
                              ┌───────────────────────────────────────--┐
                              │ audit_species_labels.py                 │
                              │  - dedup codes (RAMTUC/RHATUC, ...)      │
                              │  - drop non-species ranks (family/order) │
                              │  - reconcile 14,205/6,703 vs actual      │
                              │  - per-species/per-project count table  │
                              └───────────────────┬───────────────────--┘
                                                   ▼
                              data/species_taxonomy_crosswalk.csv  (canonical_code, rank, sci_name, perch_class_id?)
                                                   │
        data/windows_mapping_4.0overlap_segmented_v4.json  (read-only, existing, 160,244 windows)
                                                   │
                                        [Phase 1]  ▼
                              ┌───────────────────────────────────────--┐
                              │ build_species_labels.py                 │
                              │  - per-window overlap w/ species anns    │
                              │  - provenance_state assignment           │
                              └───────────────────┬───────────────────--┘
                                                   ▼
                          data/species_windows_segmented_v4.json  (window_id -> species vector + state)
                                                   │
        data/folds_segmented_v4/fold_k_XXX/{train,val,test}_split.csv  (read-only, existing, REUSED VERBATIM)
                                                   │
                                        [Phase 1]  ▼  (join by window_id)
                              ┌───────────────────────────────────────--┐
                              │ per-fold {train,val,test}_species.csv    │
                              │  = existing split cols + species vector  │
                              └───────────────────┬───────────────────--┘
                                                   │
        data/audios_48khz/*.wav  (resolved by sound_filename basename, NOT by JSON file_name_path)
                                                   │
                                        [Phase 2]  ▼
                              ┌───────────────────────────────────────--┐
                              │ extract_embeddings.py                   │
                              │  - slice window @ 48kHz -> resample 32kHz│
                              │  - perch_hoplite model.embed(waveform)   │
                              │  - store pooled (1536) + spatial (5,3,.) │
                              └───────────────────┬───────────────────--┘
                                                   ▼
              data/embeddings/perch_v2/segmented_v4/{shard_*.parquet, manifest.json}
                                                   │
                                        [Phase 4]  ▼
                              ┌───────────────────────────────────────--┐
                              │ linear_probe.py --fold k                │
                              │  - fit on train, model-select on val     │
                              │  - convex multilabel linear model        │
                              └───────────────────┬───────────────────--┘
                                                   ▼
                              ┌───────────────────────────────────────--┐
                              │ calibrate.py --fold k                   │
                              │  - per-species Platt/isotonic on val     │
                              │  - threshold selection on val            │
                              └───────────────────┬───────────────────--┘
                                                   ▼
                              ┌───────────────────────────────────────--┐
                              │ evaluate_species.py                     │
                              │  - apply to held-out project test set    │
                              │  - per-species + macro/micro metrics     │
                              │  - bootstrap CIs, cross-fold aggregation │
                              └───────────────────┬───────────────────--┘
                                                   ▼
                     outputs_species_v1/fold_k_XXX/{metrics.json, predictions.csv}
                     reports/species_v1/{cv_results.csv, ablation_results.csv, species_label_audit.md}
```

Two things to note about this diagram: (1) every box reading from the existing `windows_mapping_*` / `folds_segmented_v4/` artifacts is read-only — nothing in this design regenerates or overwrites them; (2) the embedding-extraction step is deliberately isolated behind a frozen artifact boundary (§7) so that the TensorFlow-based Perch runtime never needs to coexist in-process with the PyTorch/Lightning training runtime.

## 7. Exact proposed files, CLI interfaces, artifacts

**New top-level package** `species/` (parallel to, not modifying, `train.py`/`prepare_dataset.py`):

```
species/
  __init__.py
  audit_species_labels.py     # Phase 0
  build_species_labels.py     # Phase 1
  extract_embeddings.py       # Phase 2
  linear_probe.py             # Phase 4
  calibrate.py                # Phase 4
  evaluate_species.py         # Phase 5
  zero_shot_baseline.py       # Phase 3
  taxonomy_crosswalk.py       # Phase 0 helper (shared by audit + build)
data/
  species_config.yaml                       # new, sibling to config.yaml (see rationale below)
  species_taxonomy_crosswalk.csv             # Phase 0 output
  species_windows_segmented_v4.json          # Phase 1 output, paired with windows_mapping_4.0overlap_segmented_v4.json
  embeddings/perch_v2/segmented_v4/
    shard_{000..NNN}.parquet                # window_id, embedding_1536 (or spatial 5x3x1536), extraction metadata cols
    manifest.json                           # model preset, resolved kaggle path/version, code commit hash, extraction config hash, per-shard checksums
outputs_species_v1/
  fold_0_MAP1_segmented/{linear_probe.joblib, calibration.json, metrics.json, predictions.csv}
  fold_1_PPA1_segmented/...
  ... (one dir per existing fold, same naming convention as folds_segmented_v4/)
reports/species_v1/
  species_label_audit.md
  species_cv_results.csv        # per-fold, per-species metrics (long format)
  species_ablation_results.csv
tests/
  test_species_label_overlap.py
  test_taxonomy_crosswalk.py
  test_embedding_extraction_contract.py
  test_fold_join_integrity.py
  test_species_metrics.py
```

**Why a new `data/species_config.yaml` instead of extending `data/config.yaml`:**
- *Considered*: add a `species:` block to the existing `config.yaml`. Rejected because `config.yaml` is actively used by the binary-detector pipeline (`prepare_dataset.py`, `train.py`); coupling species-experiment iteration (frequent, exploratory, many ablations) to the same file risks accidental regressions to the binary pipeline's config, and violates separation of concerns.
- *Decision*: `data/species_config.yaml` as a sibling file. It references the same `paths.data_root`, `datasets`, and `audio.sample_rate` (48000, source) as `config.yaml` (duplicated intentionally, not inherited via code coupling, to keep the species pipeline runnable independent of the binary pipeline's evolution) plus new keys: `species.embedding_model` (`perch_v2`/`perch_8`), `species.pooling` (`mean`/`flatten`/...), `species.min_support_train`, `species.ambiguous_policy` (`exclude`/`negative`/`separate_class`), `species.eval_population` (`all_windows`/`bird_positive_only`), `species.calibration_method` (`platt`/`isotonic`), `species.random_state`.

**CLI interfaces** (argparse, mirroring existing conventions such as `--config`, `--fold_dir`, `--fold`):

```bash
# Phase 0 — audit (read-only, writes report + crosswalk)
python -m species.audit_species_labels \
    --species_json data/annotations_species.json \
    --ident_json data/annotations_identification.json \
    --species_csv data/species.csv \
    --out_crosswalk data/species_taxonomy_crosswalk.csv \
    --out_report reports/species_v1/species_label_audit.md

# Phase 1 — derive multilabel window targets, join onto existing folds
python -m species.build_species_labels \
    --windows_json data/windows_mapping_4.0overlap_segmented_v4.json \
    --species_json data/annotations_species.json \
    --ident_json data/annotations_identification.json \
    --crosswalk data/species_taxonomy_crosswalk.csv \
    --out data/species_windows_segmented_v4.json \
    --fold_dir data/folds_segmented_v4   # writes {split}_species.csv alongside existing {split}_split.csv per fold

# Phase 2 — embedding extraction (isolated TF process/venv; see §14 risk R1)
python -m species.extract_embeddings \
    --config data/species_config.yaml \
    --windows_json data/windows_mapping_4.0overlap_segmented_v4.json \
    --audio_dir data/audios_48khz \
    --model perch_v2 \
    --pooling mean \
    --out_dir data/embeddings/perch_v2/segmented_v4 \
    --shard_size 5000 \
    --num_workers 8

# Phase 3 — zero-shot Perch baseline (no training)
python -m species.zero_shot_baseline \
    --config data/species_config.yaml \
    --crosswalk data/species_taxonomy_crosswalk.csv \
    --embeddings_dir data/embeddings/perch_v2/segmented_v4 \
    --fold_dir data/folds_segmented_v4 \
    --out reports/species_v1/zero_shot_results.csv

# Phase 4 — linear probe (single fold, mirrors train.py --fold semantics)
python -m species.linear_probe \
    --config data/species_config.yaml \
    --fold_dir data/folds_segmented_v4 --fold 0 \
    --embeddings_dir data/embeddings/perch_v2/segmented_v4 \
    --out_dir outputs_species_v1

python -m species.calibrate \
    --config data/species_config.yaml \
    --fold_dir data/folds_segmented_v4 --fold 0 \
    --probe_dir outputs_species_v1/fold_0_MAP1_segmented

# Phase 5 — cross-fold aggregation + bootstrap CIs
python -m species.evaluate_species \
    --config data/species_config.yaml \
    --fold_dir data/folds_segmented_v4 \
    --probe_root outputs_species_v1 \
    --out_csv reports/species_v1/species_cv_results.csv \
    --bootstrap_n 2000
```

All scripts accept `--cross_validation` (loop over all 5 folds) analogous to `train.py`, for parity with existing workflows.

## 8. Metrics, calibration, thresholding

**Population axes (report both, do not conflate):**
- **All-windows population** (all 160,244 windows; `clean_negative` + `species_positive`, `ambiguous_unresolved` excluded per §5 default): answers "if the species head runs on every window, unconditioned on any detector."
- **Bird-positive-only population** (`species_positive` + `ambiguous_unresolved`-excluded subset of positives): answers "if the species head only ever sees windows a working binary detector already flagged" — the practically relevant cascade scenario, and the fairer comparison because it avoids the metric inflation from a majority of trivial true negatives in the all-windows population.

**Primary metric:** **macro-averaged Average Precision (AP)**, computed per canonical species then averaged **only over species meeting a minimum test-fold support threshold** (`species.min_support_train`, e.g. ≥5 positive test windows; exact value set by the Phase 0 audit's observed distribution, not guessed here). Species with zero support in a given fold's test set are excluded from that fold's macro-AP with an explicit N/A marker (§5), not silently dropped without a footnote.

*Why AP and not accuracy/F1/Hamming loss as primary*: with sparse multilabel targets and a heavy long tail, a trivial all-negative predictor scores misleadingly well on accuracy and Hamming loss. AP integrates the precision-recall trade-off without requiring a fixed threshold, which matters because thresholds must be tuned per-species on validation data (see below), not assumed.

**Secondary metrics:**
- Micro-averaged AP (pooled across all species×window pairs) — reported alongside macro, since it's dominated by common species and answers a different question ("overall event-level PR performance").
- Label-ranking average precision (LRAP) — appropriate for multilabel ranking quality per window.
- Per-species ROC-AUC, reported with an explicit caveat given known instability under extreme class imbalance (use PR-AUC as primary, ROC-AUC as a secondary/sanity cross-check only).
- Per-species precision/recall/F1 at the calibrated operating threshold (not at a blanket 0.5), to remain consistent with the existing repo's own acknowledgment that a fixed `conf_threshold` (`data/config.yaml:48`) is a simplification even for the binary task.
- Brier score and per-species Expected Calibration Error (ECE), pre- and post-calibration, to make calibration quality auditable rather than assumed.

**Calibration procedure:**
1. Fit calibration **only** on the validation split of each fold (never on test, never on train-that-generated-the-raw-scores) — Platt (logistic) scaling by default, isotonic regression as an ablation, per canonical species independently. `species.calibration_method` config-selectable.
2. Species with too few validation positives to fit a calibration curve reliably (config threshold) fall back to an uncalibrated raw-score report with an explicit flag — this must never fail silently as "calibrated" when it wasn't.
3. Threshold selection: per-species threshold maximizing F1 on the calibrated validation scores (`[RECOMMENDATION]`, consistent with the existing single global `conf_threshold` pattern generalized to per-species). Report metrics **both** at this tuned threshold and at a fixed universal 0.5 threshold, to make the sensitivity to threshold-tuning visible rather than hidden.

**Statistical summaries across held-out projects (explicit requirement):**
- Report a **per-project table** (5 rows, one per LOPO fold: MAP1, PPA1–PPA4) of primary/secondary metrics — never only a single pooled number, since with only 5 held-out projects, pooling would hide which project(s) drive the average.
- Report **mean ± std across the 5 folds** for each metric, explicitly labeled as *between-project variance* (n=5, small — do not imply CLT-based normal CIs at this n).
- Separately, within each fold, compute a **bootstrap CI** (resample test windows with replacement, stratified by species prevalence, `species.random_state`-seeded, ≥2000 resamples) to characterize *within-fold sampling uncertainty* given finite test windows.
- These two uncertainty sources (between-project, within-fold) must be reported as **distinct rows/columns**, not merged into one number — conflating them would misrepresent how much of the observed spread is "this project is just different" vs. "we don't have enough test windows to know."

## 9. Train/val/test policy and leakage control

**Core decision: reuse `data/folds_segmented_v4/fold_k_*/{train,val,test}_split.csv` verbatim — do not re-split.**

Rationale, verified from `prepare_dataset.py::run_splits`:
- Test sets are the held-out project's **non-overlapping** windows only (`start % window_size_samples == 0`), which already prevents the species-eval test set from containing near-duplicate, 80%-overlapping windows of itself `[VERIFIED: repo — prepare_dataset.py:451-466]`.
- Train/val within the remaining 4 projects are split with **`GroupShuffleSplit` grouped by `sound_id`** `[VERIFIED: repo — prepare_dataset.py:471-483]` — i.e., whole recordings are assigned wholly to train or wholly to val. This is exactly the safeguard needed given the 4 s/5 s (80%) window overlap: without sound-level grouping, adjacent overlapping windows from the same recording could split across train/val and leak. **This existing safeguard is already correct and must be preserved by joining species labels onto these CSVs by `window_id`, never by re-deriving a new split.**
- LOPO itself is the leakage control across recording projects/sites (no project appears in both train/val and test) `[VERIFIED: repo — CLAUDE.md "Cross-Validation" section, corroborated by run_splits project filtering]`.

**What the species pipeline must add on top, not replace:**
- Verify (test, §11) that joining `species_windows_segmented_v4.json` onto the existing split CSVs by `window_id` does not lose or duplicate rows, and does not introduce any `window_id` absent from `windows_mapping_4.0overlap_segmented_v4.json`.
- Embedding extraction must be **leakage-blind by construction**: embeddings are computed once, per window, independent of fold membership (no fold-specific normalization, no fold-specific fitting at the embedding stage — Perch 2 is frozen and pretrained externally, so there is no risk of embedding-level statistics leaking across folds, unlike a from-scratch feature normalizer).
- Calibration and thresholding (§8) must be fit **only** on each fold's own validation split — never on that fold's test project, and never pooled across folds (a per-fold calibrator, not one global calibrator, to respect LOPO's project-generalization framing).
- If a per-species minimum-support gate is applied for calibration/threshold-fitting, the gate must be evaluated **within each fold's train+val only**, not using knowledge of the held-out test project's label distribution (which would be a subtle form of test-set peeking).

## 10. Ablation matrix

| # | Axis | Levels | Question answered |
|---|---|---|---|
| A1 | Pooling of Perch embedding | `mean`, `first`, `max`, `flatten` (5×3×1536) | Does discarding within-window temporal structure (pooling) cost detection performance for short/edge-of-window calls? |
| A2 | Embedding model version | `perch_v2` (1536-d) vs `perch_8` (1280-d, older Perch) | Sanity check that Perch 2 embeddings are actually better than the previous generation for this dataset, not merely different. |
| A3 | Linear head formulation | scikit-learn per-species L2-regularized logistic regression (OvR) vs. a single shared linear layer trained jointly with multilabel BCE | Does joint training (shared regularization across species) change ranking vs. fully independent per-species fits? |
| A4 | Regularization strength | L2 sweep (config-driven, not hardcoded) | Standard model-selection ablation; also diagnostic for overfitting given small per-species N. |
| A5 | Ambiguous-window policy | `exclude` (default) vs `negative` (treat as background) vs `separate_class` (auxiliary "unresolved-bird" output) | Quantifies the cost of the label-noise shortcut a less careful pipeline would take by default. |
| A6 | Evaluation population | `all_windows` vs `bird_positive_only` | Separates "everything" performance from the practically relevant cascade-conditioned performance (§8). |
| A7 | Minimum species support threshold | e.g. ≥1, ≥5, ≥10, ≥20 test-fold positives | Shows how sensitive the headline macro-AP number is to which rare species are included — required for an honest single-number claim. |
| A8 | Zero-shot vs. trained | Perch 2's own ~10k-class bird logits, restricted to crosswalk-matched species (Phase 3, no training) vs. our trained linear probe | Establishes whether training a probe on PteroSet's own labels adds value over the foundation model's out-of-the-box predictions — the single most important ablation for justifying that this project's investment is warranted. |

Each ablation is run across all 5 LOPO folds (not a single fold) so its result also carries the §8 statistical-summary treatment, not a single point estimate.

## 11. Reproducibility requirements

- **Environment isolation** (`[RECOMMENDATION]`, driven by verified fact that Perch 2 requires TensorFlow while the existing stack is PyTorch/Lightning `[VERIFIED: repo — requirements.txt has no tensorflow entry]`): run embedding extraction in a separate virtual environment/process from linear-probe training. The frozen-embedding artifact (§7) is the sole interface between them — this also means a TF/CUDA version conflict in one environment can never silently corrupt the other's results.
- **Pin exact versions**: `perch-hoplite` package version (or git commit, since it is under active development — file headers observed dated 2026 `[VERIFIED: source]`), TensorFlow version, `kagglehub` version, resolved Kaggle model path/version string (from `kaggle_hub.resolve()`), all recorded in `embeddings/perch_v2/segmented_v4/manifest.json`.
- **Content-hash the source artifacts**, not just filenames: `species_windows_segmented_v4.json` and the embeddings manifest must record a hash of `windows_mapping_4.0overlap_segmented_v4.json` and `annotations_species.json` at generation time, so a silent upstream edit (as happened v1→v4 for PPA4 labels) is detectable rather than producing stale, mismatched artifacts.
- **Seed everything**: `species.random_state` in `species_config.yaml` must seed the calibration bootstrap, any stochastic linear-head solver (e.g. `sklearn`'s `saga`/`lbfgs` seeding where applicable, torch RNG if the joint-training ablation A3 is used), and the bootstrap CI resampling in `evaluate_species.py`.
- **Determinism check**: extracting embeddings twice for the same window must produce bit-identical (or float-tolerance-bounded, documented tolerance) output — a required Phase 2 test, since TF inference determinism across CPU/GPU backends is not guaranteed by default and must be verified, not assumed.
- **Config/commit logging**: every `outputs_species_v1/fold_k_*/metrics.json` must embed the resolved `species_config.yaml` contents and the git commit hash of the `species/` package at run time (mirrors the existing project's general discipline of avoiding silent config drift).

## 12. Phased milestones and acceptance criteria

| Phase | Deliverable | Acceptance criteria (go/no-go) |
|---|---|---|
| 0 — Audit | `species_taxonomy_crosswalk.csv`, `reports/species_v1/species_label_audit.md` | Duplicate codes merged; family/order/unresolved ranks explicitly excluded or, if kept, kept as a clearly separate non-species axis; 14,205/6,703-vs-actual discrepancy reconciled and dated; per-species/per-project count table produced. **No Phase 1 work starts until this report exists and is reviewed.** |
| 1 — Label derivation | `species_windows_segmented_v4.json`, per-fold `*_species.csv` | Every `window_id` in `windows_mapping_4.0overlap_segmented_v4.json` has exactly one provenance state; join against `folds_segmented_v4` loses/duplicates zero rows (tested, §13); documented % of windows in each provenance state; documented species-count distribution post-crosswalk. If fewer than a usable number of species meet any reasonable minimum-support threshold (judgment call deferred to the actual audit numbers, not pre-guessed here), flag for a scope discussion (e.g. genus-level fallback) before proceeding. |
| 2 — Embedding extraction | `embeddings/perch_v2/segmented_v4/` + manifest | CPU and/or GPU Perch 2 path runs successfully in this repo's actual compute environment (resolves the HF-card-vs-source-code CPU/GPU discrepancy empirically); resample+slice produces exactly 160,000-sample inputs for a full audit of all windows (not a sample); repeat-run determinism check passes within documented tolerance; Kaggle download/auth friction resolved and documented. |
| 3 — Zero-shot baseline | `reports/species_v1/zero_shot_results.csv` | Crosswalk between PteroSet species codes and Perch 2's own class list established (via `perch_hoplite.taxonomy`, `[OPEN QUESTION]` — exact crosswalk mechanism to confirm in Phase 3); baseline numbers reported per-project, no training involved. |
| 4 — Linear probe + calibration | `outputs_species_v1/fold_k_*/` | All 5 folds trained without solver-convergence failures (or failures explicitly logged per-species, never silent); calibration fit only on val (audited); primary metric beats the Phase 3 zero-shot baseline on a majority of folds — if not, stop and diagnose (labels? embeddings? probe capacity?) before running the full ablation matrix. |
| 5 — Ablations + aggregation | `reports/species_v1/species_ablation_results.csv`, `species_cv_results.csv` | All 8 ablation axes (§10) executed across all 5 folds; cross-fold statistical summary (mean±std, per-project table, bootstrap CIs) present for every reported number — a single pooled number without these is treated as incomplete, not publishable. |
| 6 — Write-up handoff | Design doc finalized, handed to Documenter agent for methods-section drafting | Every "verified" claim in this document re-checked against the final, frozen `species_config.yaml`/code at handoff time (facts can drift while Phases 0–5 execute). |

## 13. Tests (required, not optional)

- `test_species_label_overlap.py`: synthetic sound with hand-constructed species/identification annotations and windows; assert exact expected `provenance_state` and `species_present` for boundary cases (annotation exactly touching window edge, annotation fully inside, annotation spanning multiple windows, two co-occurring species in one window).
- `test_taxonomy_crosswalk.py`: assert `RAMTUC`/`RHATUC` and `ATRPIL`/`ATAPIL` map to the same canonical code; assert `PSITTACIDAE`/`PSITTACIFORMES`/`RHACAR`/`PICIDA_1`/`TYRANN_SP1`/`PSITTA` are excluded (or tagged non-species-rank) from the species-classification class list.
- `test_embedding_extraction_contract.py`: assert every extracted embedding corresponds to a resampled 160,000-sample, unit-scaled input; assert shape is `(1536,)` (pooled) or `(5,3,1536)` (spatial) as configured; repeat-extraction determinism within tolerance.
- `test_fold_join_integrity.py`: for each of the 5 folds, assert the joined `*_species.csv` has exactly the same `window_id` set and row count as the existing `*_split.csv`; assert no `sound_id` appears in more than one of {train, val, test} within a fold (re-asserts the existing leakage invariant, now also checked from the species pipeline's own entry point rather than trusted blindly).
- `test_species_metrics.py`: hand-computed toy multilabel examples (small K, small N) checked against macro-AP, micro-AP, LRAP, per-species ROC-AUC implementations, including the "support=0 → N/A, not 0" behavior.

## 14. Failure modes and mitigations

| # | Failure mode | Mitigation |
|---|---|---|
| R1 | TensorFlow (Perch 2) and PyTorch/CUDA coexisting in one process/env causes driver or CUDA-version conflicts | Isolate embedding extraction in its own environment/process; interface only via the frozen embeddings artifact (§6, §11). |
| R2 | Kaggle model download requires authentication/license click-through unavailable in this environment | Smoke-test in Phase 0/2 before committing to the full extraction run; document the exact credential requirement; consider the Hugging Face mirror (`cgeorgiaw/Perch`) as a fallback if `kaggle_hub` access is blocked — but note the HF mirror's own GPU-only caveat must then be re-verified (§3). |
| R3 | Hand-rolled preprocessing (resampling/normalization) silently diverges from Perch 2's expected input distribution | Use `perch_hoplite`'s own `frame_audio`/`normalize_audio`/`embed()` path rather than reimplementing peak-normalization; add a unit test asserting our resample+normalize output matches what `model.embed()` internally expects (target_peak=0.25). |
| R4 | Per-species metric silently reported as 0 when a species has zero positives in a held-out project | Explicit N/A/support=0 handling (§5, §8, tested in §13) — never let `0/0` collapse to `0`. |
| R5 | Duplicate/rank-mixed species codes inflate apparent class count and confuse the confusion matrix | Mandatory crosswalk (§4, §9 Phase 0 gate) before any labels are generated. |
| R6 | Ambiguous (`species-positive-but-unresolved`) windows silently used as hard negatives | Default `exclude` policy (§5), with `negative` only as an explicit, reported ablation (A5), never the unexamined default. |
| R7 | Rare species (long tail) cause unstable/degenerate linear-solver fits (all-negative predictor) | Log per-species train positive counts before fitting; skip/flag species below a configured minimum rather than silently fitting and reporting a meaningless AUC/AP. |
| R8 | Embedding cache goes stale relative to an updated `windows_mapping_*`/`annotations_species.json` (exactly the class of bug CLAUDE.md documents for v1→v4) | Content-hash provenance in the manifest (§11); regeneration must be forced, never silently reused, when hashes mismatch. |
| R9 | Val split has zero positives for some species, breaking per-species calibration | Explicit fallback to "uncalibrated, flagged" reporting (§8) rather than a crashing or silently-skipped calibrator. |
| R10 | Dataset's own documented annotation counts (14,205/6,703) don't match what's regenerated locally, and this goes unnoticed until publication | Phase 0 audit explicitly reconciles and dates this (§4 item 2) before any downstream artifact is built. |

## 15. Decision gates (explicit go/no-go)

1. **Before Phase 1**: Phase 0 audit report exists, is reviewed, and the crosswalk resolves all identified duplicate/rank-mixing issues.
2. **Before Phase 2 full-scale run**: a small-scale smoke test (e.g. one fold's worth of windows) confirms the Perch 2 environment runs on this repo's actual compute (resolving the CPU/GPU discrepancy) and that determinism/shape assertions pass.
3. **Before Phase 4**: label-coverage statistics from Phase 1 (species count meeting minimum support, % ambiguous windows) are judged sufficient for a linear probe to be a meaningful experiment — if not, escalate to a scope discussion rather than proceeding on underpowered classes.
4. **Before Phase 5 (full ablation matrix)**: Phase 4's trained probe beats the Phase 3 zero-shot baseline on a majority of folds for the primary metric. If it does not, do not proceed to ablations — diagnose labels/embeddings/probe capacity first (an ablation matrix built on a broken baseline produces eight ways to misinterpret the same bug).
5. **Before any publication-facing claim**: every number is reported with the §8 statistical-summary treatment (per-project table + between-project spread + within-fold bootstrap CI); no single pooled point estimate is presented alone.

## 16. Self-critique (Inquisitor-style stress test of this design)

I ran this pass myself, since no live Inquisitor agent could be spawned in this session (§13 delegation note below). Genuine open questions, not rhetorical:

- **"Why exclude ambiguous windows instead of modeling them as a genuine third class (`unresolved-bird`)?"** — Excluding them is the safer default (avoids injecting a false negative signal), but it also discards real information (a detector could plausibly learn "bird present, species unclear" as a useful output). This is exactly why A5 makes `separate_class` an ablation rather than dismissing it — I do not claim `exclude` is proven superior, only that it is the more defensible *default* given the risk of silent label noise.
- **"Why trust that Perch 2's 5.0 s/32 kHz window aligns with PteroSet's 5.0 s/48 kHz window, rather than re-verifying per-window?"** — I only verified this on a handful of sample windows read directly from the JSON, not on all 160,244. Phase 0/2 must assert this on the full set (§3 OPEN QUESTION, §12 Phase 1/2 acceptance criteria) — I am not claiming full verification here.
- **"Why is macro-AP over a minimum-support-filtered species set the primary metric, rather than something the Perch 2 authors themselves report (e.g. mAP over all classes) for comparability with BirdSet/BEANS benchmarks?"** — Comparability with external benchmarks is a legitimate secondary goal I have not built into this plan; if it becomes a project priority, add a specific "benchmark-comparable mAP" secondary metric computed identically to BirdSet/BEANS's own protocol (would require reading those protocols directly, not assuming they match ours — flagged as future work, not fabricated here).
- **"Is reusing `folds_segmented_v4` for a *different* task (species vs. binary) actually still leakage-safe, or does the binary-task-optimized split have some species-specific blind spot?"** — The `GroupShuffleSplit`-by-`sound_id` and LOPO-by-project logic is task-agnostic (it never looked at labels when forming groups, only at `sound_id`/`project`), so it remains valid for species labels. The one thing it does *not* control for is per-species representation across the train/val boundary within a fold (a species could be very unevenly split between train and val purely by chance) — this is a real limitation, addressed in §11's minimum-support-for-calibration gate, but not eliminated.
- **"Doesn't restricting the primary metric to `bird_positive_only` (A6) implicitly assume the binary detector is reliable, which is itself unproven for species-level granularity?"** — Yes; that is precisely why both populations (§8) are reported, not just one — `all_windows` is the population-agnostic, more conservative number.

## 17. Delegated Investigation Log

This session's toolset (`view`, `grep`, `glob`, `web_fetch`, `web_search`, `create`) does not include a task-spawning primitive for independent sub-agent processes. Rather than fabricate a multi-agent transcript, I performed four focused, clearly-scoped investigative passes myself, in the spirit of the roles they would otherwise occupy:

1. **Data-Validator pass** → §4 (label provenance audit): read `species.csv`, both annotation JSONs, and `windows_mapping_4.0overlap_segmented_v4.json` directly; found the duplicate-code, rank-mixing, and count-discrepancy issues by grep/diff, not inference.
2. **Experiment-Guard pass** → §9 (leakage/reproducibility of existing splits): read `prepare_dataset.py::run_splits` in full to confirm the `GroupShuffleSplit`-by-`sound_id` and non-overlapping-test-window mechanisms actually exist as CLAUDE.md claims, rather than trusting the documentation alone.
3. **Research-Explorer pass** → §3 (Perch 2 API/license verification): fetched `google-research/perch`, `google-research/perch-hoplite` (README, `LICENSE`, `model_configs.py`, `zoo_interface.py`, `taxonomy_model_tf.py`) and the Hugging Face model card directly; flagged the HF-card-vs-source-code GPU/CPU discrepancy rather than picking one silently.
4. **Inquisitor pass** → §16 (self-critique): applied the Inquisitor's WHY/HOW/WHAT-IF framework against this document's own recommendations before finalizing it.

**Recommendation for Round 2**: if a genuine multi-agent orchestration capability is available in the broader system (outside this tool-constrained session), route the Phase 0 audit through a dedicated Data-Validator pass with `Bash` access to actually execute the per-species/per-project count queries (this document specifies *what* must be computed in §4/§9 acceptance criteria, but the exact numbers were out of reach without a code-execution tool here — do not treat any count in §4 beyond the ones explicitly marked `[VERIFIED: repo]` as final).

---

## Appendix A: Verified-facts ledger (quick reference)

| Claim | Status |
|---|---|
| `train.py` multiclass path is single-label (softmax-style), not multilabel | VERIFIED (repo) |
| Species co-occurrence exists in real annotations (two `CYAVIO` events near each other) | VERIFIED (repo) |
| `annotations_identification.json` = 15,372 annotations, single category `AVEVOC` | VERIFIED (repo) |
| `annotations_species.json` = 6,702 annotations | VERIFIED (repo) |
| Dataset's own description claims 14,205 / 6,703 | VERIFIED (repo, `info.description` string) — **mismatched** vs. above, unresolved |
| `RAMTUC`/`RHATUC` and `ATRPIL`/`ATAPIL` duplicate codes | VERIFIED (repo) |
| `PSITTACIDAE`/`PSITTACIFORMES`/`RHACAR`/`PICIDA_1`/`TYRANN_SP1`/`PSITTA` are non-species-rank codes | VERIFIED (repo) |
| PteroSet windows are 5.0 s, 240,000 samples @ 48 kHz (sampled entries) | VERIFIED (repo) |
| Perch 2: 5.0 s / 32 kHz / 1536-d pooled, `(5,3,1536)` unpooled | VERIFIED (source) |
| Perch 2 preprocessing: `target_peak=0.25` peak normalization | VERIFIED (source) |
| `perch_v2_cpu` preset exists in `perch-hoplite` source | VERIFIED (source) |
| HF model card claims GPU-only, "CPU variant... soon" | VERIFIED (source) — **contradicts** the point above; treated as stale, to be resolved empirically |
| perch-hoplite code license: Apache-2.0 | VERIFIED (source) |
| Perch 2 model *weights* license = Apache-2.0 | NOT independently verified (Kaggle page unreachable in this session) — re-check before publication |
| Model authors' own statement that species logits are uncalibrated/unreliable for rare species | VERIFIED (source) |
| `folds_segmented_v4` train/val split is `GroupShuffleSplit` grouped by `sound_id`; test is non-overlapping-window-only | VERIFIED (repo) |
| Existing evaluation script resolves audio via `data/audios_48khz` by basename, not the JSON's literal `file_name_path` | VERIFIED (repo) |

STATUS: DONE
