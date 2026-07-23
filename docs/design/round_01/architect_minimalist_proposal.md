# Species Classification via Perch 2 Linear Probing - Minimalist Proposal

Author: architect-minimalist (Round 1)
Date: 2026-07-23
Status: Draft for review (Round 1 of design team)
Perspective: smallest scientifically valid baseline; minimal new machinery; rapid diagnosis; explicit go/no-go gates. This proposal actively pushes back on scope that is not required to get a first trustworthy answer to "do Perch 2 embeddings carry species signal for PteroSet?"

---

## 0. How to read this document

Every factual claim is tagged:
- [VERIFIED: repo] - confirmed by reading files in this repository (path given).
- [VERIFIED: source, date] - confirmed by reading an external authoritative source (URL + fetch date given).
- [RECOMMENDATION] - my proposed design choice, not a fact.
- [UNVERIFIED - must check in Phase 0] - a claim I could not confirm from the repo or from external docs and that must be checked empirically before it is relied on.

I did not have direct shell/Python execution access, so all "facts" about the repo are from reading files, not from running code. Anything that requires running code (timing, exact per-class counts, exact embedding tensor shape from a live model call) is marked [UNVERIFIED - must check in Phase 0] and is the first thing Phase 0 must produce.

---

## 1. Why

[VERIFIED: repo] The current pipeline (train.py, prepare_dataset.py) is a spectrogram-CNN stack: BioacousticsDataset loads precomputed .npy mel-spectrograms, ResNetClassifier is a CNN trained end-to-end, and num_classes in train.py already supports "multiclass" - but there is no multi-label handling, no species-level label derivation, and no notion of "unknown species" anywhere in the current code. Species classification is a genuinely new objective, not a flag flip on the existing trainer.

[VERIFIED: repo] Species-level ground truth is much sparser than the binary bird-presence signal the project has been trained on so far:
- windows_mapping_4.0overlap_segmented_v4.json: 160,244 windows, 35,073 binary-positive (CLAUDE.md).
- data/annotations_species.json: 6,702 annotations with a species-level determination (counted anno_id occurrences in that file), against data/annotations_identification.json: 15,372 total bird-event annotations (counted anno_id occurrences in that file) - i.e. the large majority of "there is a bird here" events have no species label at all. The Zenodo record's stated figures (14,205 total / 6,703 species-resolved) are close but not identical to what I counted directly in the two JSON files [UNVERIFIED - must check in Phase 0]; the discrepancy must be resolved (likely a stale description string vs. the shipped JSON) before trusting any coverage number downstream.
- data/species.csv (169 rows) contains at least 6 non-species placeholder codes - PSITTACIDAE, PSITTACIFORMES, RHACAR, PICIDA_1, TYRANN_SP1, PSITTA (family/order/genus-level or fully unresolved, "species" column literally "-" or a family name) - and at least one duplicated binomial mapping (RAMTUC and RHATUC both map to "Ramphastos tucanus") [VERIFIED: repo, data/species.csv]. A species classification task cannot use the raw code list as-is; it needs curation.

Given this, the two things that will most determine whether species classification succeeds are (a) whether the label pipeline correctly separates "no bird," "bird of known species," and "bird present, species unresolved," and (b) whether a fixed, off-the-shelf embedding already carries species-discriminative signal on this specific tropical soundscape. Both questions are answerable with a linear probe - the smallest model that can prove or disprove "the embedding contains the signal" without confounding the answer with head capacity, backbone fine-tuning dynamics, or optimizer/scheduler choices. This is also literally what Perch 2's own authors validate their embeddings with:

[VERIFIED: source, huggingface.co/cgeorgiaw/Perch, fetched 2026-07-23]: "The embeddings were trained with the goal of being linearly separable. For most cases training a simple linear classifier on top of the model's outputs should work well." The Perch 2.0 paper (arXiv:2508.04665, van Merrienboer et al., Google DeepMind/Google Research - [VERIFIED: source, web search abstract + secondary summaries, fetched 2026-07-23; full paper not read line-by-line, see risk in section 13]) reports state-of-the-art linear-probe transfer on BirdSet (a passive-monitoring soundscape benchmark) and BEANS.

So: do the smallest thing that tests the actual hypothesis. No backbone fine-tuning, no new heavy training framework, no re-segmentation of windows, no touching the existing binary-task artifacts.

---

## 2. What (scope)

In scope (this proposal, Phases 0-3):
1. A window-to-species multi-label derivation step that reuses the existing windows_mapping_4.0overlap_segmented_v4.json and data/folds_segmented_v4/ splits verbatim - no new window geometry, no re-running segment_windows.
2. A curated species class list with an explicit, documented exclusion list and minimum-support threshold.
3. Frozen Perch 2 embedding extraction, one 1536-d vector per existing v4 window, stored as a versioned artifact independent of any model training run.
4. A linear probe (logistic regression per class) trained per fold on frozen embeddings only.
5. Evaluation mirroring the existing plot_cv_results.py conventions (per-fold, per-project, PR/ROC-AUC based).
6. A minimal test suite for the label-join logic, the embedding artifact contract, and (as a free side-benefit) a leakage regression test against the existing fold CSVs.

Explicitly out of scope / rejected for Phase 0-3 (see section 11 for rationale):
- Fine-tuning the Perch 2 backbone.
- Multi-frame / longer-context embeddings, or combining Perch embeddings with the existing spectrogram-CNN features.
- Re-segmenting windows or changing window size/overlap.
- Using Perch's own 15,000-class classification head/logits as the species predictor (we train our own head on our own class list instead - Perch's head is uncalibrated for rare classes and not aligned 1:1 with PteroSet's species codes).
- perch-hoplite's "agile modeling" active-learning workflow - powerful, but it is a second product surface (vector DB, UI-driven labeling loop) we do not need to answer the linear-probe question.
- Joint multi-task training of bird-presence + species in the CNN - bigger surface area, harder to isolate failure causes; revisit only after the linear-probe baseline is trustworthy.

---

## 3. Facts verified about Perch 2 (external sources)

All fetched 2026-07-23 from the sources cited; no signature or claim below is guessed.

| Fact | Value | Source |
|---|---|---|
| License | Apache License 2.0 | HF model repo YAML front-matter, huggingface.co/cgeorgiaw/Perch/raw/main/README.md |
| Input | 5-second mono waveform, 32,000 Hz (160,000 samples), unit-scaled | HF model card; zoo_interface.EmbeddingModel.embed docstring, perch-hoplite/perch_hoplite/zoo/zoo_interface.py (GitHub, main branch) |
| Pooled embedding dim | 1536 | HF model card; confirmed in code: PresetInfo.embedding_dim = 1536 for PERCH_V2/PERCH_V2_GPU/PERCH_V2_CPU in perch_hoplite/zoo/model_configs.py |
| Unpooled/spatial embedding | shape (frames, 3, 1536) per HF card; code's InferenceOutputs.embeddings is generically [Frames, Channels, Features] (zoo_interface.py) | HF model card + code |
| Backbone | EfficientNet-B3, ~12M params (embedding model); + ~91M-param classification head over ~15,000 classes | HF model card |
| Logit outputs | ~15,000 classes (~10,000 birds), iNaturalist taxonomy + eBird 6-letter crosswalk asset | HF model card |
| Load API | perch_hoplite.zoo.model_configs.load_model_by_name('perch_v2') returns model; model.embed(waveform) returns InferenceOutputs(embeddings, logits, ...) | HF model card example; model_configs.py, zoo_interface.py |
| CPU path exists officially | ModelConfigName.PERCH_V2_CPU, Kaggle slug google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu, TF backend | perch_hoplite/zoo/model_configs.py (GitHub, main) |
| ONNX path exists | ModelConfigName.PERCH_V2_ONNX / PerchV2OnnxModel, uses onnxruntime, auto-selects CPUExecutionProvider if no CUDA provider | perch_hoplite/zoo/models_onnx.py |
| ONNX weight provenance | Downloaded from justinchuby/Perch-onnx on Hugging Face Hub - a third-party conversion, not an official Google Kaggle artifact; class list is still pulled from the official Kaggle labels.csv | perch_hoplite/zoo/models_onnx.py, PerchV2OnnxModel.resolve_config |
| Package / deps | perch-hoplite v1.0.2, requires-python>=3.10,<3.15; core deps are TF/JAX-free; tf extra = tensorflow>=2.20,<3.0; tf-cuda extra = tensorflow[and-cuda]>=2.20,<3.0; onnx extra = onnxruntime>=1.20; base deps include numpy>=2.0,<3.0 | perch-hoplite/pyproject.toml (GitHub, main) |
| Model download mechanism | kagglehub.model_download(...) for TF variants (requires network/Kaggle access at first run, then local cache); hf_hub.download(...) for the ONNX weights | perch_hoplite/zoo/kaggle_hub.py, models_onnx.py |
| Training data | Xeno-Canto, iNaturalist, Animal Sound Archive, FSD50k | HF model card |

Discrepancy flagged, not resolved by me: the HF model card text says "This version of the model requires TensorFlow 2.20.rc0 and a GPU... A CPU variant will be added soon," but the perch-hoplite source on GitHub main already ships a working PERCH_V2_CPU preset with its own Kaggle slug. The code is more current than the card's prose. [RECOMMENDATION] trust the code, but confirm in Phase 0 by actually loading perch_v2_cpu and running a forward pass on CPU before committing to a compute plan.

What I could not verify (no execution environment): the exact wall-clock/CPU-vs-GPU throughput of perch_v2/perch_v2_cpu on this cluster, whether kagglehub/Kaggle credentials are already configured in this environment, whether the ONNX model's outputs numerically match the official TF model closely enough to trust as a fallback, and the exact conda env's current numpy/tensorflow state. All of these are Phase 0 tasks (section 7).

---

## 4. Data / label schema (new artifacts, Phase 1)

### 4.1 Provenance chain (must be joined correctly - this is the crux of the label pipeline)

[VERIFIED: repo]
- data/annotations_identification.json: single category AVEVOC/BIO (i.e., every annotation in this file is "a bird made a sound here"); 15,372 anno_id entries. This is the file the existing binary window label is derived from (prepare_dataset.py::run_segment_windows, overlap rule: a_min < window_end_sec and a_max > window_start_sec).
- data/annotations_species.json: subset with a resolved species code; 6,702 anno_id entries; categories = the 169 rows of species.csv.
- Critical detail: data/data_reader.py::add_annotations assigns anno_id as a local incrementing counter per output file (anno_id = 0 at the start of each run, incremented only for matched rows). This means anno_id values are not comparable across the two JSON files - a species-file anno_id=5 is not the same underlying RAVEN annotation as an identification-file anno_id=5. The only safe join key is the triple (sound_id, t_min, t_max), since both files copy these fields verbatim from the same source RAVEN rows without transformation.
- Phase 1 must validate the join: every one of the 6,702 species annotations should find exactly one match in the identification file on (sound_id, t_min, t_max). If the match rate is not ~100%, that is a blocking bug to fix before any label is trusted (float round-trip through json.dump/json.load is bit-exact in Python, so a mismatch would indicate a real data problem, not a floating-point artifact).

### 4.2 Per-window label schema (new artifact: data/species_labels_segmented_v4.csv)

One row per window_id in windows_mapping_4.0overlap_segmented_v4.json (160,244 rows, same set - no window is added or removed).

| Column | Type | Definition |
|---|---|---|
| window_id | int | Foreign key into the v4 windows mapping and fold CSVs. |
| species_CODE (one column per class) | 0/1 | 1 iff a species-level annotation for CODE overlaps the window (a_min < window_end_sec and a_max > window_start_sec), using the same overlap rule as the existing binary label for consistency. |
| any_species_known | 0/1 | 1 iff at least one species_CODE column is 1. |
| has_bird | 0/1 | Copied from the existing v4 binary label field - "any AVEVOC annotation overlaps," independent of species resolution. |
| unresolved_bird | 0/1 | 1 iff has_bird==1 and at least one identification-level annotation overlaps this window that has no matching species-level annotation (i.e., a bird is present in the window whose species is not determined). This is the "unknown/background" case - not a negative for any class. |
| usable_strict | 0/1 | 1 iff unresolved_bird==0 (i.e., every bird annotation overlapping this window, if any, is species-resolved, or there is no bird at all). Only usable_strict==1 windows are used for the primary (strict) train/eval pass. |

[RECOMMENDATION] This design (no-bird / clean single-or-multi species / ambiguous-species) is the minimal schema that avoids the single biggest correctness trap: silently treating "bird present, species unknown" windows as negatives for every species class, which would poison every class's negative set with an unknown number of false negatives-of-omission. Given that annotation density on positive test windows is 1.6-1.9 mean overlapping annotations per window (reports/window_counts_by_fold.md, v3-era numbers, order of magnitude still applicable to v4), co-occurring unresolved calls in an otherwise species-labeled window are expected to be common, not an edge case.

### 4.3 Curated class list (new artifact: data/species_classes_v4.json)

[RECOMMENDATION]
- Start from data/species.csv, exclude non-species placeholder codes: PSITTACIDAE, PSITTACIFORMES, RHACAR, PICIDA_1, TYRANN_SP1, PSITTA (family/order/unresolved-genus level, confirmed by inspection - "species" column is "-" or a bare family name, not a binomial). This list must be stored in the artifact with the reason for exclusion, not silently dropped.
- Do not attempt to merge RAMTUC/RHATUC automatically - flag as a data-quality note for the dataset maintainers and treat as two separate codes for Phase 1-3 (safer default; incorrect merging is worse than a harmless duplicate class).
- Apply a minimum-support threshold: keep a class only if it has >= K positive windows (not annotations) dataset-wide, where [UNVERIFIED - must check in Phase 0/1] the exact per-class window-level counts are unknown to me (I only have annotation-level totals, and one annotation can appear in multiple overlapping 1s-hop windows or share a window with other annotations - the mapping is not 1:1). K=30 is a reasonable, literature-typical starting point for a linear probe, but the real per-class table produced by Phase 1 must be inspected before Phase 3 locks in a class list - do not hardcode K from this document without looking at the table.
- Store, per retained class: total window count, per-project window count (needed to know if a class is even representable in a given held-out fold), and whether it survives with >=1 positive usable_strict window in every project (a class absent from a held-out project's test set cannot be evaluated in that fold - this is expected and must be handled by the evaluator, not treated as a bug).

---

## 5. Split and leakage policy

[VERIFIED: repo] Existing leave-one-project-out policy in prepare_dataset.py::run_splits, unchanged and reused as-is:
- Test = non-overlapping windows only (start % window_size_samples == 0) from the held-out project.
- Train/Val = GroupShuffleSplit on the remaining 4 projects, grouped by sound_id (val_size=0.15, random_state=42), so no audio file contributes windows to both train and val.
- 5 folds: fold_0_MAP1_segmented ... fold_4_PPA4_segmented.

[RECOMMENDATION] New leakage rules specific to the probe (nothing above changes; these are additive):
1. Join, don't resplit. species_labels_segmented_v4.csv and the embedding index are joined to the existing train_split.csv/val_split.csv/test_split.csv by window_id. No new grouping/shuffling logic is introduced, so the existing sound-file-level leakage guarantee is inherited unchanged.
2. Any fitted preprocessing (StandardScaler, per-class regularization strength C) is fit on that fold's train split only, applied to val/test, never re-fit or peeked at using test data. Model selection over C uses val, not test.
3. The curated class list (section 4.3) is chosen using global (all-project) frequency, not per-fold frequency. This is a deliberate, disclosed exception: which species exist is a task definition choice, not a fitted parameter. Doing per-fold class curation would produce five different, non-comparable task definitions. Mitigation: the per-fold train/test support table for every retained class is reported alongside every metric, so a class with near-zero test support in a given fold is visibly discounted rather than silently averaged in.
4. Perch 2's own training corpus vs. PteroSet. Perch 2's training data is Xeno-Canto/iNaturalist/Animal Sound Archive/FSD50k (public, community-contributed audio) - [UNVERIFIED - low-risk, worth a one-line check] whether any PteroSet recordings (private AudioMoth deployments, dataset "under submission" to Scientific Data per README.md) could already be present in a public corpus. Given the dataset is unpublished/under-review, this is unlikely, but it costs nothing to note explicitly rather than assume.

---

## 6. Architecture (How)

### 6.1 Design decision: do not extend the existing Lightning/CNN stack

Decision: New standalone species_probe/ scripts, not a mode of train.py.
- Context: train.py's SpectrogramDataModule/BioacousticsDataset/ResNetClassifier are built around loading .npy spectrograms and training a CNN end-to-end with augmentation, MixUp, LR scheduling, etc. None of that machinery is needed or correct for a frozen 1536-d vector.
- Options considered:
  (a) Shoehorn embeddings into BioacousticsDataset by reshaping the 1536-d vector into a fake "spectrogram" tensor so ResNetClassifier still "works." Rejected: actively misleading (a CNN over a reshaped embedding vector has no spatial meaning), adds indirection, and still drags in Lightning/GPU-trainer machinery for what should be a sklearn.linear_model.LogisticRegression.fit() call.
  (b) A new PyTorch nn.Linear + BCEWithLogitsLoss trained with Lightning, mirroring train.py's structure for consistency. Rejected as the default: heavier than necessary (GPU trainer, checkpoint callbacks, schedulers) for what is architecturally a convex per-class logistic regression problem with ~120k rows x 1536 features x <= a few dozen retained classes - this fits comfortably and reproducibly in scikit-learn on CPU in the same conda env train.py already uses (no new heavy dependency).
  (c) (Chosen) A small standalone species_probe/ directory: one script for label derivation, one for embedding extraction (isolated env, see 6.3), one for probe training (sklearn.linear_model.LogisticRegression, one-vs-rest across the curated class list), one for evaluation. Every script takes plain CLI args + reads the existing v4 fold CSVs; none of them touch train.py, prepare_dataset.py, or existing artifacts.
- Consequences: this is the smallest change that answers the actual question, is trivially inspectable (weights stored as plain .npy, not framework-specific pickles), and cannot regress the existing binary-task pipeline because it never imports from it. Risk: if the answer to "is a linear probe enough" is no, a second design pass is needed for a heavier model - acceptable, because that decision should be evidence-based, not assumed up front.

### 6.2 ASCII data-flow diagram

```
                          +---------------------------------------------+
                          |   EXISTING, UNCHANGED, REUSED VERBATIM       |
                          |                                               |
  audios_48khz/*.wav      |  windows_mapping_4.0overlap_segmented_v4     |
  annotations_identif.json|  .json  (160,244 windows, binary label)      |
  annotations_species.json|                                               |
  species.csv             |  data/folds_segmented_v4/fold_{0..4}_*/      |
  metadata.csv            |    {train,val,test}_split.csv (window_id,    |
                          |     sound_id, project, label, spec_name)     |
                          +---------------------------------------------+
        |                                   |
        |  (read-only, join key = window_id)|
        v                                   v
+---------------------------+      +----------------------------------+
| NEW  Phase 1               |      | NEW  Phase 2                      |
| build_species_labels.py    |      | extract_perch_embeddings.py       |
|                             |      | (isolated env -- see 6.3)          |
| join identification.json   |      | resample window audio             |
| + species.json on          |      |  48kHz -> 32kHz, 5s window         |
| (sound_id,t_min,t_max)     |      | model.embed(waveform) -> 1536-d   |
|                             |      |                                    |
| -> species_labels_         |      | -> embeddings_perch_v2_v4.npy     |
|    segmented_v4.csv        |      |    (160244, 1536) float32         |
| -> species_classes_v4.json |      | -> embeddings_perch_v2_v4_        |
|    (curated class list)    |      |    index.csv (row -> window_id)   |
+---------------------------+      | -> embeddings_perch_v2_v4_        |
              |                     |    manifest.json (model/version)  |
              |                     +----------------------------------+
              |                                   |
              +----------------+------------------+
                               v
                  +----------------------------------+
                  | NEW  Phase 3                       |
                  | train_linear_probe.py              |
                  |                                     |
                  | for fold in 0..4:                  |
                  |   join fold's {train,val}_split    |
                  |   + labels.csv + embeddings.npy    |
                  |   drop unresolved_bird windows      |
                  |   fit StandardScaler(train only)   |
                  |   fit LogisticRegression per class  |
                  |   (C selected on val)              |
                  | -> checkpoints_species_v4/fold_N/  |
                  |    {weights.npy, bias.npy,         |
                  |     classes.json, scaler.json}     |
                  +----------------------------------+
                               |
                               v
                  +----------------------------------+
                  | NEW  Phase 3                       |
                  | evaluate_linear_probe.py            |
                  |                                     |
                  | apply fold N's fitted probe to     |
                  | fold N's non-overlapping test set  |
                  | (strict pass: usable_strict only,  |
                  |  lenient pass: sensitivity check)  |
                  | -> reports/species_probe_v4/        |
                  |    per_class_ap.csv                |
                  |    fold_summary.csv                |
                  |    summary.md                       |
                  +----------------------------------+
```

### 6.3 Isolating the Perch dependency

[RECOMMENDATION] Embedding extraction is the only step that needs TensorFlow (or ONNX Runtime) and perch-hoplite. Everything downstream (labels, probe training, evaluation) is plain numpy/pandas/scikit-learn, already in requirements.txt. Given perch-hoplite's base deps pin numpy>=2.0,<3.0 and its tf extra pins tensorflow>=2.20,<3.0 [VERIFIED: source, pyproject.toml], and the existing bioacoustics conda env's exact numpy/torch/lightning version pins are [UNVERIFIED - must check in Phase 0], the minimal-risk choice is:
- A separate conda env (e.g. bioacoustics-perch) used only to run extract_perch_embeddings.py, installed with pip install "perch-hoplite[tf]" (or [onnx] as fallback). It never needs to run train.py or import PytorchWildlife.
- The main bioacoustics env and requirements.txt are not modified. species_probe/requirements-embed.txt documents the isolated env's pins separately.
- This directly serves "minimal new machinery": the blast radius of a TF version conflict is one throwaway env used for one offline batch job, not a change to the environment every other experiment in this repo depends on.

### 6.4 Proposed files

```
species_probe/
  build_species_labels.py        # Phase 1
  extract_perch_embeddings.py    # Phase 2 (run in isolated env)
  train_linear_probe.py          # Phase 3
  evaluate_linear_probe.py       # Phase 3
  requirements-embed.txt         # perch-hoplite[tf] (or [onnx]) pins, isolated env only
  config.yaml                    # small, standalone; does not grow data/config.yaml
  tests/
    test_build_species_labels.py
    test_embedding_artifact_contract.py
    test_fold_leakage.py         # generic regression test, also covers existing v1-4 folds
    test_linear_probe_eval.py

data/                                          # gitignored, generated (existing convention)
  species_labels_segmented_v4.csv              # NEW
  species_classes_v4.json                      # NEW
  embeddings/
    perch_v2_v4.npy                            # NEW  (160244, 1536) float32, ~940 MB
    perch_v2_v4_index.csv                      # NEW  row_idx -> window_id
    perch_v2_v4_manifest.json                  # NEW  model name/version, backend, date, lib versions

checkpoints_species_v4/fold_{0..4}/            # NEW, gitignored (mirrors checkpoints_v4/ naming)
outputs_species_v4/fold_{0..4}/                # NEW, gitignored (mirrors outputs_v4/ naming)
reports/species_probe_v4/                      # NEW, gitignored (mirrors reports/ convention)
```

[VERIFIED: repo, .gitignore] Current .gitignore covers data/*.json, data/*.csv, data/fold*/, outputs*, *.ckpt, *.pt, and reports/ wholesale - but does not have a rule for .npy files outside data/spectrograms/ (which is ignored by directory name, not extension) or for checkpoints_v*-style directories (only the *.ckpt/*.pt file extensions are ignored). [RECOMMENDATION] Add *.npy to .gitignore; if plain-numpy-array probe weights are adopted (section 6.1), no further checkpoint-directory rule is needed since those weights are themselves .npy. Verify with git status after the first embedding-extraction run produces no untracked ~1 GB blob warning.

### 6.5 CLI interfaces

```bash
# Phase 1 -- label derivation (pure Python, no ML deps, runs in the main env)
python species_probe/build_species_labels.py \
    --windows_json data/windows_mapping_4.0overlap_segmented_v4.json \
    --annotations_identification data/annotations_identification.json \
    --annotations_species data/annotations_species.json \
    --species_csv data/species.csv \
    --min_count 30 \
    --exclude_codes PSITTACIDAE PSITTACIFORMES RHACAR PICIDA_1 TYRANN_SP1 PSITTA \
    --out_labels data/species_labels_segmented_v4.csv \
    --out_classes data/species_classes_v4.json

# Phase 2 -- embedding extraction (isolated env `bioacoustics-perch`)
python species_probe/extract_perch_embeddings.py \
    --windows_json data/windows_mapping_4.0overlap_segmented_v4.json \
    --audio_root data/audios_48khz \
    --model_name perch_v2_cpu \
    --backend tf \
    --batch_size 64 \
    --limit 0 \
    --out_embeddings data/embeddings/perch_v2_v4.npy \
    --out_index data/embeddings/perch_v2_v4_index.csv \
    --out_manifest data/embeddings/perch_v2_v4_manifest.json

# --limit N restricts to the first N windows; used for the Phase 0 timing spike
# and for fast CI smoke tests, without downloading/running on the full 160,244 windows.

# Phase 3 -- linear probe training (main env, sklearn only)
python species_probe/train_linear_probe.py \
    --fold_dir data/folds_segmented_v4 \
    --fold 0 \
    --labels_csv data/species_labels_segmented_v4.csv \
    --classes_json data/species_classes_v4.json \
    --embeddings data/embeddings/perch_v2_v4.npy \
    --embeddings_index data/embeddings/perch_v2_v4_index.csv \
    --exclude_ambiguous true \
    --standardize true \
    --C_grid 0.01 0.1 1.0 10.0 \
    --ckpt_dir checkpoints_species_v4 \
    --cross_validation      # loops folds 0-4 if --fold omitted, mirrors train.py's UX

# Phase 3 -- evaluation
python species_probe/evaluate_linear_probe.py \
    --fold_dir data/folds_segmented_v4 \
    --ckpt_dir checkpoints_species_v4 \
    --labels_csv data/species_labels_segmented_v4.csv \
    --classes_json data/species_classes_v4.json \
    --embeddings data/embeddings/perch_v2_v4.npy \
    --embeddings_index data/embeddings/perch_v2_v4_index.csv \
    --out_dir reports/species_probe_v4
```

---

## 7. Phased milestones with go/no-go acceptance criteria

### Phase 0 -- Feasibility spike (target: <= 1 day)
Goal: de-risk every [UNVERIFIED] item above before writing the real pipeline.
- Install perch-hoplite ([tf] extra) in a throwaway env; confirm Kaggle Hub download works in this network environment (or requires credentials/manual download - document whichever is true).
- Load perch_v2_cpu (or perch_v2 if a GPU is trivially available), call .embed() on 5-10 real PteroSet 5s clips (resampled 48k->32k), and empirically confirm the returned InferenceOutputs.embeddings shape/dtype and whether .pooled_embeddings(...) is needed to get a flat 1536-vector, or whether a single 5s input already returns exactly one frame.
- Time the call on --limit 200 windows; record CPU (and GPU if available) throughput.
- Decide TF-CPU vs TF-GPU vs ONNX backend based on the above, and record the decision + numbers in species_probe/PHASE0_NOTES.md.
- Acceptance / go-no-go: a real (1536,) embedding vector is produced end-to-end from an actual PteroSet audio window, with no NaNs, and a documented throughput number exists. If Kaggle/network access is blocked in this environment, STOP and escalate - the rest of the plan is void without a way to fetch the model weights.

### Phase 1 -- Label pipeline
- Run build_species_labels.py; verify output row count == 160,244 (matches v4 windows exactly).
- Verify the (sound_id, t_min, t_max) join between the two annotation files matches ~100% of the 6,702 species annotations against the identification file; investigate any gap.
- Spot-check 3 of the most frequent classes by manually tracing a handful of windows against the raw annotation JSON (by hand, not automated) to catch off-by-one/overlap-rule bugs.
- Produce and inspect the per-class x per-project window-count table; only then finalize --min_count.
- Acceptance: row-count match, >=99% join match rate (any gap explained), and a written table of retained classes with per-project support, reviewed before Phase 2 embeddings are spent on windows that will map to zero usable classes.

### Phase 2 -- Embedding extraction
- Run full extraction (all 160,244 windows) in the isolated env.
- Acceptance: output shape (160244, 1536), dtype float32, zero NaN/Inf rows, manifest.json records model name/version/backend/date/library versions, and a spot-check nearest-neighbor sanity check (two windows containing the same dominant species should be closer in cosine similarity than two windows with very different content) is not obviously violated - this is a sanity smell-test, not a metric to optimize.

### Phase 3 -- Linear probe + evaluation (the actual go/no-go for the whole idea)
- Train per-class logistic regression on frozen embeddings for the curated class list, 5 folds, usable_strict windows only, C selected on val.
- Evaluate on each fold's non-overlapping test split (existing leakage-safe test definition).
- Report macro-AP, micro-AP, per-class AP/ROC-AUC, and two baselines: (a) class-prior/majority baseline, (b) a "clean-window" sensitivity subset restricted to windows with exactly one overlapping annotation (upper-bound estimate, since 5s multi-species blending is a plausible confound).
- Go criterion: macro-AP over classes with >=1 test-fold positive beats the prior baseline by >=0.05 absolute AP on at least 3 of 5 folds. This threshold is a proposal for the design team to ratify, not a claimed scientific constant - record it as a pre-registered number before running Phase 3, so results can't be rationalized after the fact.
- No-go path: if the criterion fails, do not immediately reach for a bigger model. Run the failure-analysis checklist in section 9 first - a failing linear probe is diagnostic information, not just a bad result.

### Phase 4 (conditional -- only if Phase 3 passes)
Expand class coverage, calibrate per-class thresholds, integrate a summary into reports/ matching plot_cv_results.py's output style, write a comparison doc against the binary-task baseline. Not designed in detail here - premature until Phase 3's answer is in hand.

---

## 8. Metrics

[RECOMMENDATION], chosen to match the existing repo's evaluation vocabulary (plot_cv_results.py already uses precision_recall_curve/auc from sklearn.metrics for the binary task):
- Primary: per-class Average Precision (area under PR curve), macro-averaged over classes with usable test support, and micro-averaged (pooled) AP.
- Secondary: per-class ROC-AUC (comparable across classes with very different base rates; also lets us cite Perch 2's own paper claims - BirdSet ROC-AUC > 0.9 - as a rough external reference point, not a guarantee for this dataset).
- Reported but not decision-driving: subset accuracy / Hamming loss at a default 0.5 threshold - explicitly called out as low-information under the expected class imbalance, included only for continuity with readers used to accuracy-style numbers.
- Every metric is reported per fold and with the per-class train/test support table alongside it, so a high macro-AP that is actually driven by 2 well-populated classes is visible, not hidden in an average.

---

## 9. Failure analysis (what to check if Phase 3's go/no-go fails)

1. Label-join bug (wrong (sound_id,t_min,t_max) match, or overlap-rule off-by-one). Check first - cheapest to rule out, and silently wrong labels invalidate everything downstream. Re-run the Phase 1 spot-checks on a larger random sample.
2. Embedding pipeline bug (wrong resample factor, wrong normalization/peak-scaling, wrong pooling axis, or using the un-pooled (5,3,1536) tensor where a pooled (1536,) vector was intended). Re-run the Phase 0 nearest-neighbor sanity check with more pairs.
3. Insufficient positive support per class - if the highest-support classes (>=100 usable_strict positive windows) also show poor AP, this is not a support problem; if only low-support tail classes fail while high-support classes succeed, this is a support problem, and the fix is more labeled data or a different low-shot method - not immediately more model capacity.
4. Domain shift: Perch 2's training corpus is dominated by focal recordings and community-contributed audio; PteroSet is dense tropical passive-monitoring soundscape audio with frequent multi-species overlap. If even high-support classes underperform the paper's BirdSet numbers by a wide margin, suspect domain mismatch as the primary driver, not a bug - this would be evidence for eventually fine-tuning, not evidence that the pipeline is broken.
5. Window granularity vs. call duration: 5-second windows with 1s hop routinely contain multiple overlapping species (mean 1.6-1.9 annotations per positive window, historically). The "clean-window" sensitivity subset (section 7, Phase 3) directly measures how much this hurts by comparing AP on single-annotation windows vs. all usable_strict windows.

---

## 10. Tests

- test_build_species_labels.py: synthetic mini identification.json/species.json fixtures with hand-computed expected species_CODE, unresolved_bird, usable_strict columns for a handful of constructed overlap cases (no overlap, exact boundary touch, multi-species window, species+unresolved co-occurrence). Pure Python, fast, runs in CI.
- test_embedding_artifact_contract.py: given a tiny --limit 5 extraction output (or a stubbed/mocked embed function if the real model isn't available in CI), assert shape (5, 1536), dtype float32, no NaN, and that index.csv row order matches the .npy row order by window_id. Marked so it's skippable in CI environments without model/network access, matching the repo's own research-rigor guidance to never let a blocked external dependency stall the process.
- test_fold_leakage.py: a general-purpose regression test, not specific to the species task - assert, for every existing fold in data/folds_segmented_v4/, that no sound_id appears in both train_split.csv and val_split.csv, and that every row in test_split.csv satisfies start % window_size_samples == 0. This is a free, cheap addition that protects the leakage guarantees this whole design leans on, independent of whether species classification proceeds.
- test_linear_probe_eval.py: integration test - run train_linear_probe.py + evaluate_linear_probe.py on fold 0 with a tiny synthetic embeddings array (random vectors, not real Perch output) and a 2-class synthetic label set, assert output report files exist with expected columns and all metrics in [0,1]. This tests the plumbing, not the science - it should pass even before Phase 0/2 produce real embeddings, and should be written first.

---

## 11. Alternatives considered and rejected

| Alternative | Why rejected (for now) |
|---|---|
| Fine-tune Perch 2 backbone end-to-end | Premature complexity; doesn't answer "is the frozen embedding already useful," which is the cheaper and more informative first question. Revisit only if Phase 3 passes but leaves clear headroom for priority species. |
| Extend train.py's CNN/spectrogram stack to also predict species | Duplicates a large, already-in-flight effort (v1-v5 binary CNN iterations) with a different, much sparser label regime; conflates two research questions (does the existing spectrogram-CNN architecture generalize to species? vs. does Perch's embedding carry species signal?) into one experiment. |
| PyTorch nn.Linear + Lightning trainer instead of sklearn.LogisticRegression | Architecturally heavier for a convex problem that fits in memory; adds GPU-trainer/callback surface for no accuracy benefit at this scale. Acceptable to switch to later if class count or row count grows past what fits comfortably in sklearn on CPU. |
| perch-hoplite's full agile-modeling/vector-DB workflow | Solves a different problem (interactive active learning for novel concepts); adds a database layer and UI-driven labeling loop this task does not need for a first linear-probe readout. |
| ONNX backend as the primary path (lower dependency footprint than TF) | Attractive for minimizing new machinery, but the published ONNX weights come from a third-party HF repo (justinchuby/Perch-onnx), not an official Google Kaggle artifact - provenance/drift risk for what should be a scientifically trustworthy baseline. Keep as a documented fallback only, gated on a parity check against the official TF model. |
| Auto-merging near-duplicate species codes (e.g. RAMTUC/RHATUC) | Silent merging risks hiding a real annotation error under a modeling decision. Safer to keep both codes distinct and flag the duplication as a data-quality note. |

---

## 12. Compute / storage estimates (assumptions explicit; all UNVERIFIED numbers flagged)

| Item | Estimate | Basis / assumption |
|---|---|---|
| Embedding array size | 160,244 x 1536 x 4 bytes ~= 939 MiB (float32); ~= 470 MiB if stored float16 | Arithmetic, not measured. |
| Embedding extraction time | [UNVERIFIED] - order-of-magnitude guess only: if each 5s-window forward pass takes ~100-300 ms on CPU (EfficientNet-B3-sized model, unbatched, unverified), full extraction ~= 4.5-13 hours single-threaded CPU, likely 1-3 hours with batching/multiple workers; plausibly <30 min on a single GPU. Do not plan a compute budget from this row - it exists only to bound the Phase 0 timing spike's expectations. |
| Label pipeline (Phase 1) | Minutes; pure JSON parsing over ~15k + ~6.7k annotation rows and 160k windows, no ML. | Arithmetic on described data sizes. |
| Linear probe training (Phase 3) | Seconds to low minutes per fold: sklearn.LogisticRegression on ~100k-140k train rows x 1536 features x <= a few dozen classes (one-vs-rest) is a well-trodden, fast CPU workload. | Standard sklearn performance characteristics for this problem size; not benchmarked on this specific machine. |
| Disk footprint, total new artifacts | ~=1 GB embeddings + a few MB labels/classes/checkpoints/reports. Trivial relative to existing spectrograms/ (.npy per window at 224x469 float32 ~= 420 KB/window x 160k ~= 67 GB, for comparison). | Arithmetic; spectrogram figure derived from config.yaml's target_size: [224, 469]. |

Recommendation given the uncertainty: treat the Phase 0 timing spike (--limit 200) as a hard prerequisite before scheduling Phase 2 on the full dataset - do not book compute time off the guesses in this table.

---

## 13. Risks (consolidated)

1. Dependency conflict between perch-hoplite's TF/numpy pins and the existing PyTorch/Lightning env - mitigated by full env isolation (section 6.3); zero changes to requirements.txt.
2. Kaggle Hub network/credential access in this compute environment is unverified; if blocked, the entire plan needs a manual-download fallback (both kaggle_hub.py's load()/resolve() and the ONNX path accept a local path once downloaded once elsewhere) - check in Phase 0, don't discover it mid-Phase-2.
3. Label ambiguity (bird present, species unresolved) silently treated as negative - mitigated by the explicit unresolved_bird/usable_strict schema (section 4.2); this is the single most important correctness risk in the whole design and the reason Phase 1 exists as its own gated milestone before any embedding compute is spent.
4. Sparse/placeholder species codes inflating the apparent class count - mitigated by curated class list with documented exclusions (section 4.3).
5. ONNX fallback provenance (third-party conversion) - documented as fallback-only, gated on parity check (section 11).
6. Multi-species co-occurrence at 5s granularity blending signal - measured directly via the clean-window sensitivity subset (section 7 Phase 3), not assumed away.
7. Unverified per-class window counts mean the --min_count default in this document is a placeholder, not a final number - Phase 1's own output is what should set it (section 4.3).
8. The full Perch 2 paper (arXiv:2508.04665) was not read line-by-line for this proposal - only its abstract and secondary summaries were consulted, which is below this project's own research-rigor bar ("never conclude from abstracts alone," per the project's research rules). ACTION: before Phase 3's go/no-go threshold is finalized, someone should actually read the paper's methods/results (especially its BirdSet/soundscape linear-probe numbers) to calibrate what a "good" macro-AP looks like on passive-monitoring audio specifically, rather than relying on my secondhand summary.

---

## 14. Summary recommendation

Build exactly four small scripts (species_probe/{build_species_labels,extract_perch_embeddings,train_linear_probe,evaluate_linear_probe}.py), reuse the existing v4 windows and leave-one-project-out folds unchanged, isolate the one dependency (Perch 2 / TensorFlow) that doesn't belong in the main env, and gate every phase behind a concrete, inspectable acceptance check before spending compute or design effort on the next one. The single highest-leverage piece of new design in this proposal is not the embedding extraction - it's the six-state label schema in section 4.2, because getting "bird present, species unknown" wrong silently invalidates every downstream metric. Everything else is intentionally as small and as standard (linear probe on frozen embeddings) as the literature itself recommends for this exact question.

STATUS: DONE
