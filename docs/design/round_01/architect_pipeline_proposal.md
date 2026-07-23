# Linear Probing on Perch 2 Embeddings for Species Classification in PteroSet

**Author role**: Chief Architect (architect-pipeline), Design Round 1
**Status**: Proposed - ready for Inquisitor stress-test and Documenter formalization
**Scope**: First species-classification experiment on segmented_v4 (160,244 windows / 35,073 bird-positive windows), built as a parallel, non-invasive pipeline extension to the existing spectrogram/ResNet stack.

## Revision note

This document was researched and written in a single pass: repository inspection (CLAUDE.md, README.md, prepare_dataset.py, train.py, data_reader.py, config.yaml, windows_mapping_*_segmented_v4.json, data/folds_segmented_v4/*, annotations_species.json, species.csv) was performed first, followed by primary-source verification of the Perch 2 / perch-hoplite API, license, and constraints directly against Google's source and model-card repositories (not inferred from memory). Facts and recommendations are tagged throughout so a reviewer can immediately tell which statements are load-bearing (verified) versus judgment calls (proposed) that the Inquisitor should probe.

**Legend**
- [VERIFIED] - confirmed by reading this repo's code/data, or Perch 2 primary sources (repo code, model card). Source cited inline.
- [COMPUTED] - arithmetic derived directly from verified numbers (e.g., storage size).
- [RECOMMENDATION] - my architectural judgment; open to challenge.
- [ASSUMPTION - NEEDS EMPIRICAL CONFIRMATION] - plausible estimate that must be validated in Phase 0 before being trusted for planning.

No sub-agent delegation tool is exposed in this environment's toolset (no Task/spawn primitive), so the "delegate 3-5 subtasks" instruction was executed as simulated specialist passes performed directly: (1) repo-conventions audit, (2) Perch 2 API/license verification against primary sources, (3) label-provenance/data-quality audit of species.csv + annotations_species.json, (4) leakage/fold-compatibility check against prepare_dataset.py's split logic. Each is reflected in the sections below and should still be independently re-run by the Inquisitor and a data-validator pass before implementation begins.

---

## 1. Why

The repo currently answers one question well: "is there a bird in this 5s window?" (binary, outputs_v4/checkpoints_v4). The next scientific question - "which species?" - is multilabel (dawn choruses overlap), has a long-tail class distribution (139 species codes in species.csv, several with "-" as the species name, i.e. genus/family-level placeholders, not true species), and training a full CNN from spectrograms for this would be slow to iterate on and easy to leak.

Perch 2 is a frozen, general-purpose bioacoustics embedding model explicitly designed so that "training a simple linear classifier on top of the model's outputs should work well" [VERIFIED - HuggingFace model card, cgeorgiaw/Perch/README.md]. This makes frozen-embedding + linear probe the correct first experiment: it is cheap (no GPU training loop needed beyond embedding extraction, which itself is optional-GPU), fast to iterate (swap label thresholds/class filters without recomputing embeddings), and gives a defensible baseline before considering fine-tuning or a heavier head.

## 2. What (scope)

**In scope (Round 1):**
- Species-level multilabel target derived from annotations_species.json, re-using the exact window geometry and leave-one-project-out fold assignment already defined by segmented_v4 / folds_segmented_v4/.
- Frozen Perch 2 embedding extraction for the segmented_v4 window set, cached to disk with a versioned manifest.
- A linear (single nn.Linear) multilabel probe trained per fold, evaluated with the same leave-one-project-out protocol as the binary task.
- New, additive artifacts and scripts. **Zero modifications** to train.py, prepare_dataset.py, ResNetClassifier, windows_mapping_*_segmented_v4.json, or folds_segmented_v4/* - the shipped v4 binary experiment must remain byte-identical and re-runnable throughout.

**Out of scope (Round 1, flagged for future rounds - see section 12):**
- Fine-tuning the Perch 2 backbone itself.
- Hierarchical/taxonomic-aware modeling of the genus/family-level placeholder codes.
- Alternative encoders (BirdNET, SurfPerch) - the extractor is designed to make this cheap later, but only Perch 2 ships in Round 1.
- An agile-modeling / active-learning loop (perch-hoplite's agile/db sublibraries) - noted as a natural Round 2+ extension.

---

## 3. Current state (verified from this repo)

| Artifact | Fact |
|---|---|
| Window geometry | 5.0s windows, 4.0s overlap (1s hop), 48 kHz, stored as start/end in samples [VERIFIED - data/config.yaml, windows_mapping_4.0overlap_segmented_v4.json] |
| Current version | segmented_v4: 160,244 windows, 35,073 binary bird-positive [VERIFIED - CLAUDE.md] |
| Fold layout | data/folds_segmented_v4/fold_{i}_{PROJECT}_segmented/{train,val,test}_split.csv, 5 folds (leave-one-project-out over MAP1, PPA1-PPA4) [VERIFIED - directory listing] |
| Fold CSV schema | window_id,dataset,sample_rate,sound_id,start,end,label,spec_name,sound_filename,project [VERIFIED - folds_segmented_v4/fold_0_MAP1_segmented/train_split.csv] |
| Split logic | Train/val split via GroupShuffleSplit grouped by sound_id (prevents overlapping-window leakage within a recording); test set filtered to non-overlapping windows only (start % window_size_samples == 0) [VERIFIED - prepare_dataset.py run_splits()] |
| Label derivation history | v2 to v3 to v4 had a real bug: a cached windows file bypassed label re-derivation from current annotations, silently shipping stale PPA4 labels until detected [VERIFIED - CLAUDE.md changelog]. This is directly relevant precedent for the cache-key design in section 7. |
| Species annotation source | annotations_species.json: COCO-like schema, categories[].name = species code (e.g. ORTGUT), annotations[].{sound_id,category_id,category,t_min,t_max} [VERIFIED - data/annotations_species.json]. Produced by data/data_reader.py --annotation_level species, which only includes annotations with a species-level Determination [VERIFIED - data_reader.py add_categories/add_annotations]. |
| Species catalog quality | species.csv contains 139 rows. At least 6 are not true species: PSITTACIDAE, PSITTACIFORMES, RHACAR ("-"), TYRANN_SP1 ("Tyrannidae sp 1"), PSITTA ("Psittacidae sp."), PICIDA_1 ("Picidae") - coarse taxonomic placeholders [VERIFIED - data/species.csv]. Also RAMTUC and RHATUC both map to "Ramphastos tucanus" - a likely duplicate-code data-quality issue [VERIFIED - data/species.csv rows]. Neither is fixed here; both are handled by explicit config-driven exclusion (section 4) and flagged as a known caveat for the data owner, not silently patched. |
| Compute stack | PyTorch + PyTorch Lightning; **no TensorFlow/JAX** in requirements.txt [VERIFIED - requirements.txt]. Perch 2 requires TensorFlow (section 5) - this is a **new dependency axis**, not currently present. |

---

## 4. Label provenance, granularity, multilabel, unknown/background - design

**Granularity**: identical to the existing binary task - one training example = one window (5s, sound_id/start/end identity), because that is what the leakage-safe fold CSVs already key on. No new temporal granularity is introduced. [RECOMMENDATION]

**Derivation rule** (mirrors the existing run_segment_windows() label-derivation pattern, generalized to multilabel):
For window w with [ws_sec, we_sec) and species annotation a with [t_min, t_max) on the same sound_id, w is labeled positive for a.category iff t_min < we_sec and t_max > ws_sec (any temporal overlap) - by default, to stay consistent with how the existing binary label field is derived (sound_to_anns check in prepare_dataset.py). A configurable min_overlap_frac (fraction of window duration covered) is exposed but defaults to 0.0 (any overlap) so Round 1 species behavior is a strict generalization of the already-shipped binary logic, not a silent behavior change. [RECOMMENDATION - needs Inquisitor sign-off: "any overlap" can create boundary-noise multilabel positives from calls that barely clip the window edge]

**Multilabel handling**: target is a multi-hot vector over the species class list (dawn-chorus windows legitimately contain 2+ species). Trained with BCEWithLogitsLoss per class (sigmoid, not softmax) - standard for non-exclusive labels. [RECOMMENDATION]

**Class list construction** (species_labels.py, see section 8):
1. Start from all species codes appearing in annotations_species.json that overlap at least one segmented_v4 window.
2. Exclude coarse/non-species placeholder codes via an explicit excluded_codes allow/deny list in config (seeded with the 6 codes identified above) - never silently dropped, always logged to the label manifest's stats block.
3. Apply a min_class_support threshold (default 20 windows dataset-wide, [RECOMMENDATION]); species below threshold are collapsed into an explicit OTHER bucket rather than dropped, so no annotation is silently discarded - the manifest records exactly which codes were merged.
4. The resulting class_list.json (ordered species codes + OTHER) is the single source of truth for output dimensionality; it is versioned (section 7) independently from window geometry.

**Unknown / background treatment**:
- Windows with the existing binary label=0 (no bird) -> **all-zero** multilabel target. These remain useful negatives for the linear probe (BCE naturally handles all-zero targets) and are not excluded.
- Windows with binary label=1 (bird detected, per annotations_identification.json) but **no** overlapping species-level annotation (i.e., identified only to a coarser level, e.g. AVEVOC with no species Determination) are a genuine "unknown species" case. [RECOMMENDATION]: these windows are tagged has_species_label=false in the label manifest and **excluded from the species train/val/test tensors by default** (they carry no species-positive signal and would look identical to a true negative under an all-zero target, corrupting the loss). They are retained in the manifest (not deleted) so a future round can model them as an explicit UNK class or use them for calibration diagnostics.

**Leakage control**: species labels and embeddings are joined onto the **existing, unmodified** folds_segmented_v4/*/{train,val,test}_split.csv files by window_id. No new splitting logic is written. This is deliberate: the leave-one-project-out guarantee and the GroupShuffleSplit-by-sound_id guarantee that already protect the binary task are inherited for free, not re-implemented and re-risked. [VERIFIED design constraint - see section 3 split logic row]. A fail-loud assertion is added at join time: for every row in a fold's test_split.csv, the row's project must equal that fold's held-out project; any mismatch aborts the join (catches manifest/CSV misalignment early rather than training on silently-corrupted joins).

---

## 5. Perch 2 - verified facts

| Fact | Value | Source |
|---|---|---|
| Package | perch-hoplite (PyPI); install extras [tf] (CPU) or [tf-cuda] (GPU) | [VERIFIED] github.com/google-research/perch-hoplite README |
| Embedding API | from perch_hoplite.zoo import model_configs; model = model_configs.load_model_by_name('perch_v2') then model.embed(audio_array) (single) or model.batch_embed(audio_batch) (batched) returning InferenceOutputs.embeddings | [VERIFIED] perch_hoplite/zoo/model_configs.py, zoo_interface.py |
| Model variants | PERCH_V2 (auto GPU/CPU dispatch via has_gpu_tf()), PERCH_V2_GPU (explicit, Kaggle slug google/bird-vocalization-classifier/tensorFlow2/perch_v2), PERCH_V2_CPU (explicit, **different** Kaggle slug .../perch_v2_cpu), PERCH_V2_ONNX (ONNX runtime variant) | [VERIFIED] model_configs.py (ModelConfigName enum + get_preset_model_config), kaggle_hub.py (PERCH_V2_SLUG vs PERCH_V2_CPU_SLUG are distinct model artifacts, not the same weights re-served) |
| Sample rate | 32,000 Hz | [VERIFIED] model_configs.py: sample_rate = 32000 for perch_v2/perch_v2_gpu/perch_v2_cpu |
| Window / hop | window_size_s = 5.0, hop_size_s = 5.0 (internal framing config) | [VERIFIED] same file |
| Input contract | embed(audio_array): np.ndarray shape [Time], unit-scaled audio; internally peak-normalized to target_peak=0.25 by default (TaxonomyModelTF.target_peak) | [VERIFIED] zoo_interface.py (EmbeddingModel.embed docstring, normalize_audio), taxonomy_model_tf.py (target_peak: float | None = 0.25) |
| Output | InferenceOutputs.embeddings, shape [Frames, Channels, Features] (or with leading batch dim for batch_embed), Features = 1536 | [VERIFIED] zoo_interface.py docstring + model_configs.py embedding_dim = 1536 for perch_v2* |
| Exact [Frames, Channels] for a single 5s mono window fed via embed() | Expected Frames=1, Channels=1 (framing with hop_size_s == window_size_s over exactly one window's worth of samples; "Channels" = source-separated audio channels, mono maps to 1) | [ASSUMPTION - NEEDS EMPIRICAL CONFIRMATION]. The HuggingFace card additionally states unpooled embeddings have shape (5, 3, 1536) for the model's internal spatial grid before pooling - this is a different, model-internal notion than the [Frames, Channels, Features] returned by the public embed()/batch_embed() API. **Do not conflate the two** until Phase 0's smoke test empirically confirms the actual array shape returned by embed() on a real 5s clip. |
| Logit/species head | 91M-parameter classification head over ~15,000 classes (~10,000 birds) also available (LogitsOutputHead), but **not used** in Round 1 - we discard Perch's own species logits and train our own linear probe on the 1536-d embedding, because Perch's global species taxonomy will not align 1:1 with PteroSet's local species.csv codes | [VERIFIED existence] zoo_interface.py LogitsOutputHead; [RECOMMENDATION] not to use it in Round 1 |
| License (code) | Apache License 2.0 | [VERIFIED] perch-hoplite repo LICENSE file, per-file headers |
| License (model weights) | apache-2.0 (HF model card metadata tag) | [VERIFIED] huggingface.co/cgeorgiaw/Perch README front-matter |
| Weight distribution | Downloaded on first use via kagglehub.model_download(...) inside kaggle_hub.load() - requires a Kaggle account/API credentials (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json) and network egress to Kaggle | [VERIFIED] perch_hoplite/zoo/kaggle_hub.py |
| TensorFlow version | HuggingFace card states "requires TensorFlow 2.20.rc0 and a GPU; CPU variant will be added soon" | [VERIFIED - as stated on the card]. **This directly conflicts with the fact that model_configs.py already ships an explicit PERCH_V2_CPU preset with its own Kaggle slug.** [FLAGGED CONTRADICTION - must be resolved empirically in Phase 0, not assumed either way]. Treat the HF card's GPU-only claim as possibly stale relative to the code, but do not assume CPU inference works until the Phase 0 smoke test proves it end-to-end. |
| Not an officially supported product | Repo states "This is not an officially supported Google product," ineligible for Google's OSS VRP | [VERIFIED] perch-hoplite README footer - operational/support-expectations note, not a licensing blocker |
| Extensibility | Same perch_hoplite.zoo.model_configs.load_model_by_name(...) interface also exposes birdnet_V2.1..V2.4 (1024-420-d, 48kHz native, 3s window), surfperch (1280-d, 32kHz), perch_8/original Perch (1280-d, 32kHz), yamnet/vggish (16kHz) | [VERIFIED] model_configs.py full get_preset_model_config body |

**Architectural implication of the TF dependency**: this repo's requirements.txt has no TensorFlow. [RECOMMENDATION] Do **not** add TensorFlow to the main bioacoustics conda env (risk of CUDA/driver conflicts with the existing PyTorch/Lightning stack, and TF pinned to a 2.20.0rc0 release-candidate is fragile). Instead:
- Create an isolated environment (bioacoustics-embeddings) or an isolated pip install perch-hoplite[tf] inside a venv, used **only** by extract_embeddings.py.
- The two environments communicate only through files on disk (embedding .npy + manifest) - the training environment (bioacoustics, PyTorch) never imports perch_hoplite/tensorflow.
- This is also why the embedding extraction pipeline is designed as a wholly separate top-level module (embeddings/) rather than folded into prepare_dataset.py, which currently assumes a single environment for the whole pipeline.

---

## 6. Data-flow diagram

```
                                   +------------------------------+
                                   | annotations_species.json      |  (species-level COCO; from
                                   | (sound_id, t_min/t_max,       |   data_reader.py --annotation_level species)
                                   |  category=species code)       |
                                   +---------------+----------------+
                                                   |
  +---------------------------------+              |
  | windows_mapping_4.0overlap_     |              |
  | segmented_v4.json               |<-------------+  (existing, UNCHANGED,
  | (window_id, sound_id,           |                  shared with binary task)
  |  start, end, label)             |
  +----------------+-----------------+
                   |
                   v
     +-------------------------------+       +---------------------------------+
     | embeddings/species_labels.py   |       | data/folds_segmented_v4/         |
     |  build_species_label_windows   |       |  fold_{i}_{PROJECT}_segmented/   | (existing, UNCHANGED,
     |  -> multi-hot per window_id    |       |  {train,val,test}_split.csv      |  defines leakage-safe splits)
     +----------------+----------------+       +----------------+------------------+
                     |                                          |
                     v                                          |
     data/species_labels/segmented_v4/                          |
       species_labels_v1.json + class_list.json                 |
       + class_support.csv (stats)                              |
                     |                                          |
  +------------------+-------------------------------------------+-------------+
  |                  v                                          v             |
  |    +----------------------------+             +-----------------------+   |
  |    | audios_48khz/*.wav          |------------>| extract_embeddings.py |   |
  |    | (grouped & loaded once      | resample    |  (perch-hoplite env,  |   |
  |    |  per sound_id, sliced       | 48k->32k    |   TF, CPU or GPU)     |   |
  |    |  per window)                 | ONCE/file   |  perch_v2_cpu|_gpu    |   |
  |    +----------------------------+             +-----------+------------+   |
  |                                                            |               |
  |             EXTRACTION SUBSYSTEM (isolated env)            v               |
  |                                              data/embeddings/<model_tag>/  |
  |                                                {spec_name}.npy (1536-d)    |
  |                                                manifest.jsonl              |
  |                                                manifest_meta.json          |
  +------------------------------------------------------------+---------------+
                                                                 |
                                                                 v
                                       +-----------------------------------------+
                                       | linear_probe/dataset.py                  |
                                       |  EmbeddingDataset: joins fold CSV x       |
                                       |  species_labels_v1.json x manifest        |
                                       |  by window_id (fail-loud on                |
                                       |  project/split mismatch)                   |
                                       +--------------------+----------------------+
                                                             |
                                                             v
                                       +-----------------------------------------+
                                       | train_linear_probe.py                    |
                                       |  LinearProbeClassifier (PyTorch           |
                                       |  Lightning): nn.Linear(1536,C)            |
                                       |  BCEWithLogitsLoss, per fold               |
                                       +--------------------+----------------------+
                                                             |
                                                             v
                                       checkpoints_perch_v1/fold_{i}/
                                       outputs_perch_v1/{cv_results.csv, per-class metrics}
                                                             |
                                                             v
                                       eval_linear_probe.py -> aggregated
                                       macro/micro AUROC, mAP, per-class support
```

---

## 7. Embedding cache: key, versioning, storage format

**Storage decision** [RECOMMENDATION]: one .npy file per window, named identically to the existing spec_name convention already present in every fold CSV ({sound_base}_{start}_{end}.npy), stored under data/embeddings/<model_tag>/. This deliberately mirrors the existing spectrograms/ directory precedent (also one file per window) - same mental model for engineers, and it means EmbeddingDataset needs zero new join keys beyond the spec_name column that already exists in every fold CSV. A single small file is cheap here (6,144 bytes at float32 for a 1536-d vector, vs. ~420 KB per spectrogram), so the "many small files" concern that would matter for spectrograms is not a real cost for embeddings.

model_tag **must** encode the explicit model variant actually used (perch_v2_cpu or perch_v2_gpu) - **never** the ambiguous perch_v2 auto-alias - because PERCH_V2_GPU and PERCH_V2_CPU are documented as distinct exported model artifacts (different Kaggle slugs, section 5). Mixing embeddings from the two variants inside one model_tag directory would be a silent correctness bug.

**Manifest** (data/embeddings/<model_tag>/manifest.jsonl, one line per window):
```json
{"window_id": 547, "spec_name": "G6413_20240220_0_240000.npy", "sound_id": 2,
 "start": 0, "end": 240000, "source_sample_rate": 48000,
 "model_tag": "perch_v2_cpu", "model_kaggle_version": 1,
 "resample_method": "librosa.resample:soxr_hq", "target_peak": 0.25,
 "cache_key": "sha256:...", "embedding_dim": 1536, "embedding_dtype": "float32",
 "status": "ok", "error_message": null, "extracted_at": "2026-07-23T02:00:00Z"}
```

**Run-level metadata** (manifest_meta.json): full serialized perch_hoplite ConfigDict, perch-hoplite + tensorflow package versions, git commit SHA of birds_bioacoustics at extraction time, source windows_mapping_*.json filename + its SHA-256, host/device info (CPU/GPU, hostname), run start/end timestamps, counts by status.

**Cache key** (per window):
```
cache_key = sha256(
    f"{model_tag}|{model_kaggle_version}|{sample_rate_model}|{window_size_s}|"
    f"{target_peak}|{sound_id}|{start}|{end}|{source_sample_rate}|"
    f"{resample_method}|{perch_hoplite_version}"
)
```
This is the direct architectural answer to a bug this repo has already suffered once: the v2 to v3 PPA4 regression was caused by a cache that returned "already exists, skip" without checking whether the content it would produce had changed [VERIFIED - CLAUDE.md v3 to v4 changelog entry]. Here, before skipping an existing .npy because the file is present, the extractor recomputes the expected cache_key and compares it against the manifest's stored cache_key for that window_id; on mismatch, it re-extracts rather than silently trusting stale output.

**Resumability**:
- Deterministic processing order (sorted by window_id), --shard-index k --num-shards N for parallel/multi-worker runs.
- Atomic writes: write to {spec_name}.npy.tmp, os.rename() to final path (POSIX-atomic) - a crash mid-write can never leave a corrupt "final" file.
- Default behavior is resume-by-default (skip windows with a valid, matching cache_key in the manifest); --force re-extracts everything.
- --verify mode: recompute SHA-256 of a sample (or all) existing .npy files, cross-check against a file_sha256 field recorded in the manifest at write time, report corruption/staleness without re-extracting (an "fsck" pass) - needed because disks/filesystems can silently corrupt files independent of the cache-key logic above.

---

## 8. Modules, classes, CLI contracts

New top-level additions only; **no existing file is modified**.

```
embeddings/
  __init__.py
  perch_config.py      # dataclass PerchExtractionConfig
  perch_extractor.py   # class PerchEmbeddingExtractor
  manifest.py          # EmbeddingManifest: read/write/verify, compute_cache_key()
  species_labels.py     # build_species_label_windows(), class list construction

linear_probe/
  __init__.py
  dataset.py           # EmbeddingDataset(torch.utils.data.Dataset)
  model.py             # LinearProbeClassifier(pl.LightningModule)

extract_embeddings.py    # top-level CLI, mirrors prepare_dataset.py conventions
train_linear_probe.py    # top-level CLI, mirrors train.py conventions
eval_linear_probe.py     # top-level CLI, aggregates CV results (mirrors plot_cv_results.py)
```

### embeddings/perch_config.py
```python
@dataclass
class PerchExtractionConfig:
    windows_json: str                 # e.g. data/windows_mapping_4.0overlap_segmented_v4.json
    annotations_species_path: str     # data/annotations_species.json
    model_config_name: str            # "perch_v2_cpu" | "perch_v2_gpu"  (never "perch_v2")
    output_root: str = "./data/embeddings"
    batch_size: int = 64
    dtype: str = "float32"            # or "float16"
    resample_method: str = "librosa:soxr_hq"
    num_shards: int = 1
    shard_index: int = 0
    resume: bool = True
    force: bool = False
```

### embeddings/perch_extractor.py
```python
class PerchEmbeddingExtractor:
    def __init__(self, config: PerchExtractionConfig): ...
    def load(self) -> None:
        """Loads perch_hoplite model via model_configs.load_model_by_name(...)."""
    def extract_for_sound(self, sound_id: int, windows: list[dict]) -> dict[int, np.ndarray]:
        """Loads+resamples the source wav ONCE, slices+batches all windows
        belonging to this sound_id, calls model.batch_embed() per batch."""
    def run(self) -> ExtractionSummary:
        """Groups windows by sound_id, iterates, writes .npy + manifest rows,
        resumable/shardable per section 7."""
```

### CLI: extract_embeddings.py
```
python extract_embeddings.py --config data/config.yaml \
    --model perch_v2_cpu \
    --windows_json data/windows_mapping_4.0overlap_segmented_v4.json \
    --steps species_labels extract verify \
    [--shard-index K --num-shards N] [--force] [--dtype float16]
```
Steps: species_labels (writes data/species_labels/segmented_v4/species_labels_v1.json), extract (writes embeddings + manifest), verify (fsck pass).

### linear_probe/dataset.py
```python
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, fold_csv_path: str, embeddings_dir: str,
                 species_labels_path: str, class_list_path: str,
                 expected_project: str | None = None):
        """Joins fold CSV rows -> embedding .npy (by spec_name) -> multi-hot
        label (by window_id). Fail-loud if expected_project mismatch found
        in test split rows (leakage/misalignment guard, see section 4)."""
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (embedding[1536], multilabel_target[num_classes])."""
```

### linear_probe/model.py
```python
class LinearProbeClassifier(pl.LightningModule):
    def __init__(self, embedding_dim: int = 1536, num_classes: int = 100,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 pos_weight: torch.Tensor | None = None): ...
    # single nn.Linear(embedding_dim, num_classes); BCEWithLogitsLoss(pos_weight=...)
    # torchmetrics: MultilabelAUROC, MultilabelAveragePrecision (macro + per-class)
```

### CLI: train_linear_probe.py
```
python train_linear_probe.py --config data/config.yaml \
    --fold_dir data/folds_segmented_v4 \
    --embeddings_dir data/embeddings/perch_v2_cpu \
    --species_labels data/species_labels/segmented_v4/species_labels_v1.json \
    --class_list data/species_labels/segmented_v4/class_list.json \
    --cross_validation [--fold 0] \
    --ckpt_dir checkpoints_perch_v1
```
Deliberately mirrors train.py's existing --cross_validation/--fold_dir/--fold/--ckpt_dir flag names for cognitive consistency across the two pipelines.

### CLI: eval_linear_probe.py
```
python eval_linear_probe.py --fold_dir data/folds_segmented_v4 \
    --checkpoint_dir checkpoints_perch_v1 --output_dir outputs_perch_v1
```
Produces outputs_perch_v1/cv_results.csv (per-fold, per-class + macro/micro AUROC/AP) - a new artifact family, not appended to outputs_v4/.

### Config additions (data/config.yaml - additive keys only, existing keys untouched)
```yaml
species:
  annotations_file: "annotations_species.json"
  min_overlap_frac: 0.0
  excluded_codes: ["PSITTACIDAE", "PSITTACIFORMES", "RHACAR", "TYRANN_SP1", "PSITTA", "PICIDA_1"]
  min_class_support: 20
  rare_class_bucket: "OTHER"

embedding:
  model_config_name: "perch_v2_cpu"
  sample_rate: 32000
  window_size_s: 5.0
  target_peak: 0.25
  batch_size: 64
  dtype: "float32"
  output_root: "./data/embeddings"

linear_probe:
  lr: 0.001
  weight_decay: 0.0001
  epochs: 30
  pos_weight_strategy: "inverse_freq"
  threshold_calibration: "val_f1"
```

---

## 9. Manifest / artifact schemas (summary)

| Artifact | Path | Versioning axis |
|---|---|---|
| Species label targets | data/species_labels/segmented_v4/species_labels_v1.json | v1 bumps on label-derivation-rule changes (overlap threshold, excluded codes, support threshold); segmented_v4 inherited from window geometry version |
| Class list | data/species_labels/segmented_v4/class_list.json | paired with species_labels_v{N} |
| Class support stats | data/species_labels/segmented_v4/class_support.csv | paired with species_labels_v{N} |
| Embeddings | data/embeddings/<model_tag>/{spec_name}.npy | model_tag = explicit model variant string |
| Embedding manifest | data/embeddings/<model_tag>/manifest.jsonl + manifest_meta.json | one manifest per model_tag |
| Checkpoints | checkpoints_perch_v1/fold_{i}/ | new artifact family, parallel to checkpoints_v{N} |
| Outputs | outputs_perch_v1/ | new artifact family, parallel to outputs_v{N} |

All of the above are new, gitignored, data-derived artifacts - consistent with the existing "Data Artifacts (gitignored)" policy in CLAUDE.md. A new table analogous to the existing "Dataset Versions" table should be added to CLAUDE.md by the Documenter once this is implemented, documenting the species_labels_v{N} axis as orthogonal to segmented_v{N}.

---

## 10. Compute & storage estimates

| Quantity | Value | Basis |
|---|---|---|
| Embedding size/window | 1536 x 4 bytes = 6,144 B (float32); 3,072 B (float16) | [COMPUTED] from verified embedding_dim=1536 |
| Full-dataset embeddings (160,244 windows) | approx 984 MB (float32) / approx 492 MB (float16) | [COMPUTED] |
| Species-positive-only subset (35,073 windows) | approx 215 MB (float32) / approx 108 MB (float16) | [COMPUTED] |
| Recommendation | Extract for the **full** segmented_v4 window set, not just species-positive windows | [RECOMMENDATION]: embeddings are fold-agnostic and cheap; computing the full set once enables using label=0 windows as negatives and any future re-thresholding of min_overlap_frac/excluded_codes without re-extraction. |
| Extraction throughput | Not benchmarked yet | [ASSUMPTION - NEEDS EMPIRICAL CONFIRMATION]. Do not plan Phase 1 timelines against a specific windows/sec number until Phase 0's 500-window benchmark (section 11) reports real CPU and GPU throughput on this repo's actual audio files. |
| I/O efficiency | Resample each .wav file **once** (48 kHz to 32 kHz), not once per window | [RECOMMENDATION]. Windows overlap 5x (1s hop over a 5s window), so per-window resampling would redo ~80% of the same DSP work. PerchEmbeddingExtractor.extract_for_sound() loads+resamples a source file once, then slices+batches all of that file's windows through batch_embed(). |

---

## 11. Phased milestones and acceptance criteria

**Phase 0 - Spike & environment validation (target: 1-2 days)**
- Stand up an isolated bioacoustics-embeddings env; pip install 'perch-hoplite[tf]' (or [tf-cuda] if a GPU is available); pin exact resolved versions to requirements-embeddings.txt.
- Confirm Kaggle credentials are configured and model_configs.load_model_by_name('perch_v2_cpu') succeeds (downloads weights once, caches locally).
- Smoke test: feed a synthetic 5s sine wave through embed(); record and document the actual .embeddings shape/dtype (resolves the [Frames, Channels] ambiguity flagged in section 5).
- Explicitly attempt CPU-only inference and record whether it works, to resolve the HF-card-vs-code contradiction flagged in section 5.
- Benchmark throughput on 500 real PteroSet windows (CPU, and GPU if available).
- **Acceptance criteria**: documented embedding shape/dtype; documented CPU-vs-GPU numeric/behavioral parity or divergence; real throughput numbers; requirements-embeddings.txt committed; zero changes to any existing repo file.

**Phase 1 - Label derivation + full extraction (target: 2-4 days)**
- Implement embeddings/species_labels.py; run against annotations_species.json + segmented_v4 windows; produce species_labels_v1.json, class_list.json, class_support.csv.
- Implement extract_embeddings.py + PerchEmbeddingExtractor; run first on a pilot subset (e.g., all MAP1 windows), then the full 160,244-window set.
- Prove resumability: kill the extraction job mid-run, restart, confirm the final manifest is identical to an uninterrupted run (idempotency test).
- **Acceptance criteria**: 100% of windows have a manifest entry with status="ok" or a documented error; --verify passes with zero checksum mismatches; re-run after interruption is idempotent; class support stats reviewed and excluded_codes/min_class_support decisions documented.

**Phase 2 - Linear probe training + cross-validation (target: 2-3 days)**
- Implement linear_probe/ package + train_linear_probe.py; train on all 5 folds.
- Implement a trivial majority-class / marginal-frequency baseline per fold for sanity comparison.
- **Acceptance criteria**: training converges on all folds without NaNs; per-fold + aggregate macro/micro AUROC and mAP reported in outputs_perch_v1/cv_results.csv; linear probe beats the trivial baseline on at least macro-AUROC; per-class metrics reported alongside per-class support (avoid over-reading metrics for classes with fewer than 5 test windows).

**Phase 3 - Documentation (target: approx 1 day)**
- Documenter formalizes a docs/implementation/species-linear-probe-v1/ results write-up (mirroring the existing docs/implementation/v4-ppa4-fix/ pattern), updates CLAUDE.md with the new artifact tables, and records the known caveats (duplicate species codes, excluded coarse taxa, rare-class bucket threshold).

**Phase 4 - Explicitly deferred** (not part of Round 1 acceptance): fine-tuning, alternative encoders, hierarchical taxonomy, agile/active-learning loop (section 12).

---

## 12. Validation / tests

- **Unit**: compute_cache_key() determinism (identical inputs give identical key; any config field change gives a different key); manifest read/write round-trip; build_species_label_windows() edge cases (annotation fully inside window; annotation straddling a window boundary; annotation spanning multiple windows; zero annotations gives an all-zero vector; min_overlap_frac boundary conditions).
- **Integration**: end-to-end run on a tiny fixture (2-3 short synthetic wavs, a handful of windows) through extraction (using perch_v2_cpu, or a mocked EmbeddingModel for CI environments without Kaggle network access) plus 1-epoch linear-probe training; assert expected artifact files exist and are non-empty.
- **Regression / non-invasiveness**: checksum windows_mapping_4.0overlap_segmented_v4.json, folds_segmented_v4/*, and spectrograms/* before and after running any new script in this design; any diff is a hard failure. This directly protects the already-shipped outputs_v4/checkpoints_v4 experiment.
- **Data validation** (data-validator pass): verify class_support.csv sums reconcile against species_labels_v1.json; verify, for every fold's test_split.csv, that every joined window's project equals the fold's held-out project (leakage/misalignment guard from section 4) - implemented as a fail-loud assertion inside EmbeddingDataset.__init__, not just a test.

---

## 13. Operational failure modes

1. **Kaggle auth/network failure** during model download - mitigate by documenting KAGGLE_USERNAME/KAGGLE_KEY (or kaggle.json) setup, and by treating the local kagglehub cache as a durable artifact (back it up / vendor it) so re-runs on the same or another machine don't require network access.
2. **TF/CUDA driver conflicts** with the existing PyTorch stack - mitigated by environment isolation (bioacoustics-embeddings separate from bioacoustics); the two pipelines only share files on disk.
3. **Silent GPU/CPU variant mixing** - mitigated by always pinning explicit model_config_name (perch_v2_cpu/perch_v2_gpu) and encoding it in model_tag; never use the perch_v2 auto-alias for cached artifacts.
4. **Partial/corrupted .npy from crashed jobs** - mitigated by atomic tmp-then-rename writes and the --verify checksum pass.
5. **Silent stale cache reuse after a config change** (the exact bug class that caused the documented v2 to v3 PPA4 regression in this repo) - mitigated by the cache_key check before any skip-if-exists decision (section 7).
6. **Resampler non-determinism** across library versions/backends - mitigated by pinning one resampling method (resample_method) explicitly in config and recording it (and library version) in the manifest.
7. **Long-tail class imbalance** producing unstable/degenerate per-class metrics - mitigated by min_class_support + OTHER bucket + always reporting per-class support alongside metrics.
8. **Disk pressure** from a new ~1 GB artifact family on top of existing large audios_*/spectrograms/ directories - mitigated by the float16 dtype option and by the --verify step reporting total bytes used.
9. **Species-catalog data quality** (duplicate codes RAMTUC/RHATUC; coarse placeholders) silently distorting the class list - mitigated by explicit excluded_codes config and by not auto-fixing duplicates without data-owner review; documented as a known caveat, escalated to the Documenter/data-validator rather than patched silently in this design.
10. **TF-pre-release pin fragility** (~=2.20.0rc0 per the HF model card) - mitigated by pinning the exact resolved version in requirements-embeddings.txt and re-validating when TF 2.20 stable ships.

---

## 14. Future migration path

- **To fine-tuning**: Perch 2's backbone training code lives in the original chirp/JAX training stack, not exposed via perch_hoplite.zoo's inference-only wrapper - a genuine fine-tune is a separate, larger design effort (different framework, different compute profile) and is explicitly deferred. The embedding cache built in Round 1 remains directly reusable as input features for a heavier frozen-embedding head (small MLP, or a lightweight temporal model over consecutive windows' embeddings for recording-level context) without any re-extraction - this is the natural Round 2 step before considering backbone fine-tuning.
- **To alternative encoders**: embeddings/perch_extractor.py is designed as a thin adapter over perch_hoplite.zoo.model_configs's already-generic EmbeddingModel interface ([VERIFIED] the same load_model_by_name/embed/batch_embed contract is shared by birdnet_V2.1..V2.4, surfperch, perch_8, yamnet, vggish). Swapping model_config_name in PerchExtractionConfig is the only code change needed to benchmark another frozen encoder under the identical label/fold/linear-probe harness - deliberately built this way now so it is not over-engineered (no bespoke plugin system), while remaining a one-parameter change later.
- **Interesting Round 2 candidate**: BirdNET V2.4 natively expects 48 kHz audio ([VERIFIED] model_configs.py: sample_rate=48000 for all birdnet_* presets) - the same sample rate as PteroSet's native audio, eliminating the resample step entirely. Worth a direct comparison against Perch 2 once Round 1's harness exists.
- **To agile/active-learning workflows**: perch-hoplite's db/agile sublibraries (vector search + human-in-the-loop classifier refinement over pre-computed embeddings) are a natural extension once the embedding cache exists, but are out of scope for Round 1's supervised linear-probe experiment.

---

## Appendix A - Verified-fact source log

- Repository conventions: CLAUDE.md, README.md, prepare_dataset.py, train.py, data/config.yaml, data/data_reader.py, data/windows_mapping_4.0overlap_segmented_v4.json, data/folds_segmented_v4/fold_0_MAP1_segmented/train_split.csv, data/species.csv, data/annotations_species.json, outputs_v4/, checkpoints_v4/ - all read directly in this session.
- Perch 2 / perch-hoplite: huggingface.co/cgeorgiaw/Perch/README.md; github.com/google-research/perch-hoplite README, LICENSE, perch_hoplite/zoo/model_configs.py, perch_hoplite/zoo/zoo_interface.py, perch_hoplite/zoo/taxonomy_model_tf.py, perch_hoplite/zoo/kaggle_hub.py - all fetched and read directly in this session (raw source, not summarized secondary sources).
- No claim in section 5 was taken from unverified web-search summarization alone; the search result was used only to locate primary sources, which were then fetched and read directly.
