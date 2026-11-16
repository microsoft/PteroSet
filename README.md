# A strongly annotated passive acoustic dataset for tropical bird monitoring 

This repository accompanies the paper [*A strongly annotated passive acoustic dataset for tropical bird monitoring*](https://doi.org/10.1038/s41597-026-08574-5), published in the Bioacoustics Data collection of *Scientific Data* (2026). If you use this code or dataset, please cite the paper (see [Citation](#citation)).

This is a pipeline for bird vocalization detection and classification from audio recordings using spectrogram-based deep learning models. Built on top of the [PytorchWildlife](https://github.com/microsoft/CameraTraps) library, it covers data downloading, annotation creation, dataset preparation, model training, cross-validation, and evaluation.

**Dataset**: [PteroSet](https://zenodo.org/records/21829388) (Zenodo)

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Download Data

From the `data/` directory:

```bash
cd data
python download_data.py
```

This downloads and extracts the PteroSet dataset from Zenodo into the `data/` directory. The Zenodo record contains the following files:

- `MAP1.zip` — Audio recordings from project MAP1
- `PPA1.zip` — Audio recordings from project PPA1
- `PPA2.zip` — Audio recordings from project PPA2
- `PPA3.zip` — Audio recordings from project PPA3
- `PPA4.zip` — Audio recordings from project PPA4
- `labels.zip` — RAVEN Pro annotation files
- `annotations_identification.json` — Identification-level annotations (COCO format)
- `annotations_species.json` — Species-level annotations (COCO format)
- `metadata_v2.csv` — Audio file metadata
- `species.csv` — Species reference data
- `checkpoints.zip` — Pretrained model checkpoints
- `segment_manifest.csv` — File containing the 10 sec segments in the timelapse audios

Optional arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir DIR` | Directory to save and extract data | `data/` |
| `--keep-zip` | Keep ZIP files after extraction | off |
| `--workers N` | Parallel connections per file | 8 |
| `--record_id` | Zenodo record ID | Last version |
| `--audio {192khz,48khz}` | Audio to download (see below) | `192khz` |

**Reproducing the paper results.** The results reported in the paper were obtained from the 48 kHz audio of the first dataset version ([record 18563039](https://zenodo.org/records/18563039)). The default download gets the original 192 kHz recordings, which produce slightly different spectrograms and metrics once resampled. To reproduce the paper results, download with:

```bash
python download_data.py --audio 48khz
```

This downloads the 48 kHz audio instead of the per-project ZIPs and places each WAV in its `data/<project>/` folder, so the remaining steps are the same. All other files still come from the latest version.

### 3. Create Annotations

Precomputed annotation files (`annotations_species.json` and `annotations_identification.json`) are available directly from the Zenodo record. This step is only needed if regenerating from raw labels.

From the `data/` directory:

```bash
# Species-level: only annotations with species determination
# Categories = species codes (e.g., "CYAVIO"), supercategory = identification
python data_reader.py --annotation_level species

# Identification-level: all annotations included
# Categories = identification (e.g., "AVEVOC"), supercategory = type
python data_reader.py --annotation_level identification
```

Outputs `annotations_species.json` or `annotations_identification.json` in COCO format.

### 4. Configuration

`data/config.yaml` contains all experiment settings: paths, audio parameters, spectrogram configuration, training hyperparameters, and data split ratios. Key defaults:

- Audio: 48 kHz sample rate, 5-second windows with 4-second overlap
- Spectrogram: 224 mel bins, 2048 FFT, 512 hop length
- Training: ResNet18 backbone, binary classification (Birds / No Birds), batch size 32
- Splits: Leave-one-project-out with 5 projects (MAP1, PPA1-PPA4)

### 5. Prepare Dataset

```bash
# Full pipeline (stats, windows, segment_windows, spectrograms, splits)
python prepare_dataset.py --config data/config.yaml

# Or run specific steps
python prepare_dataset.py --config data/config.yaml --steps windows segment_windows spectrograms
```

Available steps:
- `stats` — Display dataset statistics
- `windows` — Build sliding windows from annotations
- `segment_windows` — Filter windows to stay within 10-second snapshots, using
  the project geometry: PPA1 snapshot starts are 9 seconds apart (1-second
  overlap), while MAP1/PPA2/PPA3/PPA4 starts are 10 seconds apart
- `spectrograms` — Compute mel spectrograms on GPU. Audio is resampled to the
  48 kHz of the config when needed. Existing spectrograms are not recomputed, so
  the step writes a `provenance.json` (source audio sample rate and spectrogram
  parameters) into the spectrograms folder and stops if the folder was computed
  from different audio or parameters
- `splits` — Create leave-one-project-out folds (segmented windows, non-overlapping test sets)

### 6. Train Model

```bash
# Single train/val/test split
python train.py --config data/config.yaml \
    --train_csv train_split.csv \
    --val_csv val_split.csv \
    --test_csv test_split.csv

# Cross-validation across all 5 folds
python train.py --config data/config.yaml \
    --cross_validation --fold_dir data/folds_segmented

# Train a specific fold
python train.py --config data/config.yaml \
    --cross_validation --fold_dir data/folds_segmented --fold 0

# Finetune from a pretrained checkpoint
python train.py --config data/config.yaml \
    --ckpt_path pretrained.ckpt --finetune true
```

Checkpoints from training are saved to `checkpoints/` (`fold_<n>/` subfolders in cross-validation; change it with `--ckpt_dir`). The pretrained checkpoints downloaded from Zenodo are kept apart in `data/checkpoints/`.

### 7. Evaluate Model

```bash
# Evaluate a checkpoint on a test set (no training)
python train.py --config data/config.yaml \
    --ckpt_path checkpoint.ckpt \
    --test_csv test_split.csv

# Evaluate every fold with its own checkpoint (data/checkpoints/fold_<n>/).
# In cross-validation --ckpt_path must be a folder, not a single .ckpt file
python train.py --config data/config.yaml \
    --cross_validation --fold_dir data/folds_segmented \
    --ckpt_path data/checkpoints

# Aggregate cross-validation results and plot precision-recall curves
python plot_cv_results.py \
    --fold_dir data/folds_segmented \
    --config data/config.yaml \
    --checkpoint_dir data/checkpoints
```

## Segment manifest (optional)

Not needed for training or evaluation. The segment manifest makes the same
snapshot geometry used by the `segment_windows` step explicit in a portable
CSV and supports extracting a segment by its stable identifier:

```bash
# Generate the segment manifest
python data/segment_manifest.py generate \
    --annotations data/annotations_identification.json \
    --metadata data/metadata_v2.csv \
    --output data/segment_manifest.csv

# Extract one manifest row to a WAV
python data/segment_manifest.py extract \
    --manifest data/segment_manifest.csv \
    --segment-id SEGMENT_ID \
    --audio-root data/MAP1 \
    --output segment.wav
```

`--audio-root` is the project folder that contains the segment's WAV. The
manifest stores sample offsets for the 192 kHz audio.

Metadata is optional when generating a manifest. Run
`python data/segment_manifest.py --help` for all options. See the
[segment manifest implementation note](data/segment_manifest_note.md)
for the schema, time semantics, and validation expectations.

## Project Structure

```
PteroSet/
├── data/
│   ├── config.yaml              # Central configuration
│   ├── download_data.py         # Download dataset from Zenodo
│   ├── data_reader.py           # Parse RAVEN labels into COCO annotations
│   ├── data_stats.py            # Dataset statistics and visualizations
│   ├── segment_manifest.py      # Generate the segment manifest and extract segments
│   └── segment_utils.py         # Shared time-lapse segment geometry 
│   └── segment_manifest_note.md # Segment manifest schema and time semantics  
├── train.py                     # Model training and evaluation
├── prepare_dataset.py           # Dataset preparation pipeline
├── plot_cv_results.py           # Cross-validation results and PR curves
├── requirements.txt             # Python dependencies
├── LICENSE.txt                  # MIT License
└── SECURITY.md                  # Microsoft security policy
```

## Citation

```bibtex
@article{ruiz2026pteroset,
  title     = {A strongly annotated passive acoustic dataset for tropical bird monitoring},
  author    = {Ruiz, Daniela and Ulloa, Juan Sebasti{\'a}n and Miao, Zhongqi and Betancourt, Nicol{\'a}s and Toro-G{\'o}mez, Maria Paula and Hern{\'a}ndez, Andr{\'e}s and Demuro, Bruno and Barona-Cort{\'e}s, Eliana and Mendoza-Henao, Angela M. and Sierra-Ricaurte, Andr{\'e}s and P{\'e}rez-Pe{\~n}a, Sebastian and Dodhia, Rahul and Arbel{\'a}ez, Pablo and Lavista, Juan},
  journal   = {Scientific Data},
  publisher = {Springer Science and Business Media LLC},
  year      = {2026},
  doi       = {10.1038/s41597-026-08574-5},
  url       = {https://doi.org/10.1038/s41597-026-08574-5}
}
```
