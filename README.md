# A standardized annotated dataset for tropical bird monitoring through passive acoustic monitoring and machine learning

This work is under submission to the Bioacoustics Data collection of *Scientific Data* journal.

Pipeline for bird vocalization detection and classification from audio recordings using spectrogram-based deep learning models. Built on top of the [PytorchWildlife](https://github.com/microsoft/CameraTraps) library, it covers data downloading, annotation creation, dataset preparation, model training, cross-validation, and evaluation.

**Dataset**: [PteroSet](https://zenodo.org/records/19137071) (Zenodo)

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

- `audios.zip` — Audio recordings
- `labels.zip` — RAVEN Pro annotation files
- `annotations_identification.json` — Identification-level annotations (COCO format)
- `annotations_species.json` — Species-level annotations (COCO format)
- `metadata.csv` — Audio file metadata
- `species.csv` — Species reference data
- `checkpoints.zip` — Pretrained model checkpoints

Optional arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir DIR` | Directory to save and extract data | `data/` |
| `--keep-zip` | Keep ZIP files after extraction | off |
| `--workers N` | Parallel connections per file | 8 |

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
- `segment_windows` — Filter windows to respect segment boundaries (removes windows that cross 10 s boundaries)
- `spectrograms` — Compute mel spectrograms on GPU
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
    --cross_validation --fold_dir data/folds_segmented_v2

# Train a specific fold
python train.py --config data/config.yaml \
    --cross_validation --fold_dir data/folds_segmented_v2 --fold 0

# Finetune from a pretrained checkpoint
python train.py --config data/config.yaml \
    --ckpt_path pretrained.ckpt --finetune true
```

### 7. Evaluate Model

```bash
# Evaluate a checkpoint on a test set (no training)
python train.py --config data/config.yaml \
    --ckpt_path checkpoint.ckpt \
    --test_csv test_split.csv

# Aggregate cross-validation results and plot precision-recall curves
python plot_cv_results.py \
    --fold_dir data/folds_segmented_v2 \
    --config data/config.yaml \
    --checkpoint_dir checkpoints
```

## Project Structure

```
birds_bioacoustics/
├── data/
│   ├── config.yaml              # Central configuration
│   ├── download_data.py         # Download dataset from Zenodo
│   ├── data_reader.py           # Parse RAVEN labels into COCO annotations
│   └── data_stats.py            # Dataset statistics and visualizations
├── train.py                     # Model training and evaluation
├── prepare_dataset.py           # Dataset preparation pipeline
├── plot_cv_results.py           # Cross-validation results and PR curves
└── requirements.txt
```

## Pipeline Overview

```
Audio files (48 kHz) + RAVEN label files
  → Annotations (COCO JSON)
  → Sliding windows (5s, 4s overlap)
  → Segmented windows (filter boundary-crossing windows)
  → Mel spectrograms (.npy)
  → Leave-one-project-out CV splits (non-overlapping test)
  → ResNet classifier (binary or multiclass)
  → Evaluation with aggregated metrics
```

For inference on new audio, see the [PytorchWildlife](https://github.com/microsoft/CameraTraps) library.
