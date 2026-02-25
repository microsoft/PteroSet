# Birds Bioacoustics

Pipeline for bird vocalization detection and classification from audio recordings using spectrogram-based deep learning models. Built on top of the PytorchWildlife library, it covers data downloading, annotation creation, dataset preparation, training, and inference.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Download Data

From the `data/` directory, run:

```bash
cd data
python download_data.py
```

This downloads and extracts the Humboldt - AI4G Bioacoustics Dataset from Zenodo into the `data/` directory, creating the `audios_48khz/`, `labels_48khz/`, and `species.csv` files.

Optional arguments:

```
--output-dir DIR    Directory to save and extract data (default: data/)
--keep-zip          Keep ZIP files after extraction
--workers N         Parallel connections per file (default: 8)
```

### 3. Create Annotations

From the `data/` directory, run `data_reader.py` with the desired annotation level:

```bash
# Species-level: only annotations with species determination
# Categories = species codes (e.g., "CYAVIO"), supercategory = identification (e.g., "AVEVOC")
python data_reader.py --annotation_level species

# Identification-level: all annotations included
# Categories = identification (e.g., "AVEVOC"), supercategory = type (e.g., "BIO")
python data_reader.py --annotation_level identification
```

This produces `annotations_species.json` or `annotations_identification.json` in the `data/` directory.

### 4. Configuration

The file `config.yaml` contains the configuration used in our experiments, including paths, audio parameters, spectrogram settings, training hyperparameters, and data split ratios.

### 5. Prepare Dataset

```bash
# Full pipeline (stats, windows, spectrograms, splits)
python prepare_dataset.py --config config.yaml

# Or run specific steps
python prepare_dataset.py --config config.yaml --steps windows spectrograms
```

### 6. Train Model

```bash
python train.py --config config.yaml \
    --train_csv train_split.csv \
    --val_csv val_split.csv \
    --test_csv test_split.csv \
```

### 7. Run Inference

```bash
python inference.py --config config.yaml \
    --checkpoint model.ckpt \
    --audios_source data/audios_48khz/ \
    --dataset my_inference
```
