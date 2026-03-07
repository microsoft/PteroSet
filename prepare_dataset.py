"""
Generic dataset preparation script for PW_Bioacoustics.

Usage:
    # Full pipeline
    python prepare_dataset.py --config data/config.yaml

    # Run specific steps only
    python prepare_dataset.py --config data/config.yaml --steps stats windows

    # Available steps: stats, windows, spectrograms, splits
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Import from PytorchWildlife core library
from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config, DomainConfig
from PytorchWildlife.data.bioacoustics.bioacoustics_windows import build_windows


def count_window_labels(windows: List[dict]) -> dict:
    """Count label distribution in windows."""
    counts = {}
    for w in windows:
        label = w.get('label', 0)
        counts[label] = counts.get(label, 0) + 1
    return counts


def run_stats(config: DomainConfig) -> None:
    """Load and display dataset statistics."""
    print(f"\n{'='*60}")
    print(f"Step: Dataset Statistics")
    print(f"{'='*60}")

    annotation_path = config.paths.annotations_path
    print(f"Loading annotations from: {annotation_path}")

    if not os.path.exists(annotation_path):
        print(f"Warning: Annotations file not found: {annotation_path}")
        return

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Dataset info
    if 'info' in data:
        print(f"\nDataset Info:")
        for key, value in data['info'].items():
            print(f"  - {key}: {value}")

    # Sound statistics
    sounds = data.get('sounds', [])
    print(f"\nSounds: {len(sounds)}")
    if sounds:
        durations = [s.get('duration', 0) for s in sounds]
        print(f"  - Total duration: {sum(durations):.1f}s ({sum(durations)/3600:.2f}h)")
        print(f"  - Mean duration: {sum(durations)/len(durations):.1f}s")
        print(f"  - Min duration: {min(durations):.1f}s")
        print(f"  - Max duration: {max(durations):.1f}s")

    # Annotation statistics
    annotations = data.get('annotations', [])
    print(f"\nAnnotations: {len(annotations)}")
    if annotations:
        categories = {}
        for ann in annotations:
            cat_id = ann.get('category_id', 0)
            categories[cat_id] = categories.get(cat_id, 0) + 1
        print(f"  - By category: {categories}")

    # Category names
    if 'categories' in data:
        print(f"\nCategories:")
        for cat in data['categories']:
            print(f"  - {cat.get('id', '?')}: {cat.get('name', 'Unknown')}")


def run_windows(config: DomainConfig) -> List[dict]:
    """Build windows from annotations."""
    print(f"\n{'='*60}")
    print(f"Step: Build Windows")
    print(f"{'='*60}")

    annotation_path = config.paths.annotations_path
    output_dir = config.paths.data_root
    os.makedirs(output_dir, exist_ok=True)

    windows_output_path = os.path.join(
        output_dir,
        f"windows_mapping_{config.audio.overlap_sec}overlap.json"
    )

    if os.path.exists(windows_output_path):
        print(f"Loading existing windows from: {windows_output_path}")
        with open(windows_output_path, 'r') as f:
            windows = json.load(f)
        print(f"Loaded {len(windows)} windows")
    else:
        strategy = config.audio.window_strategy
        print(f"Building windows with:")
        print(f"  - strategy: {strategy}")
        print(f"  - window_size: {config.audio.window_size_sec}s")
        print(f"  - overlap: {config.audio.overlap_sec}s")
        print(f"  - sample_rate: {config.audio.sample_rate}")
        print(f"  - datasets: {config.datasets}")
        if strategy == "balanced":
            print(f"  - negative_proportion: {config.audio.negative_proportion}")

        windows = build_windows(
            annotation_file=annotation_path,
            window_size_sec=config.audio.window_size_sec,
            overlap_sec=config.audio.overlap_sec,
            sample_rate=config.audio.sample_rate,
            datasets_names=config.datasets,
            strategy=strategy,
            negative_proportion=config.audio.negative_proportion,
        )

        with open(windows_output_path, 'w') as f:
            json.dump(windows, f, indent=2)
        print(f"Saved {len(windows)} windows to: {windows_output_path}")

    # Show label distribution
    counts = count_window_labels(windows)
    print(f"\nLabel distribution: {counts}")

    return windows


def run_spectrograms(config: DomainConfig, windows: List[dict]) -> None:
    """Compute mel spectrograms using GPU."""
    # Import here to avoid loading torch unnecessarily
    from PytorchWildlife.data.bioacoustics.bioacoustics_spectrograms import compute_mel_spectrograms_gpu

    print(f"\n{'='*60}")
    print(f"Step: Compute Mel Spectrograms (GPU)")
    print(f"{'='*60}")

    spectrograms_dir = config.paths.spectrograms_dir
    os.makedirs(spectrograms_dir, exist_ok=True)

    print(f"Output directory: {spectrograms_dir}")
    print(f"Spectrogram parameters:")
    print(f"  - n_fft: {config.spectrogram.n_fft}")
    print(f"  - hop_length: {config.spectrogram.hop_length}")
    print(f"  - n_mels: {config.spectrogram.n_mels}")
    print(f"  - top_db: {config.spectrogram.top_db}")
    print(f"  - fill_highfreq: {config.spectrogram.fill_highfreq}")

    # Load annotations to get audio file paths
    with open(config.paths.annotations_path, 'r') as f:
        annotations = json.load(f)

    sounds = {s['id']: s for s in annotations['sounds']}

    # Convert windows format to include sound_path
    inference_windows = []
    for win in windows:
        sound = sounds.get(win['sound_id'])
        if sound:
            inference_windows.append({
                'window_id': win['window_id'],
                'sound_path': sound['file_name_path'],
                'start': win['start'],
                'end': win['end'],
            })

    compute_mel_spectrograms_gpu(
        windows=inference_windows,
        sample_rate=config.audio.sample_rate,
        n_fft=config.spectrogram.n_fft,
        hop_length=config.spectrogram.hop_length,
        n_mels=config.spectrogram.n_mels,
        top_db=config.spectrogram.top_db,
        spectrograms_path=spectrograms_dir,
        save_npy=True,
        fill_highfreq=config.spectrogram.fill_highfreq,
        noise_db_std=config.spectrogram.noise_db_std,
        storage_dtype=config.spectrogram.storage_dtype,
    )

    print("Spectrogram computation complete!")


def run_splits(config: DomainConfig, windows: List[dict]) -> None:
    """Create leave-one-project-out cross-validation splits.

    For each of the 5 projects, hold it out as the test set and split the
    remaining 4 projects into train/val (grouped by sound_id).  Each fold
    is saved into its own subdirectory under the output root.
    """
    from sklearn.model_selection import GroupShuffleSplit

    print(f"\n{'='*60}")
    print(f"Step: Create Data Splits (Leave-One-Project-Out)")
    print(f"{'='*60}")

    spectrograms_dir = config.paths.spectrograms_dir
    output_dir = config.paths.data_root
    os.makedirs(output_dir, exist_ok=True)

    print(f"Spectrograms directory: {spectrograms_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split parameters:")
    print(f"  - val_size: {config.splits.val_size}")
    print(f"  - random_state: {config.splits.random_state}")

    # Build dataframe from windows
    df = pd.DataFrame(windows)

    # Load annotations to map sound_id -> file path & basename
    with open(config.paths.annotations_path, 'r') as f:
        annotations = json.load(f)
    sounds = {s['id']: s for s in annotations['sounds']}

    # Add spectrogram name column
    from inference import spectrogram_filename
    df['spec_name'] = df.apply(
        lambda row: spectrogram_filename(
            sounds[row['sound_id']]['file_name_path'],
            row['start'], row['end']),
        axis=1
    )

    # Add sound_filename (basename of the audio file)
    df['sound_filename'] = df['sound_id'].map(
        lambda sid: os.path.basename(sounds[sid]['file_name_path'])
    )

    # Load metadata and join to add project column
    metadata_path = os.path.join(config.paths.data_root, 'metadata.csv')
    print(f"Loading metadata from: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    # Build audio_file -> project_name mapping
    audio_to_project = dict(zip(
        metadata_df['audio_file'], metadata_df['project_name']
    ))
    df['project'] = df['sound_filename'].map(audio_to_project)

    unmapped = df['project'].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} windows could not be mapped to a project")
        df = df.dropna(subset=['project'])

    # Filter to rows where the spectrogram .npy exists on disk
    df['spec_exists'] = df['spec_name'].apply(
        lambda x: os.path.exists(os.path.join(spectrograms_dir, x))
    )
    print(f"\nTotal windows: {len(df)}")
    print(f"Existing spectrograms: {df['spec_exists'].sum()}")
    df = df[df['spec_exists']].drop(columns=['spec_exists'])

    # Iterate over unique projects (sorted for reproducibility)
    projects = sorted(df['project'].unique())
    print(f"\nProjects ({len(projects)}): {projects}")

    for fold_idx, held_out_project in enumerate(projects):
        fold_name = f"fold_{fold_idx}_{held_out_project}"
        fold_dir = os.path.join(output_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n{'-'*50}")
        print(f"Fold {fold_idx}: held-out project = {held_out_project}")
        print(f"{'-'*50}")

        # Test: all rows from the held-out project
        test_df = df[df['project'] == held_out_project].copy()

        # Remaining: split into train / val grouped by sound_id
        remaining_df = df[df['project'] != held_out_project].copy()

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=config.splits.val_size,
            random_state=config.splits.random_state,
        )
        train_idx, val_idx = next(
            gss.split(remaining_df, remaining_df['label'],
                       groups=remaining_df['sound_id'])
        )
        train_df = remaining_df.iloc[train_idx].copy()
        val_df = remaining_df.iloc[val_idx].copy()

        # Save CSVs
        train_df.to_csv(os.path.join(fold_dir, 'train_split.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val_split.csv'), index=False)
        test_df.to_csv(os.path.join(fold_dir, 'test_split.csv'), index=False)

        # Print per-fold statistics
        print(f"  Sizes:  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            label_counts = split_df['label'].value_counts().to_dict()
            proj_counts = split_df['project'].value_counts().to_dict()
            print(f"  {name:5s} labels: {label_counts}")
            print(f"  {name:5s} projects: {proj_counts}")

    print(f"\nCreated {len(projects)} folds under: {output_dir}")


def load_windows_if_exists(config: DomainConfig) -> Optional[List[dict]]:
    """Load windows from file if they exist."""
    output_dir = config.paths.data_root
    windows_output_path = os.path.join(
        output_dir,
        f"windows_mapping_{config.audio.overlap_sec}overlap.json"
    )

    if os.path.exists(windows_output_path):
        with open(windows_output_path, 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline
    python prepare_dataset.py --config data/config.yaml

    # Only compute statistics and build windows
    python prepare_dataset.py --config data/config.yaml --steps stats windows

    # Only compute spectrograms (windows must already exist)
    python prepare_dataset.py --config data/config.yaml --steps spectrograms

    # Only create splits (windows and spectrograms must already exist)
    python prepare_dataset.py --config data/config.yaml --steps splits
        """
    )

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., config/template.yaml)"
    )
    parser.add_argument(
        "--steps", type=str, nargs="+",
        default=["stats", "windows", "spectrograms", "splits"],
        choices=["stats", "windows", "spectrograms", "splits"],
        help="Steps to run (default: all)"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Track windows (needed for some steps)
    windows = None

    # Run requested steps
    if "stats" in args.steps:
        run_stats(config)

    if "windows" in args.steps:
        windows = run_windows(config)
    elif "spectrograms" in args.steps or "splits" in args.steps:
        windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return

    if "spectrograms" in args.steps:
        if windows is None:
            windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return
        run_spectrograms(config, windows)

    if "splits" in args.steps:
        if windows is None:
            windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return
        run_splits(config, windows)

    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
