"""
Generic dataset preparation script for PW_Bioacoustics.

Usage:
    # Full pipeline
    python prepare_dataset.py --config data/config.yaml

    # Run specific steps only
    python prepare_dataset.py --config data/config.yaml --steps stats windows

    # Available steps: stats, windows, segment_windows, spectrograms, splits
"""

import os
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Import from PytorchWildlife core library
from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config, DomainConfig
from PytorchWildlife.data.bioacoustics.bioacoustics_windows import build_windows


def spectrogram_filename(sound_path, start_sample, end_sample):
    """Build spectrogram .npy filename from audio path and sample range."""
    base = os.path.splitext(os.path.basename(sound_path))[0]
    return f"{base}_{start_sample}_{end_sample}.npy"


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


def run_segment_windows(config: DomainConfig, windows: List[dict], segment_duration_sec: int = 10) -> List[dict]:
    """Filter windows that cross segment boundaries and save segmented JSON.

    Keeps only windows whose start and end fall within the same fixed-length
    segment (default 10 s).  Reassigns sequential window IDs.
    """
    print(f"\n{'='*60}")
    print(f"Step: Segment Windows (filter boundary-crossing windows)")
    print(f"{'='*60}")

    output_dir = config.paths.data_root
    segmented_path = os.path.join(
        output_dir,
        f"windows_mapping_{config.audio.overlap_sec}overlap_segmented.json"
    )

    if os.path.exists(segmented_path):
        print(f"Loading existing segmented windows from: {segmented_path}")
        with open(segmented_path, 'r') as f:
            segmented = json.load(f)
        print(f"Loaded {len(segmented)} segmented windows")
    else:
        sample_rate = config.audio.sample_rate
        segment_samples = segment_duration_sec * sample_rate

        print(f"Filtering with segment_duration={segment_duration_sec}s, sample_rate={sample_rate}")
        print(f"Input windows: {len(windows)}")

        segmented = []
        for w in windows:
            start_seg = w['start'] // segment_samples
            end_seg = (w['end'] - 1) // segment_samples
            if start_seg == end_seg:
                segmented.append(w)

        # Reassign sequential window IDs
        for i, w in enumerate(segmented):
            w['window_id'] = i

        removed = len(windows) - len(segmented)
        print(f"Valid windows: {len(segmented)}")
        print(f"Removed: {removed} ({100 * removed / len(windows):.1f}%)")

        with open(segmented_path, 'w') as f:
            json.dump(segmented, f, indent=2)
        print(f"Saved to: {segmented_path}")

    counts = count_window_labels(segmented)
    print(f"\nLabel distribution: {counts}")

    return segmented


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

    Uses segmented windows.  Train/val keep overlaps within segments;
    the test set is filtered to non-overlapping windows only.
    """
    import csv
    from sklearn.model_selection import GroupShuffleSplit

    print(f"\n{'='*60}")
    print(f"Step: Create Data Splits (Leave-One-Project-Out)")
    print(f"{'='*60}")

    spectrograms_dir = config.paths.spectrograms_dir
    output_dir = config.paths.data_root

    print(f"Spectrograms directory: {spectrograms_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split parameters:")
    print(f"  - val_size: {config.splits.val_size}")
    print(f"  - random_state: {config.splits.random_state}")

    # Load annotations to map sound_id -> file path
    with open(config.paths.annotations_path, 'r') as f:
        annotations = json.load(f)
    sounds = {s['id']: s for s in annotations['sounds']}

    # Build enriched data list from windows
    data = []
    for w in windows:
        sound = sounds.get(w['sound_id'])
        if sound:
            spec_name = spectrogram_filename(sound['file_name_path'], w['start'], w['end'])
            data.append({
                'window_id': w['window_id'],
                'sound_id': w['sound_id'],
                'start': w['start'],
                'end': w['end'],
                'label': w.get('label', 0),
                'spec_name': spec_name,
                'sound_filename': os.path.basename(sound['file_name_path']),
            })

    # Add project column via metadata
    metadata_path = os.path.join(config.paths.data_root, 'metadata.csv')
    print(f"Loading metadata from: {metadata_path}")
    audio_to_project = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_to_project[row['audio_file']] = row['project_name']

    for d in data:
        d['project'] = audio_to_project.get(d['sound_filename'])

    data = [d for d in data if d['project'] is not None]
    print(f"Windows with project mapping: {len(data)}")

    # Filter to windows whose spectrogram exists on disk
    data = [d for d in data if os.path.exists(os.path.join(spectrograms_dir, d['spec_name']))]
    print(f"Windows with existing spectrograms: {len(data)}")

    projects = sorted(set(d['project'] for d in data))
    print(f"\nProjects ({len(projects)}): {projects}")

    window_size_samples = int(config.audio.window_size_sec * config.audio.sample_rate)

    fieldnames = ['window_id', 'dataset', 'sample_rate', 'sound_id',
                  'start', 'end', 'label', 'spec_name', 'sound_filename', 'project']

    def save_csv(rows, filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in rows:
                row = {k: d.get(k, '') for k in fieldnames}
                row['sample_rate'] = config.audio.sample_rate
                row['dataset'] = ''
                writer.writerow(row)

    fold_stats = []

    for fold_idx, held_out_project in enumerate(projects):
        fold_name = f"fold_{fold_idx}_{held_out_project}_segmented"
        fold_dir = os.path.join(output_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n{'-'*50}")
        print(f"Fold {fold_idx}: held-out project = {held_out_project}")
        print(f"{'-'*50}")

        # Test: non-overlapping windows from held-out project
        test_data_all = [d for d in data if d['project'] == held_out_project]

        by_sound = defaultdict(list)
        for d in test_data_all:
            by_sound[d['sound_id']].append(d)

        test_data = []
        for sound_id, sound_windows in by_sound.items():
            for d in sound_windows:
                if d['start'] % window_size_samples == 0:
                    test_data.append(d)

        print(f"  Test: {len(test_data_all)} total -> {len(test_data)} (non-overlapping)")

        # Train/Val: remaining projects, with overlaps within segments
        remaining_data = [d for d in data if d['project'] != held_out_project]

        X = list(range(len(remaining_data)))
        y = [d['label'] for d in remaining_data]
        groups = [d['sound_id'] for d in remaining_data]

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=config.splits.val_size,
            random_state=config.splits.random_state,
        )
        train_idx, val_idx = next(gss.split(X, y, groups=groups))

        train_data = [remaining_data[i] for i in train_idx]
        val_data = [remaining_data[i] for i in val_idx]

        save_csv(train_data, os.path.join(fold_dir, 'train_split.csv'))
        save_csv(val_data, os.path.join(fold_dir, 'val_split.csv'))
        save_csv(test_data, os.path.join(fold_dir, 'test_split.csv'))

        # Per-fold statistics
        print(f"  Train: {len(train_data)} (with overlaps within segments)")
        print(f"  Val:   {len(val_data)} (with overlaps within segments)")
        print(f"  Test:  {len(test_data)} (non-overlapping)")

        for name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            label_counts = defaultdict(int)
            proj_counts = defaultdict(int)
            for d in split_data:
                label_counts[d['label']] += 1
                proj_counts[d['project']] += 1
            print(f"    {name} labels: {dict(label_counts)}")
            print(f"    {name} projects: {dict(proj_counts)}")

        fold_stats.append({
            'fold': fold_name,
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data),
        })

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Created {len(projects)} folds under: {output_dir}")
    for stat in fold_stats:
        print(f"  {stat['fold']}")
        print(f"    Train: {stat['train']}, Val: {stat['val']}, Test: {stat['test']}")
    print(f"\nTrain/Val: segmented windows (no boundary-crossing, with overlaps)")
    print(f"Test: non-overlapping windows only")


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


def load_segmented_windows_if_exists(config: DomainConfig) -> Optional[List[dict]]:
    """Load segmented windows from file if they exist."""
    output_dir = config.paths.data_root
    segmented_path = os.path.join(
        output_dir,
        f"windows_mapping_{config.audio.overlap_sec}overlap_segmented.json"
    )

    if os.path.exists(segmented_path):
        with open(segmented_path, 'r') as f:
            return json.load(f)
    return None


ALL_STEPS = ["stats", "windows", "segment_windows", "spectrograms", "splits"]


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

    # Only create splits (segmented windows and spectrograms must already exist)
    python prepare_dataset.py --config data/config.yaml --steps splits
        """
    )

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., config/template.yaml)"
    )
    parser.add_argument(
        "--steps", type=str, nargs="+",
        default=ALL_STEPS,
        choices=ALL_STEPS,
        help="Steps to run (default: all)"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Track windows across steps
    windows = None
    segmented_windows = None

    # --- stats ---
    if "stats" in args.steps:
        run_stats(config)

    # --- windows ---
    if "windows" in args.steps:
        windows = run_windows(config)

    # --- segment_windows ---
    if "segment_windows" in args.steps:
        if windows is None:
            windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return
        segmented_windows = run_segment_windows(config, windows)

    # --- spectrograms (uses raw windows so all spectrograms are computed) ---
    if "spectrograms" in args.steps:
        if windows is None:
            windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return
        run_spectrograms(config, windows)

    # --- splits (uses segmented windows) ---
    if "splits" in args.steps:
        if segmented_windows is None:
            segmented_windows = load_segmented_windows_if_exists(config)
        if segmented_windows is None:
            print("\nError: Segmented windows not found. Run 'segment_windows' step first.")
            return
        run_splits(config, segmented_windows)

    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
