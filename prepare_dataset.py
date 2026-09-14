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
import re
import shutil
from collections import defaultdict
from typing import List, Optional


# Import from PytorchWildlife core library
from PytorchWildlife.data.bioacoustics.bioacoustics_configs import (
    load_config,
    DomainConfig,
)
from PytorchWildlife.data.bioacoustics.bioacoustics_windows import build_windows
from data.segment_utils import (
    PPA1_SEGMENT_STRIDE_SEC,
    iter_full_segments,
)


def spectrogram_filename(sound_path, start_sample, end_sample):
    """Build spectrogram .npy filename from audio path and sample range."""
    base = os.path.splitext(os.path.basename(sound_path))[0]
    return f"{base}_{start_sample}_{end_sample}.npy"


def count_window_labels(windows: List[dict]) -> dict:
    """Count label distribution in windows."""
    counts = {}
    for w in windows:
        label = w.get("label", 0)
        counts[label] = counts.get(label, 0) + 1
    return counts


def run_stats(config: DomainConfig) -> None:
    """Load and display dataset statistics."""
    print(f"\n{'=' * 60}")
    print("Step: Dataset Statistics")
    print(f"{'=' * 60}")

    annotation_path = config.paths.annotations_path
    print(f"Loading annotations from: {annotation_path}")

    if not os.path.exists(annotation_path):
        print(f"Warning: Annotations file not found: {annotation_path}")
        return

    with open(annotation_path, "r") as f:
        data = json.load(f)

    # Dataset info
    if "info" in data:
        print("\nDataset Info:")
        for key, value in data["info"].items():
            print(f"  - {key}: {value}")

    # Sound statistics
    sounds = data.get("sounds", [])
    print(f"\nSounds: {len(sounds)}")
    if sounds:
        durations = [s.get("duration", 0) for s in sounds]
        print(
            f"  - Total duration: {sum(durations):.1f}s ({sum(durations) / 3600:.2f}h)"
        )
        print(f"  - Mean duration: {sum(durations) / len(durations):.1f}s")
        print(f"  - Min duration: {min(durations):.1f}s")
        print(f"  - Max duration: {max(durations):.1f}s")

    # Annotation statistics
    annotations = data.get("annotations", [])
    print(f"\nAnnotations: {len(annotations)}")
    if annotations:
        categories = {}
        for ann in annotations:
            cat_id = ann.get("category_id", 0)
            categories[cat_id] = categories.get(cat_id, 0) + 1
        print(f"  - By category: {categories}")

    # Category names
    if "categories" in data:
        print("\nCategories:")
        for cat in data["categories"]:
            print(f"  - {cat.get('id', '?')}: {cat.get('name', 'Unknown')}")


def run_windows(config: DomainConfig) -> List[dict]:
    """Build windows from annotations."""
    print(f"\n{'=' * 60}")
    print("Step: Build Windows")
    print(f"{'=' * 60}")

    annotation_path = config.paths.annotations_path
    output_dir = config.paths.data_root
    os.makedirs(output_dir, exist_ok=True)

    windows_output_path = os.path.join(
        output_dir, f"windows_mapping_{config.audio.overlap_sec}overlap.json"
    )

    if os.path.exists(windows_output_path):
        print(f"Loading existing windows from: {windows_output_path}")
        with open(windows_output_path, "r") as f:
            windows = json.load(f)
        print(f"Loaded {len(windows)} windows")
    else:
        strategy = config.audio.window_strategy
        print("Building windows with:")
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

        with open(windows_output_path, "w") as f:
            json.dump(windows, f, indent=2)
        print(f"Saved {len(windows)} windows to: {windows_output_path}")

    # Show label distribution
    counts = count_window_labels(windows)
    print(f"\nLabel distribution: {counts}")

    return windows


def build_segmented_windows(
    annotations_data: dict,
    datasets: List[str],
    sample_rate: int,
    window_size_sec: float,
    overlap_sec: float,
    window_strategy: str = "sliding",
    segment_duration_sec: float = 10,
) -> List[dict]:
    """Build segmented windows directly from sound and annotation geometry."""
    if window_strategy != "sliding":
        raise ValueError(
            "annotation-driven segmented windows require window_strategy='sliding'"
        )
    window_size_samples = round(window_size_sec * sample_rate)
    hop_samples = round((window_size_sec - overlap_sec) * sample_rate)
    if window_size_samples <= 0:
        raise ValueError("window_size_sec must produce at least one sample")
    if hop_samples <= 0:
        raise ValueError("overlap_sec must be smaller than window_size_sec")
    if window_size_sec > segment_duration_sec:
        raise ValueError("window_size_sec must not exceed segment_duration_sec")

    sound_to_anns = defaultdict(list)
    for annotation in annotations_data.get("annotations", []):
        sound_to_anns[annotation["sound_id"]].append(
            (annotation["t_min"], annotation["t_max"])
        )

    dataset_names = set(datasets)
    segmented = []
    seen_geometry = set()
    for sound in annotations_data.get("sounds", []):
        project = sound.get("project")
        if project not in dataset_names:
            project = next(
                (
                    name
                    for name in datasets
                    if name in sound.get("file_name_path", "")
                ),
                None,
            )
        if project is None or project not in dataset_names:
            continue

        sound_id = sound["id"]
        for segment in iter_full_segments(
            duration_sec=sound["duration"],
            source_sample_rate=sample_rate,
            project=project,
            segment_duration_sec=segment_duration_sec,
        ):
            last_start = segment.end_sample - window_size_samples
            for start in range(segment.start_sample, last_start + 1, hop_samples):
                end = start + window_size_samples
                geometry = (sound_id, start, end)
                if geometry in seen_geometry:
                    continue
                seen_geometry.add(geometry)

                start_sec = start / sample_rate
                end_sec = end / sample_rate
                label = int(
                    any(
                        annotation_start < end_sec and annotation_end > start_sec
                        for annotation_start, annotation_end in sound_to_anns.get(
                            sound_id, []
                        )
                    )
                )
                segmented.append(
                    {
                        "window_id": len(segmented),
                        "dataset": project,
                        "sample_rate": sample_rate,
                        "sound_id": sound_id,
                        "start": start,
                        "end": end,
                        "label": label,
                    }
                )

    return segmented


DEFAULT_DATASET_VERSION = "v5"


def validate_dataset_version(version: str) -> str:
    """Validate a new annotation-derived dataset revision suffix."""
    match = re.fullmatch(r"v([1-9][0-9]*)", version)
    if match is None:
        raise ValueError("dataset version must use the form vN")
    if int(match.group(1)) < 5:
        raise ValueError("dataset versions v1-v4 are historical and read-only")
    return version


def segmented_mapping_path(config: DomainConfig, version: str) -> str:
    """Return the revision-specific segmented-window mapping path."""
    version = validate_dataset_version(version)
    return os.path.join(
        config.paths.data_root,
        (
            f"windows_mapping_{config.audio.overlap_sec}overlap"
            f"_segmented_{version}.json"
        ),
    )


def run_segment_windows(
    config: DomainConfig,
    version: str = DEFAULT_DATASET_VERSION,
    segment_duration_sec: float = 10,
) -> List[dict]:
    """Generate and save segmented windows from the current annotations."""
    print(f"\n{'=' * 60}")
    print(f"Step: Generate Segmented Windows ({version})")
    print(f"{'=' * 60}")

    output_dir = config.paths.data_root
    segmented_path = segmented_mapping_path(config, version)

    sample_rate = config.audio.sample_rate
    with open(config.paths.annotations_path, "r") as f:
        annotations_data = json.load(f)

    print(
        f"Generating with segment_duration={segment_duration_sec}s, "
        f"sample_rate={sample_rate}"
    )
    print(
        f"  PPA1 stride: {PPA1_SEGMENT_STRIDE_SEC:g}s, "
        f"default stride: {segment_duration_sec}s"
    )
    segmented = build_segmented_windows(
        annotations_data=annotations_data,
        datasets=config.datasets,
        sample_rate=sample_rate,
        window_size_sec=config.audio.window_size_sec,
        overlap_sec=config.audio.overlap_sec,
        window_strategy=config.audio.window_strategy,
        segment_duration_sec=segment_duration_sec,
    )
    print(f"Generated windows: {len(segmented)}")

    os.makedirs(output_dir, exist_ok=True)
    with open(segmented_path, "w") as f:
        json.dump(segmented, f, indent=2)
    print(f"Saved to: {segmented_path}")

    counts = count_window_labels(segmented)
    print(f"\nLabel distribution: {counts}")

    return segmented


def run_spectrograms(config: DomainConfig, windows: List[dict]) -> None:
    """Compute mel spectrograms using GPU."""
    # Import here to avoid loading torch unnecessarily
    from PytorchWildlife.data.bioacoustics.bioacoustics_spectrograms import (
        compute_mel_spectrograms_gpu,
    )

    print(f"\n{'=' * 60}")
    print("Step: Compute Mel Spectrograms (GPU)")
    print(f"{'=' * 60}")

    spectrograms_dir = config.paths.spectrograms_dir
    os.makedirs(spectrograms_dir, exist_ok=True)

    print(f"Output directory: {spectrograms_dir}")
    print("Spectrogram parameters:")
    print(f"  - n_fft: {config.spectrogram.n_fft}")
    print(f"  - hop_length: {config.spectrogram.hop_length}")
    print(f"  - n_mels: {config.spectrogram.n_mels}")
    print(f"  - top_db: {config.spectrogram.top_db}")
    print(f"  - fill_highfreq: {config.spectrogram.fill_highfreq}")

    # Load annotations to get audio file paths
    with open(config.paths.annotations_path, "r") as f:
        annotations = json.load(f)

    sounds = {s["id"]: s for s in annotations["sounds"]}

    # Convert windows format to include sound_path
    inference_windows = []
    for win in windows:
        sound = sounds.get(win["sound_id"])
        if sound:
            inference_windows.append(
                {
                    "window_id": win["window_id"],
                    "sound_path": sound["file_name_path"],
                    "start": win["start"],
                    "end": win["end"],
                }
            )

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


def run_splits(
    config: DomainConfig,
    windows: List[dict],
    folds_subdir: Optional[str] = None,
    version: str = DEFAULT_DATASET_VERSION,
) -> None:
    """Create leave-one-project-out cross-validation splits.

    Uses segmented windows.  Train/val keep overlaps within segments;
    the test set is filtered to non-overlapping windows only.
    """
    import csv
    from sklearn.model_selection import GroupShuffleSplit

    print(f"\n{'=' * 60}")
    print("Step: Create Data Splits (Leave-One-Project-Out)")
    print(f"{'=' * 60}")

    spectrograms_dir = config.paths.spectrograms_dir
    output_dir = config.paths.data_root
    version = validate_dataset_version(version)
    expected_folds_subdir = f"folds_segmented_{version}"
    if folds_subdir is None:
        folds_subdir = expected_folds_subdir
    elif folds_subdir != expected_folds_subdir:
        raise ValueError(
            f"folds_subdir must be {expected_folds_subdir!r} for version {version}"
        )
    folds_base = os.path.join(output_dir, folds_subdir)

    print(f"Spectrograms directory: {spectrograms_dir}")
    print(f"Output directory: {folds_base}")
    print("Split parameters:")
    print(f"  - val_size: {config.splits.val_size}")
    print(f"  - random_state: {config.splits.random_state}")

    # Load annotations to map sound_id -> file path
    with open(config.paths.annotations_path, "r") as f:
        annotations = json.load(f)
    sounds = {s["id"]: s for s in annotations["sounds"]}

    # Build enriched data list from windows
    data = []
    for w in windows:
        sound = sounds.get(w["sound_id"])
        if sound:
            spec_name = spectrogram_filename(
                sound["file_name_path"], w["start"], w["end"]
            )
            data.append(
                {
                    "window_id": w["window_id"],
                    "dataset": w.get("dataset") or sound.get("project"),
                    "sound_id": w["sound_id"],
                    "start": w["start"],
                    "end": w["end"],
                    "label": w.get("label", 0),
                    "spec_name": spec_name,
                    "sound_filename": os.path.basename(sound["file_name_path"]),
                    "project": w.get("dataset") or sound.get("project"),
                }
            )

    data = [d for d in data if d["project"] is not None]
    print(f"Windows with project mapping: {len(data)}")

    # Filter to windows whose spectrogram exists on disk
    data = [
        d
        for d in data
        if os.path.exists(os.path.join(spectrograms_dir, d["spec_name"]))
    ]
    print(f"Windows with existing spectrograms: {len(data)}")

    projects = sorted(set(d["project"] for d in data))
    print(f"\nProjects ({len(projects)}): {projects}")

    if os.path.islink(folds_base):
        raise ValueError(f"folds directory must not be a symlink: {folds_base}")
    if os.path.isdir(folds_base):
        generated_entries = [
            entry for entry in os.scandir(folds_base) if entry.name.startswith("fold_")
        ]
        for entry in generated_entries:
            if entry.is_symlink():
                raise ValueError(
                    f"generated fold path must not be a symlink: {entry.path}"
                )
        for entry in generated_entries:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
    os.makedirs(folds_base, exist_ok=True)

    window_size_samples = round(
        config.audio.window_size_sec * config.audio.sample_rate
    )

    fieldnames = [
        "window_id",
        "dataset",
        "sample_rate",
        "sound_id",
        "start",
        "end",
        "label",
        "spec_name",
        "sound_filename",
        "project",
    ]

    def save_csv(rows, filepath):
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in rows:
                row = {k: d.get(k, "") for k in fieldnames}
                row["sample_rate"] = config.audio.sample_rate
                writer.writerow(row)

    fold_stats = []

    for fold_idx, held_out_project in enumerate(projects):
        fold_name = f"fold_{fold_idx}_{held_out_project}_segmented"
        fold_dir = os.path.join(folds_base, fold_name)
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n{'-' * 50}")
        print(f"Fold {fold_idx}: held-out project = {held_out_project}")
        print(f"{'-' * 50}")

        # Test: non-overlapping windows from held-out project
        test_data_all = [d for d in data if d["project"] == held_out_project]

        by_sound = defaultdict(list)
        for d in test_data_all:
            by_sound[d["sound_id"]].append(d)

        test_data = []
        for sound_id, sound_windows in by_sound.items():
            for d in sound_windows:
                if d["start"] % window_size_samples == 0:
                    test_data.append(d)

        print(
            f"  Test: {len(test_data_all)} total -> {len(test_data)} (non-overlapping)"
        )

        # Train/Val: remaining projects, with overlaps within segments
        remaining_data = [d for d in data if d["project"] != held_out_project]

        X = list(range(len(remaining_data)))
        y = [d["label"] for d in remaining_data]
        groups = [d["sound_id"] for d in remaining_data]

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=config.splits.val_size,
            random_state=config.splits.random_state,
        )
        train_idx, val_idx = next(gss.split(X, y, groups=groups))

        train_data = [remaining_data[i] for i in train_idx]
        val_data = [remaining_data[i] for i in val_idx]

        save_csv(train_data, os.path.join(fold_dir, "train_split.csv"))
        save_csv(val_data, os.path.join(fold_dir, "val_split.csv"))
        save_csv(test_data, os.path.join(fold_dir, "test_split.csv"))

        # Per-fold statistics
        print(f"  Train: {len(train_data)} (with overlaps within segments)")
        print(f"  Val:   {len(val_data)} (with overlaps within segments)")
        print(f"  Test:  {len(test_data)} (non-overlapping)")

        for name, split_data in [
            ("Train", train_data),
            ("Val", val_data),
            ("Test", test_data),
        ]:
            label_counts = defaultdict(int)
            proj_counts = defaultdict(int)
            for d in split_data:
                label_counts[d["label"]] += 1
                proj_counts[d["project"]] += 1
            print(f"    {name} labels: {dict(label_counts)}")
            print(f"    {name} projects: {dict(proj_counts)}")

        fold_stats.append(
            {
                "fold": fold_name,
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
            }
        )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Created {len(projects)} folds under: {folds_base}")
    for stat in fold_stats:
        print(f"  {stat['fold']}")
        print(f"    Train: {stat['train']}, Val: {stat['val']}, Test: {stat['test']}")
    print("\nTrain/Val: segmented windows (no boundary-crossing, with overlaps)")
    print("Test: non-overlapping windows only")


def load_segmented_windows_if_exists(
    config: DomainConfig,
    version: str = DEFAULT_DATASET_VERSION,
) -> Optional[List[dict]]:
    """Load segmented windows from file if they exist."""
    segmented_path = segmented_mapping_path(config, version)

    if os.path.exists(segmented_path):
        with open(segmented_path, "r") as f:
            return json.load(f)
    return None


ALL_STEPS = ["stats", "windows", "segment_windows", "spectrograms", "splits"]
DEFAULT_STEPS = ["stats", "segment_windows", "spectrograms", "splits"]


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

    # Only compute spectrograms (segmented windows are regenerated first)
    python prepare_dataset.py --config data/config.yaml --steps spectrograms

    # Only create splits (segmented windows are regenerated first)
    python prepare_dataset.py --config data/config.yaml --steps splits
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., config/template.yaml)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        default=DEFAULT_STEPS,
        choices=ALL_STEPS,
        help="Steps to run (default: stats segment_windows spectrograms splits)",
    )
    parser.add_argument(
        "--version",
        type=validate_dataset_version,
        default=DEFAULT_DATASET_VERSION,
        help=(
            "Dataset revision suffix for segmented mapping and folds "
            f"(default: {DEFAULT_DATASET_VERSION})"
        ),
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    segmented_windows = None

    # --- stats ---
    if "stats" in args.steps:
        run_stats(config)

    # --- windows ---
    if "windows" in args.steps:
        run_windows(config)

    # --- segment_windows ---
    if "segment_windows" in args.steps:
        segmented_windows = run_segment_windows(config, version=args.version)

    # --- spectrograms (uses exactly the annotation-derived segmented windows) ---
    if "spectrograms" in args.steps:
        if segmented_windows is None:
            segmented_windows = run_segment_windows(config, version=args.version)
        run_spectrograms(config, segmented_windows)

    # --- splits (uses segmented windows) ---
    if "splits" in args.steps:
        if segmented_windows is None:
            segmented_windows = run_segment_windows(config, version=args.version)
        run_splits(
            config,
            segmented_windows,
            version=args.version,
        )

    print(f"\n{'=' * 60}")
    print("Dataset preparation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
