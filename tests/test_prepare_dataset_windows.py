"""Regression tests for annotation-driven segmented window generation."""

import json
import os
import sys
import csv
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import prepare_dataset
from prepare_dataset import build_segmented_windows, run_segment_windows


def _config(tmp_path: Path, annotations: dict, **audio_overrides):
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(annotations))
    audio = {
        "sample_rate": 100,
        "window_size_sec": 5.0,
        "overlap_sec": 4.0,
        "window_strategy": "sliding",
    }
    audio.update(audio_overrides)
    return SimpleNamespace(
        datasets=["MAP1", "PPA1", "PPA2", "PPA3", "PPA4"],
        audio=SimpleNamespace(**audio),
        paths=SimpleNamespace(
            annotations_path=str(annotations_path),
            data_root=str(tmp_path),
            spectrograms_dir=str(tmp_path / "spectrograms"),
        ),
        splits=SimpleNamespace(val_size=0.5, random_state=42),
    )


def test_builds_six_windows_per_complete_default_segment(tmp_path):
    config = _config(
        tmp_path,
        {
            "sounds": [
                {
                    "id": 1,
                    "file_name_path": "PPA2.wav",
                    "duration": 480,
                    "sample_rate": 100,
                    "project": "PPA2",
                }
            ],
            "annotations": [],
        },
    )

    windows = run_segment_windows(config)

    assert len(windows) == 48 * 6
    assert [window["start"] for window in windows[:6]] == [
        0,
        100,
        200,
        300,
        400,
        500,
    ]
    assert windows[-1]["end"] == 48_000


def test_builds_unique_windows_for_ppa1_crossfade_geometry(tmp_path):
    config = _config(
        tmp_path,
        {
            "sounds": [
                {
                    "id": "ppa1",
                    "file_name_path": "PPA1.wav",
                    "duration": 433,
                    "sample_rate": 192_000,
                    "project": "PPA1",
                }
            ],
            "annotations": [],
        },
    )

    windows = run_segment_windows(config)
    geometry = {(window["sound_id"], window["start"], window["end"]) for window in windows}

    assert len(windows) == 48 * 6
    assert len(geometry) == len(windows)
    assert windows[6]["start"] == 900
    assert windows[-1]["end"] == 43_300


def test_deduplicates_windows_shared_by_overlapping_ppa1_segments(tmp_path):
    config = _config(
        tmp_path,
        {
            "sounds": [
                {
                    "id": "ppa1",
                    "file_name_path": "PPA1.wav",
                    "duration": 433,
                    "sample_rate": 100,
                    "project": "PPA1",
                }
            ],
            "annotations": [],
        },
        window_size_sec=1.0,
        overlap_sec=0.0,
    )

    windows = run_segment_windows(config)

    assert len(windows) == 433
    assert len({(window["start"], window["end"]) for window in windows}) == 433


@pytest.mark.parametrize(
    ("duration", "expected_segments"),
    [(440, 44), (460, 46), (470, 47), (479.9, 47)],
)
def test_uses_only_complete_ppa4_segments(tmp_path, duration, expected_segments):
    config = _config(
        tmp_path,
        {
            "sounds": [
                {
                    "id": "ppa4",
                    "file_name_path": "PPA4.wav",
                    "duration": duration,
                    "sample_rate": 100,
                    "project": "PPA4",
                }
            ],
            "annotations": [],
        },
    )

    assert len(run_segment_windows(config)) == expected_segments * 6


def test_supports_general_window_and_overlap_values(tmp_path):
    config = _config(
        tmp_path,
        {
            "sounds": [
                {
                    "id": 1,
                    "file_name_path": "MAP1.wav",
                    "duration": 10,
                    "sample_rate": 100,
                    "project": "MAP1",
                }
            ],
            "annotations": [],
        },
        window_size_sec=4.0,
        overlap_sec=2.0,
    )

    windows = run_segment_windows(config)

    assert [(window["start"], window["end"]) for window in windows] == [
        (0, 400),
        (200, 600),
        (400, 800),
        (600, 1000),
    ]


def test_ignores_raw_and_segmented_caches_and_rederives_labels(tmp_path):
    annotations = {
        "sounds": [
            {
                "id": 1,
                "file_name_path": "MAP1.wav",
                "duration": 10,
                "sample_rate": 100,
                "project": "MAP1",
            }
        ],
        "annotations": [],
    }
    config = _config(tmp_path, annotations)
    raw_cache = tmp_path / "windows_mapping_4.0overlap.json"
    historical_v4 = tmp_path / "windows_mapping_4.0overlap_segmented_v4.json"
    segmented_cache = tmp_path / "windows_mapping_4.0overlap_segmented_v5.json"
    raw_cache.write_text(json.dumps([{"sound_id": 1, "start": 0, "end": 500}]))
    historical_v4.write_text(json.dumps([{"label": 4}]))
    segmented_cache.write_text(json.dumps([{"label": 99}]))

    first = run_segment_windows(config)
    assert len(first) == 6
    assert {window["label"] for window in first} == {0}

    annotations["annotations"] = [
        {"sound_id": 1, "t_min": 5.5, "t_max": 6.5, "category_id": 0}
    ]
    Path(config.paths.annotations_path).write_text(json.dumps(annotations))
    second = run_segment_windows(config)

    assert len(second) == 6
    assert [window["label"] for window in second] == [0, 1, 1, 1, 1, 1]
    assert json.loads(segmented_cache.read_text()) == second

    annotations["annotations"] = []
    Path(config.paths.annotations_path).write_text(json.dumps(annotations))
    third = run_segment_windows(config)
    assert {window["label"] for window in third} == {0}
    assert json.loads(segmented_cache.read_text()) == third
    assert json.loads(historical_v4.read_text()) == [{"label": 4}]


@pytest.mark.parametrize(
    ("t_min", "t_max", "expected_label"),
    [(6.0, 7.0, 0), (4.0, 5.0, 0), (5.999, 6.001, 1)],
)
def test_label_overlap_uses_strict_interval_intersection(
    tmp_path, t_min, t_max, expected_label
):
    config = _config(
        tmp_path,
        {
            "sounds": [
                {
                    "id": 1,
                    "file_name_path": "MAP1.wav",
                    "duration": 10,
                    "sample_rate": 100,
                    "project": "MAP1",
                }
            ],
            "annotations": [
                {"sound_id": 1, "t_min": t_min, "t_max": t_max, "category_id": 0}
            ],
        },
        window_size_sec=1.0,
        overlap_sec=0.0,
    )

    windows = run_segment_windows(config)

    assert windows[5]["label"] == expected_label


def test_spectrogram_step_regenerates_and_uses_segmented_windows(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, {"sounds": [], "annotations": []})
    generated = [{"window_id": 0, "sound_id": 1, "start": 0, "end": 500}]
    received = []
    monkeypatch.setattr(prepare_dataset, "load_config", lambda _: config)
    monkeypatch.setattr(
        prepare_dataset,
        "run_segment_windows",
        lambda _, version: generated,
    )
    monkeypatch.setattr(
        prepare_dataset,
        "run_spectrograms",
        lambda _, windows: received.extend(windows),
    )
    monkeypatch.setattr(
        prepare_dataset,
        "run_windows",
        lambda _: pytest.fail("raw windows must not be generated"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_dataset.py", "--config", "config.yaml", "--steps", "spectrograms"],
    )

    prepare_dataset.main()

    assert received == generated


def test_versioned_mapping_and_fold_paths_do_not_overwrite_v4(tmp_path):
    config = _config(tmp_path, {"sounds": [], "annotations": []})
    historical_v4 = tmp_path / "windows_mapping_4.0overlap_segmented_v4.json"
    historical_v4.write_text("historical")

    prepare_dataset.run_segment_windows(config, version="v5")

    assert historical_v4.read_text() == "historical"
    assert (tmp_path / "windows_mapping_4.0overlap_segmented_v5.json").exists()


def test_rejects_historical_fold_destination_without_deleting_it(tmp_path):
    config = _config(tmp_path, {"sounds": [], "annotations": []})
    historical_fold = tmp_path / "folds_segmented_v4" / "fold_0_MAP1_segmented"
    historical_fold.mkdir(parents=True)
    historical_csv = historical_fold / "train_split.csv"
    historical_csv.write_text("historical")

    with pytest.raises(ValueError, match="folds_subdir must be"):
        prepare_dataset.run_splits(
            config,
            [],
            folds_subdir="folds_segmented_v4",
            version="v5",
        )

    assert historical_csv.read_text() == "historical"


@pytest.mark.parametrize("version", ["v4", "v1", "../v5", "v5/test", "5"])
def test_rejects_historical_or_invalid_output_versions(tmp_path, version):
    config = _config(tmp_path, {"sounds": [], "annotations": []})

    with pytest.raises(ValueError, match="historical|form vN"):
        prepare_dataset.run_segment_windows(config, version=version)


def test_rejects_non_sliding_window_strategy(tmp_path):
    config = _config(
        tmp_path,
        {"sounds": [], "annotations": []},
        window_strategy="balanced",
    )

    with pytest.raises(ValueError, match="window_strategy='sliding'"):
        prepare_dataset.run_segment_windows(config)


def test_split_step_regenerates_versioned_windows(tmp_path, monkeypatch):
    config = _config(tmp_path, {"sounds": [], "annotations": []})
    generated = [{"window_id": 0, "dataset": "MAP1"}]
    calls = []
    monkeypatch.setattr(prepare_dataset, "load_config", lambda _: config)
    monkeypatch.setattr(
        prepare_dataset,
        "run_segment_windows",
        lambda _, version: calls.append(("generate", version)) or generated,
    )
    monkeypatch.setattr(
        prepare_dataset,
        "run_splits",
        lambda _, windows, version: calls.append(("split", windows, version)),
    )
    monkeypatch.setattr(
        prepare_dataset,
        "load_segmented_windows_if_exists",
        lambda *_args, **_kwargs: pytest.fail("stale mapping must not be loaded"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dataset.py",
            "--config",
            "config.yaml",
            "--steps",
            "splits",
            "--version",
            "v5",
        ],
    )

    prepare_dataset.main()

    assert calls == [("generate", "v5"), ("split", generated, "v5")]


def test_splits_use_window_dataset_without_metadata_csv(tmp_path):
    sounds = []
    windows = []
    spectrograms_dir = tmp_path / "spectrograms"
    spectrograms_dir.mkdir()
    window_id = 0
    for project_index, project in enumerate(("MAP1", "PPA1", "PPA2")):
        for sound_index in range(2):
            sound_id = project_index * 10 + sound_index
            file_name = f"{project}-{sound_index}.wav"
            sounds.append(
                {
                    "id": sound_id,
                    "file_name_path": file_name,
                    "duration": 10,
                    "sample_rate": 100,
                    "project": project,
                }
            )
            windows.append(
                {
                    "window_id": window_id,
                    "dataset": project,
                    "sample_rate": 100,
                    "sound_id": sound_id,
                    "start": 0,
                    "end": 500,
                    "label": sound_index % 2,
                }
            )
            window_id += 1
            (spectrograms_dir / f"{project}-{sound_index}_0_500.npy").touch()

    config = _config(tmp_path, {"sounds": sounds, "annotations": []})
    stale_fold = tmp_path / "folds_segmented_v5" / "fold_9_STALE_segmented"
    stale_fold.mkdir(parents=True)
    (stale_fold / "train_split.csv").write_text("stale")
    prepare_dataset.run_splits(config, windows, version="v5")

    folds_root = tmp_path / "folds_segmented_v5"
    assert folds_root.is_dir()
    assert not stale_fold.exists()
    assert not (tmp_path / "metadata.csv").exists()
    csv_paths = list(folds_root.glob("*/*_split.csv"))
    assert csv_paths
    for csv_path in csv_paths:
        with csv_path.open() as csv_file:
            rows = list(csv.DictReader(csv_file))
        for row in rows:
            assert row["dataset"] in {"MAP1", "PPA1", "PPA2"}
            assert row["dataset"] == row["project"]


def test_split_non_overlap_filter_uses_rounded_window_samples(tmp_path):
    sounds = []
    windows = []
    spectrograms_dir = tmp_path / "spectrograms"
    spectrograms_dir.mkdir()
    for project_index, project in enumerate(("MAP1", "PPA1", "PPA2")):
        for sound_index in range(2):
            sound_id = project_index * 10 + sound_index
            file_name = f"{project}-{sound_index}.wav"
            sounds.append(
                {
                    "id": sound_id,
                    "file_name_path": file_name,
                    "duration": 10,
                    "sample_rate": 10,
                    "project": project,
                }
            )
            windows.append(
                {
                    "window_id": len(windows),
                    "dataset": project,
                    "sample_rate": 10,
                    "sound_id": sound_id,
                    "start": 3,
                    "end": 6,
                    "label": 0,
                }
            )
            (spectrograms_dir / f"{project}-{sound_index}_3_6.npy").touch()

    config = _config(
        tmp_path,
        {"sounds": sounds, "annotations": []},
        sample_rate=10,
        window_size_sec=0.26,
        overlap_sec=0.0,
    )
    prepare_dataset.run_splits(config, windows, version="v5")

    test_csv = (
        tmp_path
        / "folds_segmented_v5"
        / "fold_0_MAP1_segmented"
        / "test_split.csv"
    )
    with test_csv.open() as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert {int(row["start"]) for row in rows} == {3}


def test_current_annotations_produce_expected_project_totals():
    annotations_path = os.environ.get("PTEROSET_ANNOTATIONS")
    if not annotations_path:
        pytest.skip("set PTEROSET_ANNOTATIONS to validate the full local dataset")

    annotations = json.loads(Path(annotations_path).read_text())
    windows = build_segmented_windows(
        annotations_data=annotations,
        datasets=["MAP1", "PPA1", "PPA2", "PPA3", "PPA4"],
        sample_rate=48_000,
        window_size_sec=5.0,
        overlap_sec=4.0,
    )

    assert Counter(window["dataset"] for window in windows) == {
        "MAP1": 13_248,
        "PPA1": 31_104,
        "PPA2": 39_456,
        "PPA3": 43_488,
        "PPA4": 34_770,
    }
    assert len(windows) == 162_066
