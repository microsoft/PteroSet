"""Focused tests for historical inference comparison inputs."""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

import compare_inference_strategies


def _config(tmp_path: Path):
    spectrograms = tmp_path / "spectrograms"
    spectrograms.mkdir()
    return SimpleNamespace(
        audio=SimpleNamespace(sample_rate=100),
        paths=SimpleNamespace(
            data_root=str(tmp_path),
            spectrograms_dir=str(spectrograms),
        ),
    )


def _fold_dirs(tmp_path: Path) -> Path:
    folds = tmp_path / "folds"
    for fold_idx, project in enumerate(compare_inference_strategies.PROJECTS):
        (folds / f"fold_{fold_idx}_{project}_segmented").mkdir(parents=True)
    return folds


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def test_generates_csvs_without_metadata_using_window_dataset_and_sound_fallback(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    folds = _fold_dirs(tmp_path)
    monkeypatch.setattr(
        compare_inference_strategies,
        "spectrogram_filename",
        lambda file_name, start, end: f"{Path(file_name).stem}_{start}_{end}.npy",
    )
    for name in ("map_0_500.npy", "ppa1_0_500.npy"):
        (Path(config.paths.spectrograms_dir) / name).touch()

    windows = [
        {
            "window_id": 1,
            "dataset": "MAP1",
            "sound_id": 10,
            "start": 0,
            "end": 500,
            "label": 1,
        },
        {
            "window_id": 2,
            "sound_id": 20,
            "start": 0,
            "end": 500,
            "label": 0,
        },
    ]
    annotations = {
        "sounds": [
            {"id": 10, "file_name_path": "map.wav", "project": "MAP1"},
            {"id": 20, "file_name_path": "ppa1.wav", "project": "PPA1"},
        ]
    }

    paths = compare_inference_strategies.generate_overlapping_test_csvs(
        config,
        windows,
        str(folds),
        annotations,
    )

    map_rows = _read_rows(paths[0])
    ppa1_rows = _read_rows(paths[1])
    assert [(row["dataset"], row["project"]) for row in map_rows] == [
        ("MAP1", "MAP1")
    ]
    assert [(row["dataset"], row["project"]) for row in ppa1_rows] == [
        ("PPA1", "PPA1")
    ]
    assert not (tmp_path / "metadata.csv").exists()


def test_rejects_conflicting_window_and_annotation_projects(tmp_path, monkeypatch):
    config = _config(tmp_path)
    folds = _fold_dirs(tmp_path)
    monkeypatch.setattr(
        compare_inference_strategies,
        "spectrogram_filename",
        lambda *_: "unused.npy",
    )
    windows = [
        {
            "window_id": 1,
            "dataset": "MAP1",
            "sound_id": 10,
            "start": 0,
            "end": 500,
        }
    ]
    annotations = {
        "sounds": [
            {"id": 10, "file_name_path": "sound.wav", "project": "PPA1"}
        ]
    }

    with pytest.raises(ValueError, match="Project mismatch"):
        compare_inference_strategies.generate_overlapping_test_csvs(
            config,
            windows,
            str(folds),
            annotations,
        )


def test_rejects_missing_project_identity(tmp_path, monkeypatch):
    config = _config(tmp_path)
    folds = _fold_dirs(tmp_path)
    monkeypatch.setattr(
        compare_inference_strategies,
        "spectrogram_filename",
        lambda *_: "unused.npy",
    )
    windows = [
        {"window_id": 1, "sound_id": 10, "start": 0, "end": 500}
    ]
    annotations = {
        "sounds": [{"id": 10, "file_name_path": "sound.wav"}]
    }

    with pytest.raises(ValueError, match="No project identity"):
        compare_inference_strategies.generate_overlapping_test_csvs(
            config,
            windows,
            str(folds),
            annotations,
        )


def test_rejects_window_with_unknown_sound(tmp_path):
    config = _config(tmp_path)
    folds = _fold_dirs(tmp_path)
    windows = [
        {
            "window_id": 1,
            "dataset": "MAP1",
            "sound_id": 999,
            "start": 0,
            "end": 500,
        }
    ]

    with pytest.raises(ValueError, match="unknown sound_id 999"):
        compare_inference_strategies.generate_overlapping_test_csvs(
            config,
            windows,
            str(folds),
            {"sounds": []},
        )


def test_rejects_unsupported_project_identity(tmp_path, monkeypatch):
    config = _config(tmp_path)
    folds = _fold_dirs(tmp_path)
    monkeypatch.setattr(
        compare_inference_strategies,
        "spectrogram_filename",
        lambda *_: "unused.npy",
    )
    windows = [
        {
            "window_id": 1,
            "dataset": "UNKNOWN",
            "sound_id": 10,
            "start": 0,
            "end": 500,
        }
    ]
    annotations = {
        "sounds": [
            {"id": 10, "file_name_path": "sound.wav", "project": "UNKNOWN"}
        ]
    }

    with pytest.raises(ValueError, match="Unsupported project 'UNKNOWN'"):
        compare_inference_strategies.generate_overlapping_test_csvs(
            config,
            windows,
            str(folds),
            annotations,
        )
