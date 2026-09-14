"""Regression tests for annotation-driven segmented window generation."""

import json
import os
import sys
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
    }
    audio.update(audio_overrides)
    return SimpleNamespace(
        datasets=["MAP1", "PPA1", "PPA2", "PPA3", "PPA4"],
        audio=SimpleNamespace(**audio),
        paths=SimpleNamespace(
            annotations_path=str(annotations_path),
            data_root=str(tmp_path),
        ),
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
    segmented_cache = tmp_path / "windows_mapping_4.0overlap_segmented.json"
    raw_cache.write_text(json.dumps([{"sound_id": 1, "start": 0, "end": 500}]))
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
        prepare_dataset, "run_segment_windows", lambda _: generated
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
