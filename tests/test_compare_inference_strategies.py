"""Focused tests for historical inference comparison inputs."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

import compare_inference_strategies


def _config(tmp_path: Path):
    spectrograms = tmp_path / "spectrograms"
    spectrograms.mkdir()
    return SimpleNamespace(
        audio=SimpleNamespace(sample_rate=100, overlap_sec=4.0),
        paths=SimpleNamespace(
            data_root=str(tmp_path),
            spectrograms_dir=str(spectrograms),
        ),
    )


def _fold_dirs(tmp_path: Path, sounds: dict | None = None) -> Path:
    folds = tmp_path / "folds"
    sounds = sounds or {}
    for fold_idx, project in enumerate(compare_inference_strategies.PROJECTS):
        fold_dir = folds / f"fold_{fold_idx}_{project}_segmented"
        fold_dir.mkdir(parents=True)
        with (fold_dir / "test_split.csv").open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["sound_id", "sound_filename", "dataset", "project"],
            )
            writer.writeheader()
            for sound_id, sound_filename in sounds.get(project, []):
                writer.writerow(
                    {
                        "sound_id": sound_id,
                        "sound_filename": sound_filename,
                        "dataset": project,
                        "project": project,
                    }
                )
    return folds


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def test_generates_csvs_without_metadata_using_window_dataset_and_sound_fallback(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    folds = _fold_dirs(
        tmp_path,
        {
            "MAP1": [(10, "map.wav")],
            "PPA1": [(20, "ppa1.wav")],
        },
    )
    staging = tmp_path / "outputs" / "staging" / "v3"
    monkeypatch.setattr(
        compare_inference_strategies,
        "spectrogram_filename",
        lambda file_name, start, end: f"{Path(file_name).stem}_{start}_{end}.npy",
    )
    for name in ("map_0_500.npy", "ppa1_0_500.npy"):
        (Path(config.paths.spectrograms_dir) / name).touch()
    historical_before = {
        path.relative_to(folds): path.read_bytes()
        for path in folds.rglob("*")
        if path.is_file()
    }

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
    paths = compare_inference_strategies.generate_overlapping_test_csvs(
        config,
        windows,
        str(folds),
        str(staging),
        "v3",
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
    provenance = json.loads((staging / "provenance.json").read_text())
    assert provenance["mapping_version"] == "v3"
    assert provenance["historical_fold_dir"] == str(folds.resolve())
    historical_after = {
        path.relative_to(folds): path.read_bytes()
        for path in folds.rglob("*")
        if path.is_file()
    }
    assert historical_after == historical_before
    assert all(not Path(path).is_relative_to(folds) for path in paths.values())


def test_rejects_conflicting_window_and_historical_fold_projects(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    folds = _fold_dirs(tmp_path, {"PPA1": [(10, "sound.wav")]})
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
    with pytest.raises(ValueError, match="Project mismatch"):
        compare_inference_strategies.generate_overlapping_test_csvs(
            config,
            windows,
            str(folds),
            str(tmp_path / "staging"),
            "v3",
        )


def test_annotations_are_disabled_by_default_without_opening_current_file(
    monkeypatch,
):
    args = compare_inference_strategies.build_arg_parser().parse_args([])
    assert args.annotations is None
    assert args.annotations_version is None
    assert args.cv_results == "outputs_v3/cv_results.csv"

    def fail_open(*_args, **_kwargs):
        raise AssertionError("annotations must not be opened implicitly")

    monkeypatch.setattr("builtins.open", fail_open)
    assert (
        compare_inference_strategies.load_annotation_metrics_input(
            args.annotations,
            args.annotations_version,
            "v3",
        )
        is None
    )


def test_annotations_must_explicitly_match_historical_mapping_version(tmp_path):
    annotations = tmp_path / "current_annotations.json"
    annotations.write_text('{"sounds": [], "annotations": []}')

    with pytest.raises(ValueError, match="must be 'v3'"):
        compare_inference_strategies.load_annotation_metrics_input(
            str(annotations),
            "v5",
            "v3",
        )

    loaded = compare_inference_strategies.load_annotation_metrics_input(
        str(annotations),
        "v3",
        "v3",
    )
    assert loaded == {"sounds": [], "annotations": []}


@pytest.mark.parametrize(
    ("fold_dir", "checkpoint_dir", "mismatched_artifact"),
    [
        ("data/folds_segmented_v5", "checkpoints_v3", "fold directory"),
        ("data/folds_segmented_v3", "checkpoints_v5", "checkpoint directory"),
    ],
)
def test_rejects_non_v3_folds_or_checkpoints(
    fold_dir,
    checkpoint_dir,
    mismatched_artifact,
):
    with pytest.raises(ValueError, match=mismatched_artifact):
        compare_inference_strategies.validate_historical_artifact_paths(
            fold_dir,
            checkpoint_dir,
            "v3",
        )


def test_second_run_without_annotations_removes_stale_boundary_output(tmp_path):
    output_dir = tmp_path / "comparison"
    first_results = pd.DataFrame(
        [
            {"resolution": "5s", "f1": 0.5},
            {"resolution": "1s", "f1": 0.4},
            {"resolution": "1s_interior", "f1": 0.6},
        ]
    )
    first_boundary = pd.DataFrame([{"distance": 1, "f1": 0.4}])
    first_summary = pd.DataFrame([{"resolution": "5s", "f1_mean": 0.5}])

    compare_inference_strategies.publish_comparison_tables(
        str(output_dir),
        first_results,
        first_boundary,
        first_summary,
    )
    boundary_path = output_dir / "comparison_boundary_sensitivity.csv"
    assert boundary_path.exists()

    second_results = pd.DataFrame([{"resolution": "5s", "f1": 0.7}])
    second_summary = pd.DataFrame([{"resolution": "5s", "f1_mean": 0.7}])
    compare_inference_strategies.publish_comparison_tables(
        str(output_dir),
        second_results,
        pd.DataFrame(),
        second_summary,
    )

    assert not boundary_path.exists()
    published = pd.read_csv(output_dir / "comparison_results.csv")
    assert published.to_dict("records") == [
        {"resolution": "5s", "f1": 0.7}
    ]


@pytest.mark.parametrize("path_kind", ["missing", "directory"])
def test_cv_results_reference_requires_existing_file(tmp_path, path_kind):
    cv_results = tmp_path / "cv_results.csv"
    if path_kind == "directory":
        cv_results.mkdir()

    with pytest.raises(FileNotFoundError, match="Historical CV results not found"):
        compare_inference_strategies.load_cv_results_reference(
            str(cv_results),
            len(compare_inference_strategies.PROJECTS),
        )


def test_cv_results_reference_requires_complete_fold_coverage(tmp_path):
    cv_results = tmp_path / "cv_results.csv"
    pd.DataFrame(
        [
            {"fold": fold, "f1": 0.5, "auprc": 0.6}
            for fold in range(len(compare_inference_strategies.PROJECTS) - 1)
        ]
    ).to_csv(cv_results, index=False)

    with pytest.raises(ValueError, match=r"incomplete fold coverage.*missing=\[4\]"):
        compare_inference_strategies.load_cv_results_reference(
            str(cv_results),
            len(compare_inference_strategies.PROJECTS),
        )


def _baseline_metrics(f1: float = 0.5, auprc: float = 0.6) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": fold,
                "strategy": "baseline",
                "resolution": "5s",
                "f1": f1,
                "auprc": auprc,
            }
            for fold in range(len(compare_inference_strategies.PROJECTS))
        ]
    )


def test_metric_mismatch_leaves_comparison_outputs_unpublished(tmp_path):
    output_dir = tmp_path / "comparison"
    results_5s = _baseline_metrics(f1=0.5)
    reference = _baseline_metrics(f1=0.9)[["fold", "f1", "auprc"]]

    with pytest.raises(ValueError, match="do not match historical CV results"):
        compare_inference_strategies.verify_and_publish_comparison_tables(
            results_5s,
            reference,
            str(output_dir),
            results_5s,
            pd.DataFrame(),
            pd.DataFrame([{"resolution": "5s"}]),
        )

    assert not output_dir.exists()


def test_valid_cv_reference_allows_comparison_publication(tmp_path):
    output_dir = tmp_path / "comparison"
    results_5s = _baseline_metrics()
    reference = results_5s[["fold", "f1", "auprc"]]
    summary = pd.DataFrame([{"resolution": "5s", "f1_mean": 0.5}])

    compare_inference_strategies.verify_and_publish_comparison_tables(
        results_5s,
        reference,
        str(output_dir),
        results_5s,
        pd.DataFrame(),
        summary,
    )

    assert (output_dir / "comparison_results.csv").is_file()
    assert (output_dir / "comparison_summary.csv").is_file()


def test_rejects_missing_project_identity(tmp_path, monkeypatch):
    config = _config(tmp_path)
    folds = _fold_dirs(tmp_path)
    fold_csv = (
        folds / "fold_0_MAP1_segmented" / "test_split.csv"
    )
    fold_csv.write_text(
        "sound_id,sound_filename,dataset,project\n10,sound.wav,,\n"
    )
    monkeypatch.setattr(
        compare_inference_strategies,
        "spectrogram_filename",
        lambda *_: "unused.npy",
    )
    windows = [
        {"window_id": 1, "sound_id": 10, "start": 0, "end": 500}
    ]
    with pytest.raises(ValueError, match="No project identity"):
        compare_inference_strategies.generate_overlapping_test_csvs(
            config,
            windows,
            str(folds),
            str(tmp_path / "staging"),
            "v3",
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
            str(tmp_path / "staging"),
            "v3",
        )


def test_rejects_unsupported_project_identity(tmp_path, monkeypatch):
    config = _config(tmp_path)
    folds = _fold_dirs(tmp_path)
    fold_csv = (
        folds / "fold_0_MAP1_segmented" / "test_split.csv"
    )
    fold_csv.write_text(
        "sound_id,sound_filename,dataset,project\n"
        "10,sound.wav,UNKNOWN,UNKNOWN\n"
    )
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
    with pytest.raises(ValueError, match="Unsupported project 'UNKNOWN'"):
        compare_inference_strategies.generate_overlapping_test_csvs(
            config,
            windows,
            str(folds),
            str(tmp_path / "staging"),
            "v3",
        )


def test_missing_spectrograms_abort_before_writing_evaluation_inputs(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    folds = _fold_dirs(tmp_path, {"MAP1": [(10, "sound.wav")]})
    staging = tmp_path / "staging"
    monkeypatch.setattr(
        compare_inference_strategies,
        "spectrogram_filename",
        lambda _file_name, start, _end: f"missing_{start}.npy",
    )
    windows = [
        {
            "window_id": 42,
            "dataset": "MAP1",
            "sound_id": 10,
            "start": 0,
            "end": 500,
        },
        {
            "window_id": 43,
            "dataset": "MAP1",
            "sound_id": 10,
            "start": 100,
            "end": 600,
        },
    ]

    with pytest.raises(
        FileNotFoundError,
        match=r"Missing 2 expected spectrogram files.*window 42.*window 43",
    ):
        compare_inference_strategies.generate_overlapping_test_csvs(
            config,
            windows,
            str(folds),
            str(staging),
            "v3",
        )

    assert not staging.exists()
