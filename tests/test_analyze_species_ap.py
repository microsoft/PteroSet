import sys
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_species_ap as analysis


def test_benjamini_hochberg_is_monotonic_in_rank():
    adjusted = analysis.benjamini_hochberg([0.01, 0.03, 0.02, 0.8])
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    assert adjusted[0] <= adjusted[2] <= adjusted[1] <= adjusted[3]


def test_benjamini_hochberg_matches_known_values():
    adjusted = analysis.benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008])


def test_partial_spearman_controls_shared_rank_signal():
    rng = np.random.default_rng(4)
    z = np.arange(200)
    x = z + rng.normal(0, 8, len(z))
    y = z + rng.normal(0, 8, len(z))
    raw = analysis.spearmanr(x, y).statistic
    partial, _ = analysis.partial_spearman(x, y, [z])
    assert raw > 0.9
    assert abs(partial) < 0.2


def test_support_bins_cover_species_once():
    table = pd.DataFrame({"test_windows": [1, 2, 3, 5, 10, 20], "ap": np.linspace(0, 1, 6)})
    summary = analysis.support_bin_table(table)
    assert summary["n_species"].sum() == len(table)


def test_bootstrap_spearman_reproducible():
    x = np.arange(20)
    y = x + np.sin(x)
    first = analysis.bootstrap_spearman(x, y, seed=42, n_bootstrap=100)
    second = analysis.bootstrap_spearman(x, y, seed=42, n_bootstrap=100)
    assert first == second


def test_correlation_table_contains_ap_and_lift():
    rng = np.random.default_rng(0)
    table = pd.DataFrame({column: rng.integers(1, 50, 20) for column in analysis.SUPPORT_COLUMNS})
    table["ap"] = rng.uniform(0, 1, 20)
    table["ap_lift"] = table["ap"] - 0.001
    result = analysis.correlation_table(table, seed=1, n_bootstrap=50)
    assert set(result["response"]) == {"ap", "ap_lift"}
    assert len(result) == 2 * len(analysis.SUPPORT_COLUMNS)


def test_cluster_bootstrap_marks_single_positive_sound_non_estimable():
    probabilities = np.array([[0.9], [0.1], [0.2], [0.3]])
    targets = np.array([[1], [0], [0], [0]], dtype=np.uint8)
    sound_ids = np.array([1, 1, 2, 3])
    result = analysis.cluster_bootstrap_ap(
        probabilities, targets, sound_ids, seed=42, n_bootstrap=50
    )
    assert result.loc[0, "test_positive_sound_ids"] == 1
    assert not result.loc[0, "ap_ci_estimable"]
    assert np.isnan(result.loc[0, "ap_ci_low"])
    assert "response" not in result.columns
    assert "spearman_q_bh" not in result.columns


def test_cluster_bootstrap_computes_interval_for_multiple_positive_sounds():
    probabilities = np.array([[0.9], [0.8], [0.1], [0.2], [0.7], [0.3]])
    targets = np.array([[1], [1], [0], [0], [1], [0]], dtype=np.uint8)
    sound_ids = np.array([1, 1, 2, 2, 3, 4])
    result = analysis.cluster_bootstrap_ap(
        probabilities, targets, sound_ids, seed=42, n_bootstrap=100
    )
    assert result.loc[0, "test_positive_sound_ids"] == 2
    assert result.loc[0, "ap_ci_estimable"]
    assert 0 <= result.loc[0, "ap_ci_low"] <= result.loc[0, "ap_ci_high"] <= 1


def test_correlation_sensitivity_subsets_are_reported():
    rng = np.random.default_rng(2)
    table = pd.DataFrame(
        {
            "ap": rng.uniform(0, 1, 20),
            "test_windows": np.arange(1, 21),
            "test_sound_ids": np.maximum(1, np.arange(1, 21) // 2),
            "train_aug_windows": np.arange(10, 30),
            "train_canonical_windows": np.arange(5, 25),
            "train_aug_sound_ids": np.arange(1, 21),
            "train_canonical_sound_ids": np.arange(1, 21),
        }
    )
    table["ap_lift"] = table["ap"] - 0.001
    result = analysis.correlation_sensitivity_table(table)
    assert set(result["subset"]) == {
        "all",
        "test_pos_ge5",
        "test_pos_ge10",
        "test_positive_sound_ids_ge2",
    }
    assert "spearman_q_bh" in result.columns
    for response in ("ap", "ap_lift"):
        response_rows = result["response"] == response
        expected = analysis.benjamini_hochberg(result.loc[response_rows, "spearman_p"])
        np.testing.assert_allclose(result.loc[response_rows, "spearman_q_bh"], expected)


def test_partial_correlation_table_adjusts_each_p_value_family_by_response():
    rng = np.random.default_rng(3)
    table = pd.DataFrame(
        {
            "ap": rng.uniform(0, 1, 30),
            "test_windows": rng.integers(1, 30, 30),
            "val_windows": rng.integers(1, 30, 30),
            "train_aug_windows": rng.integers(1, 100, 30),
            "train_canonical_windows": rng.integers(1, 50, 30),
            "train_aug_sound_ids": rng.integers(1, 30, 30),
            "train_canonical_sound_ids": rng.integers(1, 20, 30),
        }
    )
    table["ap_lift"] = table["ap"] - 0.001
    result = analysis.partial_correlation_table(table)
    for response in ("ap", "ap_lift"):
        response_rows = result["response"] == response
        expected_test = analysis.benjamini_hochberg(
            result.loc[response_rows, "controls_test_p"]
        )
        expected_val_test = analysis.benjamini_hochberg(
            result.loc[response_rows, "controls_val_test_p"]
        )
        np.testing.assert_allclose(
            result.loc[response_rows, "controls_test_q_bh"], expected_test
        )
        np.testing.assert_allclose(
            result.loc[response_rows, "controls_val_test_q_bh"], expected_val_test
        )


def test_sound_support_bins_cover_species_once():
    table = pd.DataFrame({"test_sound_ids": [1, 2, 3, 4, 9, 10], "ap": np.linspace(0, 1, 6)})
    summary = analysis.sound_support_bin_table(table)
    assert summary["n_species"].sum() == len(table)


def _write_support_fixture(tmp_path: Path):
    codes = ["A", "B"]
    class_list = [{"index": i, "code": code, "species": f"Species {code}"} for i, code in enumerate(codes)]
    class_list_path = tmp_path / "class_list.json"
    class_list_path.write_text(json.dumps(class_list))

    target = np.array([[1, 0], [0, 1], [0, 0], [1, 0]], dtype=np.uint8)
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.2], [0.7, 0.3]])
    predictions_path = tmp_path / "predictions.npz"
    np.savez(
        predictions_path,
        probabilities=probabilities,
        target_vector=target,
        sound_id=np.array([1, 2, 3, 1]),
    )
    aps = [analysis.average_precision_score(target[:, i], probabilities[:, i]) for i in range(2)]
    species_ap = pd.DataFrame(
        {
            "index": [0, 1],
            "code": codes,
            "species": ["Species A", "Species B"],
            "ap": aps,
            "test_prevalence": target.mean(axis=0),
            "prevalence_baseline_ap": target.mean(axis=0),
            "train_pos": [4, 2],
            "val_pos": [1, 1],
            "test_pos": target.sum(axis=0),
        }
    )
    species_ap_path = tmp_path / "species_ap.csv"
    species_ap.to_csv(species_ap_path, index=False)

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    fields = ["sound_id", "target_codes"]
    rows_by_file = {
        "train_split.csv": [(1, "A"), (1, "A"), (2, "A;B"), (3, "B"), (6, "A")],
        "canonical_train_split.csv": [(1, "A"), (2, "A;B"), (3, "B")],
        "val_split.csv": [(4, "A"), (5, "B")],
        "test_split.csv": [(1, "A"), (2, "B"), (3, ""), (1, "A")],
    }
    for filename, rows in rows_by_file.items():
        with (split_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sound_id, target_codes in rows:
                writer.writerow({"sound_id": sound_id, "target_codes": target_codes})
    return class_list_path, species_ap_path, predictions_path, split_dir


def test_build_per_species_table_validates_and_joins_support(tmp_path):
    paths = _write_support_fixture(tmp_path)
    table = analysis.build_per_species_table(*paths)
    assert table["code"].tolist() == ["A", "B"]
    assert table["train_aug_windows"].tolist() == [4, 2]
    assert table["train_canonical_windows"].tolist() == [2, 2]
    assert table["train_aug_sound_ids"].tolist() == [3, 2]
    assert table["train_canonical_sound_ids"].tolist() == [2, 2]
    assert table["test_windows"].tolist() == [2, 1]


def test_build_per_species_table_rejects_ap_mismatch(tmp_path):
    class_list, species_ap, predictions, split_dir = _write_support_fixture(tmp_path)
    frame = pd.read_csv(species_ap)
    frame.loc[0, "ap"] = 0.0
    frame.to_csv(species_ap, index=False)
    with pytest.raises(ValueError, match="does not reproduce"):
        analysis.build_per_species_table(class_list, species_ap, predictions, split_dir)


def test_support_from_split_rejects_unknown_code(tmp_path):
    path = tmp_path / "split.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sound_id", "target_codes"])
        writer.writeheader()
        writer.writerow({"sound_id": 1, "target_codes": "UNKNOWN"})
    with pytest.raises(ValueError, match="unknown target code"):
        analysis.support_from_split(path, ["A"])
