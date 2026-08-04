"""Analyze per-species average precision against train/validation/test support."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import average_precision_score


SUPPORT_COLUMNS = (
    "train_aug_windows",
    "train_canonical_windows",
    "train_aug_sound_ids",
    "train_canonical_sound_ids",
    "val_windows",
    "val_sound_ids",
    "test_windows",
    "test_sound_ids",
)
PRIMARY_SUPPORT_COLUMNS = ("train_canonical_windows", "val_windows", "test_windows")


def sound_support_bin_table(table: pd.DataFrame) -> pd.DataFrame:
    labels = ["1", "2-3", "4-9", "10+"]
    bins = [0, 1, 3, 9, np.inf]
    data = table.copy()
    data["test_sound_bin"] = pd.cut(data["test_sound_ids"], bins=bins, labels=labels)
    return (
        data.groupby("test_sound_bin", observed=True)["ap"]
        .agg(n_species="count", mean_ap="mean", median_ap="median", std_ap="std", min_ap="min", max_ap="max")
        .reset_index()
    )


def correlation_sensitivity_table(table: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "all": np.ones(len(table), dtype=bool),
        "test_pos_ge5": table["test_windows"].to_numpy() >= 5,
        "test_pos_ge10": table["test_windows"].to_numpy() >= 10,
        "test_positive_sound_ids_ge2": table["test_sound_ids"].to_numpy() >= 2,
    }
    predictors = (
        "train_aug_windows",
        "train_canonical_windows",
        "train_aug_sound_ids",
        "train_canonical_sound_ids",
        "test_windows",
    )
    rows = []
    for response in ("ap", "ap_lift"):
        for subset_name, mask in subsets.items():
            subset = table.loc[mask]
            for predictor in predictors:
                result = spearmanr(subset[predictor], subset[response])
                rows.append(
                    {
                        "response": response,
                        "subset": subset_name,
                        "n_species": len(subset),
                        "predictor": predictor,
                        "spearman_rho": float(result.statistic),
                        "spearman_p": float(result.pvalue),
                    }
                )
    result = pd.DataFrame(rows)
    result["spearman_q_bh"] = np.nan
    for response in result["response"].unique():
        mask = result["response"] == response
        result.loc[mask, "spearman_q_bh"] = benjamini_hochberg(
            result.loc[mask, "spearman_p"]
        )
    return result


def load_class_list(path: Path) -> pd.DataFrame:
    entries = json.loads(path.read_text())
    frame = pd.DataFrame(entries).sort_values("index")
    if frame["index"].tolist() != list(range(len(frame))):
        raise ValueError("class_list indices must be contiguous and ordered")
    if frame["code"].duplicated().any():
        raise ValueError("class_list contains duplicate codes")
    return frame


def support_from_split(path: Path, class_codes: Sequence[str]) -> tuple[dict[str, int], dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    sounds: dict[str, set[int]] = defaultdict(set)
    known = set(class_codes)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for code in filter(None, row["target_codes"].split(";")):
                if code not in known:
                    raise ValueError(f"{path}: unknown target code {code}")
                counts[code] += 1
                sounds[code].add(int(row["sound_id"]))
    return dict(counts), {code: len(values) for code, values in sounds.items()}


def recompute_ap(predictions_path: Path) -> np.ndarray:
    with np.load(predictions_path, allow_pickle=False) as arrays:
        probabilities = arrays["probabilities"]
        targets = arrays["target_vector"]
    if probabilities.shape != targets.shape:
        raise ValueError("probability and target shapes differ")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    values = []
    for index in range(targets.shape[1]):
        if not np.isfinite(probabilities[:, index]).all() or np.unique(targets[:, index]).size < 2:
            values.append(np.nan)
        else:
            values.append(average_precision_score(targets[:, index], probabilities[:, index]))
    return np.asarray(values)


def load_predictions(predictions_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(predictions_path, allow_pickle=False) as arrays:
        return (
            arrays["probabilities"].astype(np.float64),
            arrays["target_vector"].astype(np.uint8),
            arrays["sound_id"].astype(np.int64),
        )


def build_per_species_table(
    class_list_path: Path,
    species_ap_path: Path,
    predictions_path: Path,
    split_dir: Path,
) -> pd.DataFrame:
    classes = load_class_list(class_list_path)
    metrics = pd.read_csv(species_ap_path).sort_values("index")
    if metrics["code"].tolist() != classes["code"].tolist():
        raise ValueError("species_ap and class_list orders differ")

    recomputed = recompute_ap(predictions_path)
    if not np.allclose(metrics["ap"], recomputed, rtol=0.0, atol=1e-12, equal_nan=True):
        raise ValueError("species AP does not reproduce from test_predictions.npz")

    codes = classes["code"].tolist()
    support_specs = {
        "train_aug": "train_split.csv",
        "train_canonical": "canonical_train_split.csv",
        "val": "val_split.csv",
        "test": "test_split.csv",
    }
    support: dict[str, dict[str, int]] = {}
    for prefix, filename in support_specs.items():
        counts, sounds = support_from_split(split_dir / filename, codes)
        support[f"{prefix}_windows"] = counts
        support[f"{prefix}_sound_ids"] = sounds

    table = metrics[
        [
            "index",
            "code",
            "species",
            "ap",
            "test_prevalence",
            "train_pos",
            "val_pos",
            "test_pos",
        ]
    ].copy()
    for column, values in support.items():
        table[column] = table["code"].map(values).fillna(0).astype(int)

    table["ap_lift"] = table["ap"] - table["test_prevalence"]
    table["train_augmentation_factor"] = (
        table["train_aug_windows"] / table["train_canonical_windows"].replace(0, np.nan)
    )

    checks = {
        "train_pos": "train_aug_windows",
        "val_pos": "val_windows",
        "test_pos": "test_windows",
    }
    for metric_column, derived_column in checks.items():
        if not np.array_equal(table[metric_column].to_numpy(), table[derived_column].to_numpy()):
            raise ValueError(f"{metric_column} does not match {derived_column}")
    return table


def cluster_bootstrap_ap(
    probabilities: np.ndarray,
    targets: np.ndarray,
    sound_ids: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    unique_sounds = np.unique(sound_ids)
    group_indices = [np.flatnonzero(sound_ids == sound_id) for sound_id in unique_sounds]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(group_indices), size=(n_bootstrap, len(group_indices)))
    rows = []
    for species_index in range(targets.shape[1]):
        positive_sound_ids = len(np.unique(sound_ids[targets[:, species_index] == 1]))
        values: list[float] = []
        if positive_sound_ids >= 2:
            for draw in draws:
                indices = np.concatenate([group_indices[group_index] for group_index in draw])
                y_true = targets[indices, species_index]
                if y_true.min() == y_true.max():
                    continue
                values.append(
                    float(average_precision_score(y_true, probabilities[indices, species_index]))
                )
        low, high = (np.percentile(values, [2.5, 97.5]) if values else (np.nan, np.nan))
        rows.append(
            {
                "index": species_index,
                "test_positive_sound_ids": positive_sound_ids,
                "ap_ci_estimable": positive_sound_ids >= 2 and len(values) >= n_bootstrap // 2,
                "ap_ci_low": float(low),
                "ap_ci_high": float(high),
                "ap_ci_valid_bootstraps": len(values),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(x)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        ranked_x = rankdata(x[indices])
        ranked_y = rankdata(y[indices])
        centered_x = ranked_x - ranked_x.mean()
        centered_y = ranked_y - ranked_y.mean()
        denominator = np.sqrt(np.sum(centered_x**2) * np.sum(centered_y**2))
        value = np.sum(centered_x * centered_y) / denominator if denominator else np.nan
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    return tuple(np.percentile(values, [2.5, 97.5]).tolist())


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    n = len(values)
    for position in range(n - 1, -1, -1):
        running = min(running, ranked[position] * n / (position + 1))
        adjusted[position] = running
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def correlation_table(
    table: pd.DataFrame,
    *,
    seed: int,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    for response in ("ap", "ap_lift"):
        for index, predictor in enumerate(SUPPORT_COLUMNS):
            x = table[predictor].to_numpy(dtype=float)
            y = table[response].to_numpy(dtype=float)
            spearman = spearmanr(x, y)
            pearson = pearsonr(np.log1p(x), y)
            low, high = bootstrap_spearman(
                x, y, seed=seed + index + (100 if response == "ap_lift" else 0), n_bootstrap=n_bootstrap
            )
            rows.append(
                {
                    "response": response,
                    "predictor": predictor,
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "spearman_ci_low": low,
                    "spearman_ci_high": high,
                    "pearson_log1p_r": float(pearson.statistic),
                    "pearson_log1p_p": float(pearson.pvalue),
                }
            )
    result = pd.DataFrame(rows)
    result["spearman_q_bh"] = benjamini_hochberg(result["spearman_p"])
    return result


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: Sequence[np.ndarray]) -> tuple[float, float]:
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    design = np.column_stack([np.ones(len(x)), *[rankdata(control) for control in controls]])
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    result = pearsonr(residual_x, residual_y)
    return float(result.statistic), float(result.pvalue)


def partial_correlation_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictors = (
        "train_aug_windows",
        "train_canonical_windows",
        "train_aug_sound_ids",
        "train_canonical_sound_ids",
        "val_windows",
    )
    for response in ("ap", "ap_lift"):
        for predictor in predictors:
            rho_test, p_test = partial_spearman(
                table[predictor].to_numpy(),
                table[response].to_numpy(),
                [table["test_windows"].to_numpy()],
            )
            controls = [table["test_windows"].to_numpy()]
            if predictor != "val_windows":
                controls.append(table["val_windows"].to_numpy())
            rho_all, p_all = partial_spearman(
                table[predictor].to_numpy(), table[response].to_numpy(), controls
            )
            rows.append(
                {
                    "response": response,
                    "predictor": predictor,
                    "controls_test_rho": rho_test,
                    "controls_test_p": p_test,
                    "controls_val_test_rho": rho_all,
                    "controls_val_test_p": p_all,
                }
            )
    result = pd.DataFrame(rows)
    result["controls_test_q_bh"] = np.nan
    result["controls_val_test_q_bh"] = np.nan
    for response in result["response"].unique():
        mask = result["response"] == response
        result.loc[mask, "controls_test_q_bh"] = benjamini_hochberg(
            result.loc[mask, "controls_test_p"]
        )
        result.loc[mask, "controls_val_test_q_bh"] = benjamini_hochberg(
            result.loc[mask, "controls_val_test_p"]
        )
    return result


def support_bin_table(table: pd.DataFrame) -> pd.DataFrame:
    labels = ["1-2", "3-4", "5-9", "10-19", "20+"]
    bins = [0, 2, 4, 9, 19, np.inf]
    data = table.copy()
    data["test_support_bin"] = pd.cut(data["test_windows"], bins=bins, labels=labels)
    return (
        data.groupby("test_support_bin", observed=True)["ap"]
        .agg(n_species="count", mean_ap="mean", median_ap="median", std_ap="std", min_ap="min", max_ap="max")
        .reset_index()
    )


def plot_support_scatter(table: pd.DataFrame, correlations: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    labels = {
        "train_canonical_windows": "Canonical train positives",
        "val_windows": "Validation positives",
        "test_windows": "Test positives",
    }
    for axis, predictor in zip(axes, PRIMARY_SUPPORT_COLUMNS):
        axis.scatter(table[predictor], table["ap"], alpha=0.75, s=35, edgecolor="none")
        axis.set_xscale("log")
        axis.set_xlabel(labels[predictor])
        axis.set_ylabel("Per-species AP")
        row = correlations[(correlations["response"] == "ap") & (correlations["predictor"] == predictor)].iloc[0]
        axis.set_title(
            f"Spearman rho={row.spearman_rho:.2f}\n95% CI [{row.spearman_ci_low:.2f}, {row.spearman_ci_high:.2f}]"
        )
        for _, item in table.nlargest(2, "ap").iterrows():
            axis.annotate(item["code"], (item[predictor], item["ap"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_ap_by_species(table: pd.DataFrame, output: Path) -> None:
    ordered = table.sort_values("ap")
    colors = plt.cm.viridis(
        (np.log1p(ordered["test_windows"]) - np.log1p(ordered["test_windows"]).min())
        / max(np.ptp(np.log1p(ordered["test_windows"])), 1e-12)
    )
    fig, axis = plt.subplots(figsize=(9, 17), constrained_layout=True)
    axis.barh(ordered["code"], ordered["ap"], color=colors)
    estimable = ordered["ap_ci_estimable"].to_numpy(dtype=bool)
    if estimable.any():
        xerr = np.vstack(
            [
                ordered.loc[estimable, "ap"] - ordered.loc[estimable, "ap_ci_low"],
                ordered.loc[estimable, "ap_ci_high"] - ordered.loc[estimable, "ap"],
            ]
        )
        axis.errorbar(
            ordered.loc[estimable, "ap"],
            np.flatnonzero(estimable),
            xerr=xerr,
            fmt="none",
            ecolor="black",
            elinewidth=0.6,
            alpha=0.55,
        )
    axis.set_xlabel("Average precision")
    axis.set_ylabel("Species code")
    axis.set_title("Per-species AP (color indicates log test-positive support)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_support_bins(table: pd.DataFrame, output: Path) -> None:
    labels = ["1-2", "3-4", "5-9", "10-19", "20+"]
    bins = [0, 2, 4, 9, 19, np.inf]
    groups = [
        table.loc[pd.cut(table["test_windows"], bins=bins, labels=labels) == label, "ap"].to_numpy()
        for label in labels
    ]
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.boxplot(groups, tick_labels=labels, showmeans=True)
    axis.set_xlabel("Positive test windows per species")
    axis.set_ylabel("Per-species AP")
    axis.set_title("AP uncertainty and performance by test support")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 3) -> str:
    selected = frame[list(columns)].copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        finite = selected[column].dropna()
        integer_like = not finite.empty and np.allclose(finite, np.round(finite))
        if integer_like:
            selected[column] = selected[column].map(
                lambda value: "NA" if pd.isna(value) else str(int(round(value)))
            )
        else:
            selected[column] = selected[column].map(
                lambda value: "NA" if pd.isna(value) else f"{value:.{digits}f}"
            )
    headers = [str(column) for column in selected.columns]
    rows = [[str(value) for value in row] for row in selected.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    path: Path,
    table: pd.DataFrame,
    correlations: pd.DataFrame,
    partial: pd.DataFrame,
    bins: pd.DataFrame,
    sound_bins: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    n_species = len(table)
    ap_sensitivity = sensitivity[sensitivity["response"] == "ap"]
    ap_partial = partial[partial["response"] == "ap"]
    primary = correlations[
        (correlations["response"] == "ap") & correlations["predictor"].isin(PRIMARY_SUPPORT_COLUMNS)
    ]
    top = table.nlargest(10, "ap")
    bottom = table.nsmallest(10, "ap")
    lines = [
        "# Per-Species AP and Data Representation Analysis",
        "",
        "## Why",
        "",
        "This analysis asks whether species with more representation in the actual training, validation,",
        f"and test data obtain higher average precision (AP). Correlations are across the {n_species} species,",
        "not across windows, and therefore describe association rather than causation.",
        "",
        "![AP versus train/validation/test support](figures/ap_vs_support.png)",
        "",
        "## Main Result",
        "",
        f"Per-species AP increases moderately with representation in the unfiltered {n_species}-species analysis.",
        "The strongest unadjusted association is with canonical train positives, while test-positive support",
        "also has a strong association because thin test sets produce high-variance AP estimates. Training and",
        "test supports are themselves highly correlated by the stratified split, so their effects cannot be",
        "cleanly separated observationally.",
        "",
        markdown_table(
            primary,
            [
                "predictor",
                "spearman_rho",
                "spearman_ci_low",
                "spearman_ci_high",
                "spearman_q_bh",
                "pearson_log1p_r",
            ],
        ),
        "",
        "After controlling for test-positive support, the partial rank association between canonical train",
        f"positives and AP is {ap_partial.loc[ap_partial.predictor == 'train_canonical_windows', 'controls_test_rho'].iloc[0]:.3f} "
        f"(BH-adjusted q={ap_partial.loc[ap_partial.predictor == 'train_canonical_windows', 'controls_test_q_bh'].iloc[0]:.3g}). "
        "The distinct-train-audio association largely disappears after this control, indicating that window",
        "support and split-wide species commonness are more closely associated with the observed pattern than",
        "audio-file count alone.",
        "",
        "The association weakens when thin-test species are removed. For canonical train windows, Spearman",
        f"rho changes from {ap_sensitivity[(ap_sensitivity.subset == 'all') & (ap_sensitivity.predictor == 'train_canonical_windows')].spearman_rho.iloc[0]:.3f} "
        f"({n_species} species) to {ap_sensitivity[(ap_sensitivity.subset == 'test_pos_ge5') & (ap_sensitivity.predictor == 'train_canonical_windows')].spearman_rho.iloc[0]:.3f} "
        f"for `test_pos>=5` and {ap_sensitivity[(ap_sensitivity.subset == 'test_pos_ge10') & (ap_sensitivity.predictor == 'train_canonical_windows')].spearman_rho.iloc[0]:.3f} "
        "for `test_pos>=10`. Thus, the raw correlation should not be read as a stable dose-response relationship.",
        "",
        "## Test-Support Strata",
        "",
        markdown_table(
            bins,
            ["test_support_bin", "n_species", "mean_ap", "median_ap", "std_ap", "min_ap", "max_ap"],
        ),
        "",
        "Species with only one or two positive test windows have highly unstable AP: their mean is lower, but",
        "the range includes both near-zero values and AP=1.0. The support-restricted macro-AP values in the",
        "main results (>=5 and >=10 positives) are therefore more stable summaries than the all-species value.",
        "",
        "Distinct positive audio files give a more independent support view than window counts:",
        "",
        markdown_table(
            sound_bins,
            ["test_sound_bin", "n_species", "mean_ap", "median_ap", "std_ap", "min_ap", "max_ap"],
        ),
        "",
        "Ten species have positives from only one test audio file. Their AP confidence interval is marked",
        "non-estimable because resampling cannot create independent positive evidence that does not exist.",
        "",
        "![AP by positive test support](figures/ap_by_test_support_bin.png)",
        "",
        "## Highest AP Species",
        "",
        "![Per-species AP with cluster-bootstrap intervals where estimable](figures/ap_by_species.png)",
        "",
        markdown_table(
            top,
            ["code", "species", "ap", "ap_ci_low", "ap_ci_high", "test_positive_sound_ids", "train_canonical_windows", "train_canonical_sound_ids", "test_windows"],
        ),
        "",
        "## Lowest AP Species",
        "",
        markdown_table(
            bottom,
            ["code", "species", "ap", "ap_ci_low", "ap_ci_high", "test_positive_sound_ids", "train_canonical_windows", "train_canonical_sound_ids", "test_windows"],
        ),
        "",
        "## Interpretation",
        "",
        "- More represented species generally perform better in the full descriptive analysis, supporting",
        "  annotation quantity as one constraint, but the relationship weakens in better-supported subsets.",
        "- Canonical train support correlates slightly more strongly with AP than augmented train support;",
        "  overlapping augmentation increases volume but not independent evidence.",
        f"- Train augmentation inflates positive-window support by {table.train_augmentation_factor.min():.2f}x "
        f"to {table.train_augmentation_factor.max():.2f}x across species (mean {table.train_augmentation_factor.mean():.2f}x).",
        "- Test support affects both metric stability and observed correlation. It must not be interpreted as",
        "  a causal improvement in the trained classifier.",
        "- Validation support is associated with AP, but the validation set only selected one global C; it did",
        "  not tune a separate classifier per species.",
        "- AP lift above prevalence gives nearly the same correlations, so the result is not explained only by",
        "  AP's test-metric prevalence floor. It does not remove broader species-commonness confounding.",
        "- Distinct train audio files have little residual association after controlling test support; species",
        "  commonness and window volume remain more strongly associated than recording count alone.",
        "- In the `test_pos>=10` subset, the train-audio-file correlation becomes weakly negative and",
        "  non-significant, reinforcing that the full-data association is not a stable dose-response.",
        "",
        "## Limitations",
        "",
        "1. All support variables are strongly correlated because the split was stratified by species.",
        f"2. The unit of correlation is species (n={n_species}); bootstrap intervals resample species and do not capture",
        "   uncertainty from re-splitting or retraining.",
        "3. AP for thin-support species is discrete and unstable. A single positive can produce AP=1.0.",
        "4. Window counts are not independent biological events; canonical windows and distinct sound IDs are",
        "   reported to expose this distinction.",
        "5. This analysis is descriptive and cannot establish that adding a specific number of annotations will",
        "   cause a corresponding AP increase.",
        "",
        "## Reproducible Artifacts",
        "",
        "- `analysis/ap_support_per_species.csv`",
        "- `analysis/ap_support_correlations.csv`",
        "- `analysis/ap_support_partial_correlations.csv`",
        "- `analysis/ap_by_test_support_bin.csv`",
        "- `analysis/ap_by_test_sound_bin.csv`",
        "- `analysis/ap_support_correlation_sensitivity.csv`",
        "- `figures/ap_vs_support.png`",
        "- `figures/ap_by_species.png`",
        "- `figures/ap_by_test_support_bin.png`",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", default="data/splits_species_v1")
    parser.add_argument("--results-dir", default="checkpoints/perch/species_v1")
    parser.add_argument("--output-dir", default="docs/implementation/species-linear-probe-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--cluster-bootstrap", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    analysis_dir = output_dir / "analysis"
    figure_dir = output_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    table = build_per_species_table(
        split_dir / "class_list.json",
        results_dir / "species_ap.csv",
        results_dir / "test_predictions.npz",
        split_dir,
    )
    probabilities, targets, sound_ids = load_predictions(results_dir / "test_predictions.npz")
    uncertainty = cluster_bootstrap_ap(
        probabilities,
        targets,
        sound_ids,
        seed=args.seed,
        n_bootstrap=args.cluster_bootstrap,
    )
    table = table.merge(uncertainty, on="index", validate="one_to_one")
    correlations = correlation_table(table, seed=args.seed, n_bootstrap=args.bootstrap)
    partial = partial_correlation_table(table)
    bins = support_bin_table(table)
    sound_bins = sound_support_bin_table(table)
    sensitivity = correlation_sensitivity_table(table)

    table.to_csv(analysis_dir / "ap_support_per_species.csv", index=False)
    correlations.to_csv(analysis_dir / "ap_support_correlations.csv", index=False)
    partial.to_csv(analysis_dir / "ap_support_partial_correlations.csv", index=False)
    bins.to_csv(analysis_dir / "ap_by_test_support_bin.csv", index=False)
    sound_bins.to_csv(analysis_dir / "ap_by_test_sound_bin.csv", index=False)
    sensitivity.to_csv(analysis_dir / "ap_support_correlation_sensitivity.csv", index=False)

    plot_support_scatter(table, correlations, figure_dir / "ap_vs_support.png")
    plot_ap_by_species(table, figure_dir / "ap_by_species.png")
    plot_support_bins(table, figure_dir / "ap_by_test_support_bin.png")
    write_report(
        output_dir / "ap_support_analysis.md",
        table,
        correlations,
        partial,
        bins,
        sound_bins,
        sensitivity,
    )
    print(f"Wrote AP/support analysis to {output_dir}")


if __name__ == "__main__":
    main()
