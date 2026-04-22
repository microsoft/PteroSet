"""
Compute window-level statistics for v3 cross-validation folds.

This script does not load any model. It reads the fold CSVs and the
windows-mapping / annotations JSONs to produce the tables that inform
the Technical Validation section of the paper.

Tier 1 (always produced):
  - Positive / negative window counts per fold and split (train/val/test)
  - Positive rate and total hours per split
  - Confusion-matrix counts (TP/FP/TN/FN) from test_split_with_predictions.csv

Tier 2 (always produced):
  - Best-F1 threshold sweep per fold (from probability column)
  - Annotation density per positive test window (mean/median/max overlaps)
  - Dataset-wide and per-project positive rates (from the windows-mapping JSON;
    these are the canonical rates over *all* windows in the dataset, whereas
    the fold-CSV rates reflect per-fold subsampling)

Outputs (written to --output_dir, default ``reports/``):
  window_counts_by_fold.csv          long-format per-split stats
  confusion_counts_by_fold.csv       TP/FP/TN/FN from test predictions
  threshold_stats_by_fold.csv        best-F1 threshold sweep per fold
  annotation_density_by_fold.csv     annotations per positive test window
  project_positive_rates.csv         canonical pos rate per project + dataset total
  window_counts_by_fold.md           paper-ready markdown summary

Usage:
    python plot_cv_results_window_stats.py \
        --fold_dir data/folds_segmented_v3 \
        --windows_mapping data/windows_mapping_4.0overlap_segmented_v3.json \
        --annotations data/annotations_identification.json \
        --output_dir reports
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

WINDOW_DURATION_S = 5.0
FOLD_DIR_RE = re.compile(r"^fold_(\d+)_([A-Za-z0-9]+)_segmented$")


def discover_folds(fold_dir: Path) -> List[Tuple[int, str, Path]]:
    """Return sorted [(fold_num, project, path), ...] for all folds in fold_dir."""
    folds = []
    for entry in sorted(os.listdir(fold_dir)):
        match = FOLD_DIR_RE.match(entry)
        if not match:
            continue
        fold_num = int(match.group(1))
        project = match.group(2)
        folds.append((fold_num, project, fold_dir / entry))
    if not folds:
        raise RuntimeError(f"No fold_*_*_segmented directories found under {fold_dir}")
    return folds


def split_stats(csv_path: Path) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    n_total = len(df)
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    pos_rate = n_pos / n_total if n_total else float("nan")
    hours = n_total * WINDOW_DURATION_S / 3600.0
    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": pos_rate,
        "hours": hours,
    }


def confusion_counts(pred_csv_path: Path) -> Dict[str, int]:
    df = pd.read_csv(pred_csv_path)
    counts = df["prediction_type"].value_counts().to_dict()
    return {k: int(counts.get(k, 0)) for k in ("TP", "FP", "TN", "FN")}


def best_f1_stats(pred_csv_path: Path) -> Dict[str, float]:
    """Sweep thresholds on probability column; return the best-F1 operating point."""
    df = pd.read_csv(pred_csv_path)
    y = df["label"].to_numpy()
    probs = df["probability"].to_numpy()
    if len(np.unique(y)) < 2:
        return {
            "best_f1": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
            "best_threshold": float("nan"),
        }
    precision, recall, thresholds = precision_recall_curve(y, probs)
    # precision_recall_curve returns arrays with one extra leading entry
    # (precision=1, recall=0); align with thresholds by dropping the last point
    prec = precision[:-1]
    rec = recall[:-1]
    denom = prec + rec
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2.0 * prec * rec / np.where(denom > 0, denom, 1.0), 0.0)
    if f1.size == 0:
        return {
            "best_f1": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
            "best_threshold": float("nan"),
        }
    best_idx = int(np.argmax(f1))
    return {
        "best_f1": float(f1[best_idx]),
        "best_precision": float(prec[best_idx]),
        "best_recall": float(rec[best_idx]),
        "best_threshold": float(thresholds[best_idx]),
    }


def project_positive_rates(windows_mapping_path: Path) -> pd.DataFrame:
    """Per-project and dataset-wide positive rates from the v3 windows mapping.

    Returns a DataFrame with one row per project plus a trailing 'ALL' row
    representing the whole dataset.
    """
    with open(windows_mapping_path) as f:
        windows = json.load(f)
    df = pd.DataFrame(windows)
    by_proj = (
        df.groupby("dataset")
        .agg(n_total=("label", "size"), n_pos=("label", "sum"))
        .reset_index()
        .rename(columns={"dataset": "project"})
        .sort_values("project")
    )
    by_proj["n_neg"] = by_proj["n_total"] - by_proj["n_pos"]
    by_proj["pos_rate"] = by_proj["n_pos"] / by_proj["n_total"]
    overall = pd.DataFrame(
        [
            {
                "project": "ALL",
                "n_total": int(df.shape[0]),
                "n_pos": int(df["label"].sum()),
                "n_neg": int(df.shape[0] - df["label"].sum()),
                "pos_rate": float(df["label"].mean()),
            }
        ]
    )
    return pd.concat([by_proj, overall], ignore_index=True)


def build_annotation_index(annotations_path: Path) -> Dict[int, List[dict]]:
    """Index annotations by sound_id for fast overlap lookup."""
    with open(annotations_path) as f:
        data = json.load(f)
    annotations = data.get("annotations", [])
    index: Dict[int, List[dict]] = defaultdict(list)
    for ann in annotations:
        sid = int(ann["sound_id"])
        index[sid].append(
            {
                "t_min": float(ann["t_min"]),
                "t_max": float(ann["t_max"]),
            }
        )
    return index


def annotation_density_for_fold(
    test_csv: Path, ann_index: Dict[int, List[dict]]
) -> Dict[str, float]:
    """Compute overlap density (annotations per positive window) for test-set positives."""
    df = pd.read_csv(test_csv)
    pos = df[df["label"] == 1]
    densities = []
    for _, row in pos.iterrows():
        sr = int(row["sample_rate"])
        t0 = int(row["start"]) / sr
        t1 = int(row["end"]) / sr
        sid = int(row["sound_id"])
        anns = ann_index.get(sid, [])
        overlaps = sum(1 for a in anns if a["t_max"] > t0 and a["t_min"] < t1)
        densities.append(overlaps)
    if not densities:
        return {
            "n_pos_test": 0,
            "mean_ann_per_pos": float("nan"),
            "median_ann_per_pos": float("nan"),
            "max_ann_per_pos": float("nan"),
            "pos_windows_with_zero_overlap": 0,
        }
    densities_arr = np.asarray(densities)
    return {
        "n_pos_test": int(len(densities)),
        "mean_ann_per_pos": float(densities_arr.mean()),
        "median_ann_per_pos": float(np.median(densities_arr)),
        "max_ann_per_pos": int(densities_arr.max()),
        "pos_windows_with_zero_overlap": int((densities_arr == 0).sum()),
    }


def write_markdown_summary(
    counts_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    density_df: pd.DataFrame,
    project_rates_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a paper-ready markdown summary."""
    lines: List[str] = []
    lines.append("# Cross-validation window statistics (v3)")
    lines.append("")

    lines.append("## Positive rate per project and dataset-wide")
    lines.append("")
    lines.append("| Project | # windows | # pos | # neg | Positive rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for _, row in project_rates_df.iterrows():
        lines.append(
            f"| {row['project']} | {int(row['n_total'])} | "
            f"{int(row['n_pos'])} | {int(row['n_neg'])} | "
            f"{row['pos_rate']:.3f} |"
        )
    lines.append("")

    lines.append("## Window counts per fold")
    lines.append("")
    lines.append(
        "| Fold | Project | Split | # windows | # pos | # neg | Positive rate | Hours |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in counts_df.iterrows():
        lines.append(
            f"| {row['fold']} | {row['project']} | {row['split']} | "
            f"{int(row['n_total'])} | {int(row['n_pos'])} | {int(row['n_neg'])} | "
            f"{row['pos_rate']:.3f} | {row['hours']:.2f} |"
        )
    lines.append("")

    lines.append("## Confusion-matrix counts on test set (at saved threshold)")
    lines.append("")
    lines.append("| Fold | Project | TP | FP | TN | FN |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in confusion_df.iterrows():
        lines.append(
            f"| {row['fold']} | {row['project']} | "
            f"{int(row['TP'])} | {int(row['FP'])} | "
            f"{int(row['TN'])} | {int(row['FN'])} |"
        )
    lines.append("")

    lines.append("## Best-F1 operating point per fold (threshold sweep)")
    lines.append("")
    lines.append(
        "| Fold | Project | Best F1 | Precision @ best F1 | Recall @ best F1 | Threshold |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in threshold_df.iterrows():
        lines.append(
            f"| {row['fold']} | {row['project']} | "
            f"{row['best_f1']:.3f} | {row['best_precision']:.3f} | "
            f"{row['best_recall']:.3f} | {row['best_threshold']:.3f} |"
        )
    lines.append("")

    lines.append("## Annotation density per positive test window")
    lines.append("")
    lines.append(
        "| Fold | Project | # pos windows | Mean anns / pos | Median anns / pos | Max anns / pos | Pos windows with 0 overlapping anns |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in density_df.iterrows():
        lines.append(
            f"| {row['fold']} | {row['project']} | "
            f"{int(row['n_pos_test'])} | {row['mean_ann_per_pos']:.2f} | "
            f"{row['median_ann_per_pos']:.2f} | {int(row['max_ann_per_pos'])} | "
            f"{int(row['pos_windows_with_zero_overlap'])} |"
        )
    lines.append("")

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold_dir",
        type=Path,
        default=Path("data/folds_segmented_v3"),
        help="Directory containing fold_X_*_segmented subdirectories (v3)",
    )
    parser.add_argument(
        "--windows_mapping",
        type=Path,
        default=Path("data/windows_mapping_4.0overlap_segmented_v3.json"),
        help="v3 windows-mapping JSON (used only for cross-check; not required)",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations_identification.json"),
        help="Annotations JSON used to compute annotation density",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated CSVs and the markdown summary",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = discover_folds(args.fold_dir)

    if args.windows_mapping.exists():
        project_rates_df = project_positive_rates(args.windows_mapping)
    else:
        print(f"[warn] windows mapping not found at {args.windows_mapping}; project-level rates skipped")
        project_rates_df = pd.DataFrame(columns=["project", "n_total", "n_pos", "n_neg", "pos_rate"])

    counts_rows: List[dict] = []
    confusion_rows: List[dict] = []
    threshold_rows: List[dict] = []
    density_rows: List[dict] = []

    ann_index = build_annotation_index(args.annotations) if args.annotations.exists() else {}
    if not ann_index:
        print(f"[warn] annotations not found at {args.annotations}; density will be empty")

    for fold_num, project, fold_path in folds:
        print(f"[fold {fold_num}] project={project} path={fold_path}")

        for split in ("train", "val", "test"):
            csv_path = fold_path / f"{split}_split.csv"
            if not csv_path.exists():
                print(f"  [skip] {csv_path} not found")
                continue
            stats = split_stats(csv_path)
            counts_rows.append(
                {"fold": fold_num, "project": project, "split": split, **stats}
            )

        pred_csv = fold_path / "test_split_with_predictions.csv"
        if pred_csv.exists():
            cm = confusion_counts(pred_csv)
            confusion_rows.append({"fold": fold_num, "project": project, **cm})
            threshold_rows.append(
                {"fold": fold_num, "project": project, **best_f1_stats(pred_csv)}
            )
        else:
            print(f"  [warn] {pred_csv} not found; confusion/threshold stats skipped")

        test_csv = fold_path / "test_split.csv"
        if test_csv.exists() and ann_index:
            density_rows.append(
                {
                    "fold": fold_num,
                    "project": project,
                    **annotation_density_for_fold(test_csv, ann_index),
                }
            )

    counts_df = pd.DataFrame(counts_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    density_df = pd.DataFrame(density_rows)

    counts_df.to_csv(args.output_dir / "window_counts_by_fold.csv", index=False)
    confusion_df.to_csv(args.output_dir / "confusion_counts_by_fold.csv", index=False)
    threshold_df.to_csv(args.output_dir / "threshold_stats_by_fold.csv", index=False)
    density_df.to_csv(args.output_dir / "annotation_density_by_fold.csv", index=False)
    project_rates_df.to_csv(args.output_dir / "project_positive_rates.csv", index=False)

    write_markdown_summary(
        counts_df,
        confusion_df,
        threshold_df,
        density_df,
        project_rates_df,
        args.output_dir / "window_counts_by_fold.md",
    )

    print(f"\nWrote outputs to {args.output_dir.resolve()}:")
    for name in (
        "window_counts_by_fold.csv",
        "confusion_counts_by_fold.csv",
        "threshold_stats_by_fold.csv",
        "annotation_density_by_fold.csv",
        "project_positive_rates.csv",
        "window_counts_by_fold.md",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
