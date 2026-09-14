"""
Rigorous comparison of non-overlapping vs overlapping inference strategies.

Compares:
  1. Baseline: predictions on non-overlapping 5s windows (2 per 10s segment)
  2. Overlapping: predictions on all overlapping 5s windows (1s stride),
     aggregated via weighted mean, max, and unweighted mean

Evaluation at two granularities:
  A. 5-second resolution (same evaluation units and labels as baseline)
  B. 1-second resolution (finer temporal analysis from annotations)

Usage:
    python compare_inference_strategies.py --config data/config.yaml

    # Enable annotation-derived metrics only with a matching v3 artifact:
    python compare_inference_strategies.py \
        --annotations data/annotations_identification_v3.json \
        --annotations_version v3
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

from prepare_dataset import spectrogram_filename, load_segmented_windows_if_exists
from plot_cv_results import evaluate_fold, FOLD_PROJECT_NAMES

from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECTS = ['MAP1', 'PPA1', 'PPA2', 'PPA3', 'PPA4']
HISTORICAL_MAPPING_VERSION = 'v3'
PPA1_STRIDE_SEC = 9
DEFAULT_STRIDE_SEC = 10
SEGMENT_DURATION_SEC = 10
WINDOW_SIZE_SEC = 5.0
AGGREGATION_METHODS = ['weighted_mean', 'max', 'unweighted_mean']


# ---------------------------------------------------------------------------
# Step 1: Generate overlapping test CSVs
# ---------------------------------------------------------------------------
def generate_overlapping_test_csvs(
    config,
    segmented_windows: List[dict],
    folds_base: str,
    staging_base: str,
    mapping_version: str,
) -> Dict[int, str]:
    """Create test CSVs containing ALL segmented overlapping windows for each fold's test project.

    Project identity comes from each historical window's ``dataset`` field,
    checked against the matching historical fold row. Derived CSVs are written
    only under ``staging_base``, never into the read-only historical folds.
    Returns dict mapping fold_idx -> path to overlapping test CSV.
    """
    print("\n" + "=" * 60)
    print("Step 1: Generate overlapping test CSVs")
    print("=" * 60)

    folds_path = os.path.abspath(folds_base)
    staging_path = os.path.abspath(staging_base)
    if os.path.commonpath([folds_path, staging_path]) == folds_path:
        raise ValueError("staging_base must be outside the historical folds directory")

    sounds = {}
    source_csvs = []
    for fold_idx, held_out_project in enumerate(PROJECTS):
        fold_name = f"fold_{fold_idx}_{held_out_project}_segmented"
        source_csv = os.path.join(folds_base, fold_name, 'test_split.csv')
        source_csvs.append(os.path.abspath(source_csv))
        with open(source_csv, newline='') as f:
            for row in csv.DictReader(f):
                sound_id = str(row['sound_id'])
                sound = {
                    'file_name_path': row['sound_filename'],
                    'project': row.get('project') or row.get('dataset'),
                }
                previous = sounds.get(sound_id)
                if previous is not None and previous != sound:
                    raise ValueError(
                        f"Conflicting historical fold metadata for sound_id "
                        f"{row['sound_id']!r}"
                    )
                sounds[sound_id] = sound

    spectrograms_dir = config.paths.spectrograms_dir

    # Enrich windows with spec_name, sound_filename, project
    enriched = []
    missing_spectrograms = []
    for w in segmented_windows:
        sound = sounds.get(str(w['sound_id']))
        if sound is None:
            raise ValueError(
                f"Window {w['window_id']} references unknown sound_id "
                f"{w['sound_id']!r}"
            )
        spec_name = spectrogram_filename(sound['file_name_path'], w['start'], w['end'])
        sound_fname = os.path.basename(sound['file_name_path'])
        window_project = w.get('dataset')
        sound_project = sound.get('project')
        if window_project and sound_project and window_project != sound_project:
            raise ValueError(
                f"Project mismatch for sound_id {w['sound_id']!r}: "
                f"window dataset is {window_project!r}, annotation project is "
                f"{sound_project!r}"
            )
        project = window_project or sound_project
        if not project:
            raise ValueError(
                f"No project identity for sound_id {w['sound_id']!r}; "
                "set window['dataset'] or annotation sound['project']"
            )
        if project not in PROJECTS:
            raise ValueError(
                f"Unsupported project {project!r} for sound_id {w['sound_id']!r}"
            )
        spec_path = os.path.join(spectrograms_dir, spec_name)
        if not os.path.isfile(spec_path):
            missing_spectrograms.append(
                (w['window_id'], os.path.abspath(spec_path))
            )
            continue
        enriched.append({
            'window_id': w['window_id'],
            'dataset': project,
            'sound_id': w['sound_id'],
            'start': w['start'],
            'end': w['end'],
            'label': w.get('label', 0),
            'spec_name': spec_name,
            'sound_filename': sound_fname,
            'project': project,
        })

    if missing_spectrograms:
        examples = ", ".join(
            f"window {window_id}: {path}"
            for window_id, path in missing_spectrograms[:5]
        )
        raise FileNotFoundError(
            f"Missing {len(missing_spectrograms)} expected spectrogram files; "
            f"no evaluation inputs were written. Examples: {examples}"
        )

    fieldnames = ['window_id', 'dataset', 'sample_rate', 'sound_id',
                  'start', 'end', 'label', 'spec_name', 'sound_filename', 'project']

    overlapping_csvs = {}
    os.makedirs(staging_base, exist_ok=True)

    for fold_idx, held_out_project in enumerate(PROJECTS):
        fold_name = f"fold_{fold_idx}_{held_out_project}_segmented"
        fold_dir = os.path.join(staging_base, fold_name)
        os.makedirs(fold_dir, exist_ok=True)

        test_data = [d for d in enriched if d['project'] == held_out_project]
        csv_path = os.path.join(fold_dir, 'test_split_overlapping.csv')

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in test_data:
                row = {k: d.get(k, '') for k in fieldnames}
                row['sample_rate'] = config.audio.sample_rate
                writer.writerow(row)

        overlapping_csvs[fold_idx] = csv_path
        print(f"  Fold {fold_idx} ({held_out_project}): {len(test_data)} overlapping windows -> {csv_path}")

    provenance_path = os.path.join(staging_base, 'provenance.json')
    with open(provenance_path, 'w') as f:
        json.dump(
            {
                'mapping_version': mapping_version,
                'mapping_source': (
                    f"windows_mapping_{config.audio.overlap_sec}overlap"
                    f"_segmented_{mapping_version}.json"
                ),
                'historical_fold_dir': folds_path,
                'historical_test_csvs': source_csvs,
            },
            f,
            indent=2,
        )

    return overlapping_csvs


# ---------------------------------------------------------------------------
# Step 2: Run inference on both strategies
# ---------------------------------------------------------------------------
def run_inference_both_strategies(
    config,
    folds_base: str,
    checkpoint_dir: str,
    overlapping_csvs: Dict[int, str],
) -> Tuple[List[Dict], List[Dict]]:
    """Run evaluate_fold() on baseline and overlapping CSVs for each fold.

    Returns (baseline_results, overlapping_results) — each a list of 5 dicts.
    """
    print("\n" + "=" * 60)
    print("Step 2: Run inference on both strategies")
    print("=" * 60)

    baseline_results = []
    overlapping_results = []

    for fold_idx, held_out_project in enumerate(PROJECTS):
        print(f"\n--- Fold {fold_idx} ({held_out_project}) ---")
        fold_name = f"fold_{fold_idx}_{held_out_project}_segmented"
        fold_dir = os.path.join(folds_base, fold_name)

        # Find checkpoint
        ckpt_dir = os.path.join(checkpoint_dir, f"fold_{fold_idx}")
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') and f != 'last.ckpt']
        ckpt_path = os.path.join(ckpt_dir, ckpts[0])

        eval_kwargs = dict(
            checkpoint_path=ckpt_path,
            spectrograms_root=config.paths.spectrograms_dir,
            x_col=config.training.x_col,
            target_size=config.training.target_size,
            batch_size=config.training.batch_size,
            num_workers=0,
            normalize=config.training.normalize,
        )

        # Baseline (non-overlapping)
        baseline_csv = os.path.join(fold_dir, 'test_split.csv')
        print(f"  Baseline: {baseline_csv}")
        baseline_result = evaluate_fold(test_csv=baseline_csv, **eval_kwargs)
        baseline_results.append(baseline_result)

        # Overlapping (all windows)
        overlap_csv = overlapping_csvs[fold_idx]
        print(f"  Overlapping: {overlap_csv}")
        overlap_result = evaluate_fold(test_csv=overlap_csv, **eval_kwargs)
        overlapping_results.append(overlap_result)

    return baseline_results, overlapping_results


# ---------------------------------------------------------------------------
# Step 3 & 4: Temporal alignment and aggregation
# ---------------------------------------------------------------------------
def _get_sound_strides(annotations_data: dict, sample_rate: int) -> Dict[int, int]:
    """Return per-sound stride in samples (9s for PPA1, 10s for others)."""
    sound_stride = {}
    for s in annotations_data['sounds']:
        if s.get('project') == 'PPA1':
            sound_stride[s['id']] = PPA1_STRIDE_SEC * sample_rate
        else:
            sound_stride[s['id']] = DEFAULT_STRIDE_SEC * sample_rate
    return sound_stride


def _get_valid_segments(sound_id: int, duration_sec: float, stride_sec: int) -> List[Tuple[float, float]]:
    """Return list of (start_sec, end_sec) for valid segments within a sound."""
    segments = []
    t = 0.0
    while t + SEGMENT_DURATION_SEC <= duration_sec + 0.001:  # small epsilon for float
        segments.append((t, t + SEGMENT_DURATION_SEC))
        t += stride_sec
    return segments


def aggregate_to_per_second(
    result: Dict,
    sample_rate: int,
) -> Dict[int, Dict[int, Dict[str, float]]]:
    """Aggregate window-level predictions to per-second resolution.

    Returns:
        {sound_id: {second: {'weighted_mean': p, 'max': p, 'unweighted_mean': p}}}
    """
    test_df = result['test_df']
    probs = result['probs']

    # Group by sound_id
    per_sound = defaultdict(list)
    for i, (_, row) in enumerate(test_df.iterrows()):
        sound_id = int(row['sound_id'])
        start_sec = int(row['start']) / sample_rate
        end_sec = int(row['end']) / sample_rate
        prob = float(probs[i])
        per_sound[sound_id].append((start_sec, end_sec, prob))

    aggregated = {}
    for sound_id, windows in per_sound.items():
        # Determine time range
        min_start = min(w[0] for w in windows)
        max_end = max(w[1] for w in windows)

        sound_agg = {}
        for second in range(int(min_start), int(max_end)):
            # Find overlapping windows
            covering = []
            weights = []
            for w_start, w_end, prob in windows:
                overlap_start = max(w_start, second)
                overlap_end = min(w_end, second + 1)
                overlap_dur = max(0.0, overlap_end - overlap_start)
                if overlap_dur > 0:
                    covering.append(prob)
                    weights.append(overlap_dur)

            if covering:
                covering = np.array(covering)
                weights = np.array(weights)
                sound_agg[second] = {
                    'weighted_mean': float(np.average(covering, weights=weights)),
                    'max': float(np.max(covering)),
                    'unweighted_mean': float(np.mean(covering)),
                    'n_windows': len(covering),
                }

        aggregated[sound_id] = sound_agg

    return aggregated


def aggregate_overlapping_to_5s_blocks(
    overlapping_result: Dict,
    baseline_result: Dict,
    sample_rate: int,
) -> Dict[str, np.ndarray]:
    """Aggregate overlapping predictions onto the baseline's 5s evaluation units.

    For each non-overlapping 5s window in the baseline, collect all overlapping
    windows that cover its time range and aggregate.

    Returns dict of aggregation_method -> array of probabilities aligned to baseline order.
    """
    baseline_df = baseline_result['test_df']
    overlap_df = overlapping_result['test_df']
    overlap_probs = overlapping_result['probs']

    # Build lookup: (sound_id, start_sec, end_sec) -> prob for overlapping
    overlap_windows = []
    for i, (_, row) in enumerate(overlap_df.iterrows()):
        overlap_windows.append({
            'sound_id': int(row['sound_id']),
            'start_sec': int(row['start']) / sample_rate,
            'end_sec': int(row['end']) / sample_rate,
            'prob': float(overlap_probs[i]),
        })

    # Index by sound_id for fast lookup
    by_sound = defaultdict(list)
    for w in overlap_windows:
        by_sound[w['sound_id']].append(w)

    agg_probs = {method: [] for method in AGGREGATION_METHODS}

    for _, row in baseline_df.iterrows():
        sound_id = int(row['sound_id'])
        block_start = int(row['start']) / sample_rate
        block_end = int(row['end']) / sample_rate

        # Find overlapping windows for this block
        covering = []
        weights = []
        for w in by_sound.get(sound_id, []):
            overlap_start = max(w['start_sec'], block_start)
            overlap_end = min(w['end_sec'], block_end)
            overlap_dur = overlap_end - overlap_start
            if overlap_dur > 0:
                covering.append(w['prob'])
                weights.append(overlap_dur)

        if covering:
            covering = np.array(covering)
            weights = np.array(weights)
            agg_probs['weighted_mean'].append(float(np.average(covering, weights=weights)))
            agg_probs['max'].append(float(np.max(covering)))
            agg_probs['unweighted_mean'].append(float(np.mean(covering)))
        else:
            # Fallback: no overlapping windows found (shouldn't happen)
            agg_probs['weighted_mean'].append(0.0)
            agg_probs['max'].append(0.0)
            agg_probs['unweighted_mean'].append(0.0)

    return {k: np.array(v) for k, v in agg_probs.items()}


# ---------------------------------------------------------------------------
# Step 5: Derive per-second ground truth from annotations
# ---------------------------------------------------------------------------
def derive_per_second_ground_truth(
    annotations_data: dict,
    sound_ids: set,
    sample_rate: int,
) -> Dict[int, Dict[int, int]]:
    """Derive 1-second resolution ground truth from annotations.

    Only considers seconds within valid segment boundaries.

    Returns {sound_id: {second: label}} where label is 0 or 1.
    """
    sounds = {s['id']: s for s in annotations_data['sounds']}

    # Index annotations by sound_id
    anns_by_sound = defaultdict(list)
    for ann in annotations_data['annotations']:
        if ann['sound_id'] in sound_ids:
            anns_by_sound[ann['sound_id']].append(ann)

    ground_truth = {}

    for sound_id in sound_ids:
        sound = sounds.get(sound_id)
        if sound is None:
            continue

        duration = sound['duration']
        project = sound.get('project', '')
        stride_sec = PPA1_STRIDE_SEC if project == 'PPA1' else DEFAULT_STRIDE_SEC

        # Get valid segments
        segments = _get_valid_segments(sound_id, duration, stride_sec)
        valid_seconds = set()
        for seg_start, seg_end in segments:
            for t in range(int(seg_start), int(seg_end)):
                valid_seconds.add(t)

        # Assign labels
        sound_gt = {}
        sound_anns = anns_by_sound.get(sound_id, [])
        for t in sorted(valid_seconds):
            label = 0
            for ann in sound_anns:
                # Annotation overlaps second [t, t+1) if t_min < t+1 AND t_max > t
                if ann['t_min'] < t + 1 and ann['t_max'] > t:
                    label = 1
                    break
            sound_gt[t] = label

        ground_truth[sound_id] = sound_gt

    return ground_truth


# ---------------------------------------------------------------------------
# Step 6 & 7: Compute metrics and comparison
# ---------------------------------------------------------------------------
def compute_binary_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute binary classification metrics."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    prec_curve, rec_curve, _ = precision_recall_curve(targets, probs)
    auprc = auc(rec_curve, prec_curve)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auprc': auprc,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def compare_at_5s_resolution(
    baseline_results: List[Dict],
    overlapping_results: List[Dict],
    sample_rate: int,
) -> pd.DataFrame:
    """Compare strategies at 5-second resolution.

    Uses baseline's evaluation units and labels. Overlapping predictions are
    aggregated onto the same 5s blocks.
    """
    print("\n" + "=" * 60)
    print("Comparison at 5-second resolution")
    print("=" * 60)

    rows = []

    for fold_idx in range(len(PROJECTS)):
        project = PROJECTS[fold_idx]
        baseline = baseline_results[fold_idx]
        overlapping = overlapping_results[fold_idx]

        targets = baseline['targets']
        baseline_probs = baseline['probs']

        # Aggregate overlapping predictions onto baseline's 5s blocks
        agg_probs = aggregate_overlapping_to_5s_blocks(
            overlapping, baseline, sample_rate
        )

        # Baseline metrics
        bl_metrics = compute_binary_metrics(targets, baseline_probs)
        rows.append({
            'fold': fold_idx,
            'project': project,
            'strategy': 'baseline',
            'aggregation': 'none',
            'resolution': '5s',
            **bl_metrics,
        })

        print(f"\n  Fold {fold_idx} ({project}):")
        print(f"    Baseline:        F1={bl_metrics['f1']:.4f}  AUPRC={bl_metrics['auprc']:.4f}  Prec={bl_metrics['precision']:.4f}  Rec={bl_metrics['recall']:.4f}")

        # Overlapping metrics for each aggregation method
        for method in AGGREGATION_METHODS:
            ov_metrics = compute_binary_metrics(targets, agg_probs[method])
            rows.append({
                'fold': fold_idx,
                'project': project,
                'strategy': 'overlapping',
                'aggregation': method,
                'resolution': '5s',
                **ov_metrics,
            })
            print(f"    Overlapping ({method:16s}): F1={ov_metrics['f1']:.4f}  AUPRC={ov_metrics['auprc']:.4f}  Prec={ov_metrics['precision']:.4f}  Rec={ov_metrics['recall']:.4f}")

    return pd.DataFrame(rows)


def compare_at_1s_resolution(
    baseline_results: List[Dict],
    overlapping_results: List[Dict],
    annotations_data: dict,
    sample_rate: int,
) -> pd.DataFrame:
    """Compare strategies at 1-second resolution.

    Derives per-second ground truth from annotations. Baseline assigns each
    second the probability of its containing 5s block. Overlapping uses
    aggregated per-second probabilities.
    """
    print("\n" + "=" * 60)
    print("Comparison at 1-second resolution")
    print("=" * 60)

    sounds = {s['id']: s for s in annotations_data['sounds']}
    rows = []

    for fold_idx in range(len(PROJECTS)):
        project = PROJECTS[fold_idx]
        baseline = baseline_results[fold_idx]
        overlapping = overlapping_results[fold_idx]

        # Get sound_ids in this fold's test set
        test_sound_ids = set(int(sid) for sid in baseline['test_df']['sound_id'].unique())

        # Per-second ground truth
        gt = derive_per_second_ground_truth(annotations_data, test_sound_ids, sample_rate)

        # Baseline: map each second to its containing 5s block's probability
        baseline_per_second = _baseline_to_per_second(baseline, sample_rate)

        # Overlapping: aggregate to per-second
        overlap_per_second = aggregate_to_per_second(overlapping, sample_rate)

        # Build aligned arrays (only seconds present in both GT and predictions)
        bl_targets, bl_probs_arr = [], []
        ov_targets = {m: [] for m in AGGREGATION_METHODS}
        ov_probs_arr = {m: [] for m in AGGREGATION_METHODS}

        for sound_id in sorted(gt.keys()):
            for second in sorted(gt[sound_id].keys()):
                label = gt[sound_id][second]

                # Baseline
                if sound_id in baseline_per_second and second in baseline_per_second[sound_id]:
                    bl_targets.append(label)
                    bl_probs_arr.append(baseline_per_second[sound_id][second])

                # Overlapping
                if sound_id in overlap_per_second and second in overlap_per_second[sound_id]:
                    for method in AGGREGATION_METHODS:
                        ov_targets[method].append(label)
                        ov_probs_arr[method].append(overlap_per_second[sound_id][second][method])

        bl_targets = np.array(bl_targets)
        bl_probs_arr = np.array(bl_probs_arr)

        # Baseline metrics at 1s
        if len(bl_targets) > 0:
            bl_metrics = compute_binary_metrics(bl_targets, bl_probs_arr)
            rows.append({
                'fold': fold_idx,
                'project': project,
                'strategy': 'baseline',
                'aggregation': 'none',
                'resolution': '1s',
                **bl_metrics,
            })
            print(f"\n  Fold {fold_idx} ({project}) — {len(bl_targets)} seconds evaluated:")
            print(f"    Baseline:        F1={bl_metrics['f1']:.4f}  AUPRC={bl_metrics['auprc']:.4f}  Prec={bl_metrics['precision']:.4f}  Rec={bl_metrics['recall']:.4f}")

        # Overlapping metrics at 1s
        for method in AGGREGATION_METHODS:
            t = np.array(ov_targets[method])
            p = np.array(ov_probs_arr[method])
            if len(t) > 0:
                ov_metrics = compute_binary_metrics(t, p)
                rows.append({
                    'fold': fold_idx,
                    'project': project,
                    'strategy': 'overlapping',
                    'aggregation': method,
                    'resolution': '1s',
                    **ov_metrics,
                })
                print(f"    Overlapping ({method:16s}): F1={ov_metrics['f1']:.4f}  AUPRC={ov_metrics['auprc']:.4f}  Prec={ov_metrics['precision']:.4f}  Rec={ov_metrics['recall']:.4f}")

    return pd.DataFrame(rows)


def _baseline_to_per_second(
    baseline_result: Dict,
    sample_rate: int,
) -> Dict[int, Dict[int, float]]:
    """Map baseline 5s window predictions to per-second resolution.

    Each second within a non-overlapping 5s block inherits the block's probability.
    """
    test_df = baseline_result['test_df']
    probs = baseline_result['probs']

    per_second = defaultdict(dict)

    for i, (_, row) in enumerate(test_df.iterrows()):
        sound_id = int(row['sound_id'])
        start_sec = int(row['start']) / sample_rate
        end_sec = int(row['end']) / sample_rate
        prob = float(probs[i])

        for t in range(int(start_sec), int(end_sec)):
            per_second[sound_id][t] = prob

    return dict(per_second)


# ---------------------------------------------------------------------------
# Step 8: Boundary sensitivity analysis
# ---------------------------------------------------------------------------
def boundary_sensitivity_analysis(
    overlapping_results: List[Dict],
    annotations_data: dict,
    sample_rate: int,
) -> pd.DataFrame:
    """Sensitivity analysis excluding boundary seconds (first 4 and last 4 of each segment).

    Only evaluates seconds with full coverage (>= 5 overlapping windows), i.e.,
    seconds 4 and 5 of each 10s segment.
    """
    print("\n" + "=" * 60)
    print("Boundary sensitivity analysis (1s, interior seconds only)")
    print("=" * 60)

    sounds = {s['id']: s for s in annotations_data['sounds']}
    rows = []

    for fold_idx in range(len(PROJECTS)):
        project = PROJECTS[fold_idx]
        overlapping = overlapping_results[fold_idx]

        test_sound_ids = set(int(sid) for sid in overlapping['test_df']['sound_id'].unique())
        gt = derive_per_second_ground_truth(annotations_data, test_sound_ids, sample_rate)
        overlap_per_second = aggregate_to_per_second(overlapping, sample_rate)

        # Determine interior seconds (coverage >= 5)
        for method in AGGREGATION_METHODS:
            targets_list, probs_list = [], []
            for sound_id in sorted(gt.keys()):
                sound = sounds.get(sound_id)
                if sound is None:
                    continue
                stride_sec = PPA1_STRIDE_SEC if sound.get('project') == 'PPA1' else DEFAULT_STRIDE_SEC
                segments = _get_valid_segments(sound_id, sound['duration'], stride_sec)

                # Interior seconds: offset 4..5 within each segment
                interior = set()
                for seg_start, seg_end in segments:
                    for offset in range(4, 6):
                        t = int(seg_start) + offset
                        if t < int(seg_end):
                            interior.add(t)

                for second in sorted(interior):
                    if second in gt[sound_id] and sound_id in overlap_per_second and second in overlap_per_second[sound_id]:
                        targets_list.append(gt[sound_id][second])
                        probs_list.append(overlap_per_second[sound_id][second][method])

            if targets_list:
                t = np.array(targets_list)
                p = np.array(probs_list)
                metrics = compute_binary_metrics(t, p)
                rows.append({
                    'fold': fold_idx,
                    'project': project,
                    'aggregation': method,
                    'resolution': '1s_interior',
                    **metrics,
                })
                print(f"  Fold {fold_idx} ({project}) {method:16s}: F1={metrics['f1']:.4f}  AUPRC={metrics['auprc']:.4f}  ({len(t)} seconds)")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pr_curves(
    baseline_results: List[Dict],
    overlapping_results: List[Dict],
    sample_rate: int,
    output_path: str,
):
    """Plot side-by-side PR curves for baseline vs overlapping (weighted mean) at 5s resolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.magma(np.linspace(0.2, 0.85, 5))

    # Left: Baseline
    ax = axes[0]
    ax.set_title("Baseline (non-overlapping)", fontsize=13, fontweight='bold')
    for fold_idx in range(len(PROJECTS)):
        targets = baseline_results[fold_idx]['targets']
        probs = baseline_results[fold_idx]['probs']
        prec, rec, _ = precision_recall_curve(targets, probs)
        ap = auc(rec, prec)
        ax.plot(rec, prec, color=colors[fold_idx], lw=2,
                label=f"Fold {fold_idx} - {PROJECTS[fold_idx]} ({ap:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    # Right: Overlapping (weighted mean aggregated onto 5s blocks)
    ax = axes[1]
    ax.set_title("Overlapping (weighted mean)", fontsize=13, fontweight='bold')
    for fold_idx in range(len(PROJECTS)):
        targets = baseline_results[fold_idx]['targets']
        agg = aggregate_overlapping_to_5s_blocks(
            overlapping_results[fold_idx], baseline_results[fold_idx], sample_rate
        )
        probs = agg['weighted_mean']
        prec, rec, _ = precision_recall_curve(targets, probs)
        ap = auc(rec, prec)
        ax.plot(rec, prec, color=colors[fold_idx], lw=2,
                label=f"Fold {fold_idx} - {PROJECTS[fold_idx]} ({ap:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.suptitle("Precision-Recall Curves: Baseline vs Overlapping (5s resolution)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPR curves saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean +/- std across folds for each (strategy, aggregation, resolution) group."""
    metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'auprc']
    group_cols = ['strategy', 'aggregation', 'resolution']

    summary_rows = []
    for group_key, group_df in results_df.groupby(group_cols):
        row = dict(zip(group_cols, group_key))
        for col in metric_cols:
            vals = group_df[col].values
            row[f'{col}_mean'] = vals.mean()
            row[f'{col}_std'] = vals.std(ddof=1)
        row['n_folds'] = len(group_df)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_summary(summary_df: pd.DataFrame):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY: Mean +/- Std across folds")
    print("=" * 100)

    metric_cols = ['f1', 'auprc', 'precision', 'recall', 'accuracy']

    for resolution in summary_df['resolution'].unique():
        res_df = summary_df[summary_df['resolution'] == resolution]
        print(f"\n--- Resolution: {resolution} ---")
        print(f"{'Strategy':<12} {'Aggregation':<18} ", end="")
        for col in metric_cols:
            print(f"{col:>18s}", end="")
        print()
        print("-" * 100)

        for _, row in res_df.iterrows():
            print(f"{row['strategy']:<12} {row['aggregation']:<18} ", end="")
            for col in metric_cols:
                mean = row[f'{col}_mean']
                std = row[f'{col}_std']
                print(f"  {mean:.4f}+/-{std:.4f}", end="")
            print()

    print("\n" + "=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare non-overlapping vs overlapping inference strategies")
    parser.add_argument("--config", type=str, default="data/config.yaml")
    parser.add_argument("--fold_dir", type=str, default="data/folds_segmented_v3")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_v3")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_v3/inference_strategy_comparison",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help=(
            "Explicit annotation artifact for annotation-derived metrics; "
            "must be paired with --annotations_version v3"
        ),
    )
    parser.add_argument(
        "--annotations_version",
        type=str,
        default=None,
        help="Dataset version represented by --annotations",
    )
    return parser


def load_annotation_metrics_input(
    annotations_path: str,
    annotations_version: str,
    mapping_version: str,
):
    """Load explicitly version-matched annotations or disable derived metrics."""
    if annotations_path is None:
        if annotations_version is not None:
            raise ValueError("--annotations_version requires --annotations")
        print(
            "Annotation-derived 1-second and boundary analyses disabled: "
            "no explicit version-matched annotations were provided"
        )
        return None
    if annotations_version != mapping_version:
        raise ValueError(
            f"--annotations_version must be {mapping_version!r} for the "
            f"{mapping_version} mapping and checkpoints"
        )
    with open(annotations_path, 'r') as f:
        return json.load(f)


def main():
    args = build_arg_parser().parse_args()

    config = load_config(args.config)
    sample_rate = config.audio.sample_rate
    os.makedirs(args.output_dir, exist_ok=True)

    # Load segmented windows
    segmented_windows = load_segmented_windows_if_exists(
        config,
        version=HISTORICAL_MAPPING_VERSION,
    )
    if segmented_windows is None:
        raise FileNotFoundError(
            "Historical segmented windows v3 not found; restore the read-only "
            "v3 mapping artifact rather than regenerating it"
        )
    print(f"Loaded {len(segmented_windows)} segmented windows")

    annotations_data = load_annotation_metrics_input(
        args.annotations,
        args.annotations_version,
        HISTORICAL_MAPPING_VERSION,
    )

    # Step 1: Generate overlapping test CSVs
    staging_dir = os.path.join(
        args.output_dir,
        'staging',
        HISTORICAL_MAPPING_VERSION,
    )
    overlapping_csvs = generate_overlapping_test_csvs(
        config,
        segmented_windows,
        args.fold_dir,
        staging_dir,
        HISTORICAL_MAPPING_VERSION,
    )

    # Step 2: Run inference on both strategies
    baseline_results, overlapping_results = run_inference_both_strategies(
        config, args.fold_dir, args.checkpoint_dir, overlapping_csvs
    )

    # Steps 3-7: Compare at both granularities
    results_5s = compare_at_5s_resolution(baseline_results, overlapping_results, sample_rate)
    if annotations_data is None:
        results_1s = pd.DataFrame()
        results_boundary = pd.DataFrame()
    else:
        results_1s = compare_at_1s_resolution(
            baseline_results,
            overlapping_results,
            annotations_data,
            sample_rate,
        )
        results_boundary = boundary_sensitivity_analysis(
            overlapping_results,
            annotations_data,
            sample_rate,
        )

    # Combine all results
    all_results = pd.concat(
        [result for result in (results_5s, results_1s) if not result.empty],
        ignore_index=True,
    )
    results_path = os.path.join(args.output_dir, "comparison_results.csv")
    all_results.to_csv(results_path, index=False)
    print(f"\nAll results saved to: {results_path}")

    if not results_boundary.empty:
        boundary_path = os.path.join(args.output_dir, "comparison_boundary_sensitivity.csv")
        results_boundary.to_csv(boundary_path, index=False)
        print(f"Boundary sensitivity saved to: {boundary_path}")

    # Summary
    summary = compute_summary(all_results)
    summary_path = os.path.join(args.output_dir, "comparison_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    print_summary(summary)

    # PR curves
    pr_path = os.path.join(args.output_dir, "comparison_pr_curves.png")
    plot_pr_curves(baseline_results, overlapping_results, sample_rate, pr_path)

    # Verification: baseline 5s metrics should match cv_results.csv
    print("\n" + "=" * 60)
    print("VERIFICATION: Baseline 5s metrics vs cv_results.csv")
    print("=" * 60)
    cv_results_path = os.path.join(args.output_dir, "cv_results.csv")
    if os.path.exists(cv_results_path):
        cv_df = pd.read_csv(cv_results_path)
        for fold_idx in range(len(PROJECTS)):
            bl_row = results_5s[(results_5s['fold'] == fold_idx) & (results_5s['strategy'] == 'baseline')].iloc[0]
            cv_row = cv_df[cv_df['fold'] == fold_idx].iloc[0]
            f1_match = abs(bl_row['f1'] - cv_row['f1']) < 1e-4
            auprc_match = abs(bl_row['auprc'] - cv_row['auprc']) < 1e-4
            status = "OK" if (f1_match and auprc_match) else "MISMATCH"
            print(f"  Fold {fold_idx}: F1 {bl_row['f1']:.6f} vs {cv_row['f1']:.6f} | "
                  f"AUPRC {bl_row['auprc']:.6f} vs {cv_row['auprc']:.6f} [{status}]")


if __name__ == "__main__":
    main()
