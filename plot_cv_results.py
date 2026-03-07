"""
Script to visualize cross-validation results.

This script:
1. Loads checkpoints from each fold
2. Runs evaluation to get predictions and targets
3. Plots precision-recall curves for each fold
4. Displays all metrics in a formatted table including overall accuracy
5. Computes mean and std across folds

Usage:
    python plot_cv_results.py --fold_dir data --config data/config.yaml --checkpoint_dir checkpoints
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import pytorch_lightning as pl

# Import from training script
from train import SpectrogramDataModule, DataModuleConfig

# Import from PytorchWildlife core library
from PytorchWildlife.models.bioacoustics import ResNetClassifier
from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config


def evaluate_fold(
    checkpoint_path: str,
    test_csv: str,
    spectrograms_root: str,
    x_col: str,
    target_size: List[int],
    batch_size: int,
    num_workers: int,
    normalize: bool,
) -> Dict:
    """
    Evaluate a single fold and return predictions, targets, and probabilities.
    
    Returns:
        Dictionary with 'preds', 'targets', 'probs', and 'metrics'
    """
    # Load model from checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    model = ResNetClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    
    # Create test dataset
    dm_cfg = DataModuleConfig(
        train_csv=test_csv,  # Dummy, not used
        val_csv=test_csv,    # Dummy, not used
        test_csv=test_csv,
        spectrograms_root=spectrograms_root,
        x_col=x_col,
        target_size=target_size,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        use_specaug=False,
        use_mixup=False,
    )
    
    from train import SpectrogramDataModule
    dm = SpectrogramDataModule(dm_cfg)
    dm.setup()
    
    # Run inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    all_logits = []
    all_targets = []
    all_preds = []
    all_paths = []
    
    test_loader = dm.test_dataloader()
    
    with torch.no_grad():
        for batch in test_loader:
            x, y, path = batch
            x = x.to(device)
            
            logits = model(x)
            
            if model.is_binary:
                logits = logits.squeeze(1)
                probs = torch.sigmoid(logits)
                preds = (probs > model.hparams.conf_threshold).int()
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_paths.extend(path)
    
    # Concatenate results
    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    
    # Convert logits to probabilities for binary
    if model.is_binary:
        all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    else:
        all_probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)  # softmax
    
    # Calculate metrics
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate metrics
    if model.is_binary:
        tn, fp, fn, tp = cm.ravel()
        
        # Overall accuracy
        acc = (tp + tn) / (tp + tn + fp + fn)
        
        # Negative class accuracy (specificity)
        acc_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Positive class accuracy (recall/sensitivity)
        acc_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUPRC
        prec_curve, rec_curve, _ = precision_recall_curve(all_targets, all_probs)
        auprc = auc(rec_curve, prec_curve)
        
        metrics = {
            'acc': acc,
            'acc_neg': acc_neg,
            'acc_pos': acc_pos,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auprc': auprc,
            'loss': np.nan,  # Not computed here
        }
    else:
        # Multiclass metrics
        acc = (all_preds == all_targets).mean()
        metrics = {
            'acc': acc,
        }
    
    return {
        'preds': all_preds,
        'targets': all_targets,
        'probs': all_probs,
        'paths': all_paths,
        'test_df': pd.read_csv(test_csv),
        'metrics': metrics,
        'is_binary': model.is_binary,
    }


def plot_precision_recall_curves(fold_results: List[Dict], output_path: str):
    """
    Plot precision-recall curves for all folds.
    
    Args:
        fold_results: List of result dictionaries from evaluate_fold
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_results)))
    
    for i, result in enumerate(fold_results):
        targets = result['targets']
        probs = result['probs']
        auprc = result['metrics']['auprc']
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(targets, probs)
        
        # Plot
        ax.plot(recall, precision, color=colors[i], lw=2, 
                label=f'Fold {i} (AUPRC = {auprc:.3f})')
    
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curves - Cross-Validation', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPrecision-Recall curves saved to: {output_path}")
    plt.close()


def _hz_to_mel_bin(freq_hz, max_freq_hz, n_mels):
    """Convert a frequency in Hz to a mel-scale bin position."""
    mel = 2595.0 * np.log10(1.0 + freq_hz / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + max_freq_hz / 700.0)
    return mel / mel_max * n_mels


def plot_qualitative_results(
    fold_results: List[Dict],
    output_path: str,
    audio_dir: str,
    annotations_path: str = None,
    class_names: List[str] = None,
    seed: int = 42,
    duration_sec: float = 5.0,
    max_freq_hz: float = 24000.0,
    n_mels: int = 224,
):
    """
    Plot a grid of spectrograms showing one TP, TN, FP, and FN example per fold,
    and save the corresponding trimmed audio clips.

    Each row corresponds to a category and each column to a fold.
    For TP and FN examples, overlapping annotations are drawn as rectangles.
    Audio clips are saved next to the plot as
    ``<output_stem>_fold<F>_<category>.wav``.

    Args:
        fold_results: List of result dictionaries from evaluate_fold.
        output_path: Path to save the plot image.
        audio_dir: Directory containing the source audio files.
        annotations_path: Path to the annotations JSON file.
        class_names: List mapping class indices to human-readable names.
        seed: Random seed for reproducibility.
        duration_sec: Duration of each spectrogram window in seconds.
        max_freq_hz: Maximum frequency in Hz (Nyquist frequency).
        n_mels: Number of mel frequency bins.
    """
    rng = np.random.default_rng(seed)
    categories = ['TP', 'TN', 'FP', 'FN']
    cat_colors = {'TP': 'green', 'TN': 'green', 'FP': 'red', 'FN': 'red'}
    n_folds = len(fold_results)
    n_rows = len(categories)
    out_stem = Path(output_path).stem
    out_dir = Path(output_path).parent

    # Load annotations indexed by sound_id for TP/FN overlay
    annotations_by_sound = {}
    if annotations_path and os.path.exists(annotations_path):
        with open(annotations_path) as f:
            ann_data = json.load(f)
        for ann in ann_data.get('annotations', []):
            sid = ann['sound_id']
            annotations_by_sound.setdefault(sid, []).append(ann)

    # Precompute mel-scale frequency tick positions
    freq_ticks_hz = np.array([0, 1000, 2000, 4000, 8000, 16000, 24000])
    freq_ticks_hz = freq_ticks_hz[freq_ticks_hz <= max_freq_hz]
    mel_ticks = 2595.0 * np.log10(1.0 + freq_ticks_hz / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + max_freq_hz / 700.0)
    # Map mel values to bin positions (0 to n_mels)
    mel_tick_positions = mel_ticks / mel_max * n_mels
    freq_tick_labels = [f"{int(f/1000)}k" if f >= 1000 else str(int(f)) for f in freq_ticks_hz]

    fig, axes = plt.subplots(n_rows, n_folds, figsize=(3 * n_folds, 3.5 * n_rows))
    if n_folds == 1:
        axes = axes[:, np.newaxis]

    # Pre-compute per-fold category indices
    fold_cat_indices = []
    for result in fold_results:
        preds = result['preds']
        targets = result['targets']
        fold_cat_indices.append({
            'TP': np.where((preds == 1) & (targets == 1))[0],
            'TN': np.where((preds == 0) & (targets == 0))[0],
            'FP': np.where((preds == 1) & (targets == 0))[0],
            'FN': np.where((preds == 0) & (targets == 1))[0],
        })

    for row, cat in enumerate(categories):
        for col, result in enumerate(fold_results):
            ax = axes[row, col]
            paths = result['paths']
            targets = result['targets']
            test_df = result['test_df']
            indices = fold_cat_indices[col][cat]

            # Column headers (fold numbers) on the first row
            if row == 0:
                ax.set_title(f"Fold {col}", fontsize=11, fontweight='bold')

            if len(indices) == 0:
                ax.text(0.5, 0.5, f'No {cat}', ha='center', va='center',
                        fontsize=11, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                idx = rng.choice(indices)
                spec = np.load(paths[idx])
                n_time_frames = spec.shape[-1]
                ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')

                # X-axis: time in seconds
                time_ticks_sec = np.arange(0, duration_sec + 0.5, 1.0)
                time_tick_positions = time_ticks_sec / duration_sec * n_time_frames
                ax.set_xticks(time_tick_positions)
                if row == n_rows - 1:
                    ax.set_xticklabels([f"{t:.0f}" for t in time_ticks_sec], fontsize=7)
                    ax.set_xlabel("Time (s)", fontsize=8)
                else:
                    ax.set_xticklabels([])

                # Y-axis: frequency in mel scale
                ax.set_yticks(mel_tick_positions)
                if col == 0:
                    ax.set_yticklabels(freq_tick_labels, fontsize=7)
                    ax.set_ylabel("Freq (Hz)", fontsize=9, labelpad=1)
                else:
                    ax.set_yticklabels([])

                # Overlay annotations for TP and FN
                spec_basename = os.path.basename(paths[idx])
                spec_to_row = {r['spec_name']: r for _, r in test_df.iterrows()}
                csv_row = spec_to_row.get(spec_basename)

                if cat in ('TP', 'FN') and csv_row is not None and annotations_by_sound:
                    sound_id = int(csv_row['sound_id'])
                    sr = int(csv_row['sample_rate'])
                    win_start_sec = int(csv_row['start']) / sr
                    win_end_sec = int(csv_row['end']) / sr

                    for ann in annotations_by_sound.get(sound_id, []):
                        t_min = ann['t_min']
                        t_max = ann['t_max']
                        # Check overlap with window
                        if t_max <= win_start_sec or t_min >= win_end_sec:
                            continue
                        # Clip to window bounds and convert to pixel coords
                        t0 = max(t_min - win_start_sec, 0.0)
                        t1 = min(t_max - win_start_sec, duration_sec)
                        x0 = t0 / duration_sec * n_time_frames
                        x1 = t1 / duration_sec * n_time_frames

                        f_min = ann.get('f_min', 0.0)
                        f_max = ann.get('f_max', max_freq_hz)
                        y0 = _hz_to_mel_bin(f_min, max_freq_hz, n_mels)
                        y1 = _hz_to_mel_bin(f_max, max_freq_hz, n_mels)

                        rect = mpatches.Rectangle(
                            (x0, y0), x1 - x0, y1 - y0,
                            linewidth=1.5, edgecolor='cyan', facecolor='none',
                            linestyle='--',
                        )
                        ax.add_patch(rect)

                # Save trimmed audio clip
                if csv_row is not None:
                    audio_path = os.path.join(audio_dir, csv_row['sound_filename'])
                    start_sample = int(csv_row['start'])
                    end_sample = int(csv_row['end'])
                    sr = int(csv_row['sample_rate'])
                    try:
                        audio_data, _ = sf.read(
                            audio_path,
                            start=start_sample,
                            stop=end_sample,
                            dtype='float32',
                        )
                        clip_path = out_dir / f"{out_stem}_fold{col}_{cat}.wav"
                        sf.write(str(clip_path), audio_data, sr)
                        print(f"  Saved audio: {clip_path}")
                    except Exception as e:
                        print(f"  Warning: could not save audio for fold {col} {cat}: {e}")

            # Category label on the left for the first column
            if col == 0:
                ax.text(
                    -0.17, 0.5, cat,
                    transform=ax.transAxes,
                    fontsize=10, fontweight='bold', color=cat_colors[cat],
                    ha='right', va='center', rotation=90,
                )

    # General title
    fig.suptitle("Qualitative Examples per Fold", fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nQualitative results saved to: {output_path}")
    plt.close()


def print_results_table(fold_results: List[Dict], saved_metrics: Dict = None):
    """
    Print results in a formatted table.
    
    Args:
        fold_results: List of result dictionaries from evaluate_fold
        saved_metrics: Optional dict with metrics from training output
    """
    print("\n" + "="*90)
    print("CROSS-VALIDATION RESULTS")
    print("="*90 + "\n")
   
    # Determine metrics to display
    if fold_results[0]['is_binary']:
        metric_names = ['loss', 'f1', 'auprc', 'precision', 'recall', 'acc', 'acc_neg', 'acc_pos']
        metric_labels = ['Loss', 'F1', 'AUPRC', 'Precision', 'Recall', 'Accuracy', 'Acc (Neg)', 'Acc (Pos)']
    else:
        metric_names = ['loss', 'f1', 'acc']
        metric_labels = ['Loss', 'F1', 'Accuracy']
    
    # Per-fold results
    print("Per-Fold Results:")
    print("-" * 90)
    
    for fold_num, result in enumerate(fold_results):
        print(f"\nFold {fold_num}:")
        metrics = result['metrics']
        
        # If saved_metrics provided, use those (includes loss which we don't compute)
        if saved_metrics and fold_num in saved_metrics:
            metrics = saved_metrics[fold_num]
        
        for metric_name in metric_names:
            if metric_name in metrics:
                value = metrics[metric_name]
                if not np.isnan(value):
                    print(f"    test/{metric_name}: {value:.4f}")
    
    # Aggregate statistics
    print("\n" + "-" * 90)
    print("Aggregate Statistics:")
    print("-" * 90)
    
    # Create dataframe for easier computation
    df_data = []
    for fold_num, result in enumerate(fold_results):
        metrics = result['metrics']
        if saved_metrics and fold_num in saved_metrics:
            metrics = saved_metrics[fold_num]
        df_data.append(metrics)
    
    df = pd.DataFrame(df_data)
    
    # Print mean and std
    print(f"\n{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 90)
    
    for metric_name, metric_label in zip(metric_names, metric_labels):
        if metric_name in df.columns:
            values = df[metric_name].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                min_val = values.min()
                max_val = values.max()
                print(f"{metric_label:<20} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")
    
    print("\n" + "="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot cross-validation results")
    
    # Data arguments
    parser.add_argument("--fold_dir", type=str, default="data",
                        help="Base directory containing fold_X subdirectories")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Base directory containing fold_X checkpoint subdirectories")
    parser.add_argument("--config", type=str, default="data/config.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save plots")
    parser.add_argument("--audio_dir", type=str, default="data/audios_48khz",
                        help="Directory containing source audio files")
    parser.add_argument("--annotations", type=str, default="data/annotations_identification.json",
                        help="Path to annotations JSON file")
    
    # Model arguments
    parser.add_argument("--checkpoint_name", type=str, default=None,
                        help="Specific checkpoint filename (default: best checkpoint)")
    
    # Saved metrics (optional)
    parser.add_argument("--use_saved_metrics", action="store_true",
                        help="Use metrics provided in script rather than recomputing")
    
    args = parser.parse_args()
    
    # Load config
    class_names = None
    if args.config:
        cfg = load_config(args.config)
        spectrograms_root = cfg.paths.spectrograms_dir
        x_col = cfg.training.x_col
        target_size = cfg.training.target_size
        batch_size = cfg.training.batch_size
        num_workers = cfg.training.num_workers
        normalize = cfg.training.normalize
        if hasattr(cfg, 'class_names') and cfg.class_names:
            class_names = [cfg.class_names[k] for k in sorted(cfg.class_names.keys())]
    else:
        spectrograms_root = "data/spectrograms"
        x_col = "spec_name"
        target_size = [224, 469]
        batch_size = 32
        num_workers = 4
        normalize = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all fold directories
    fold_dirs = sorted([d for d in os.listdir(args.fold_dir) if d.startswith('fold_')])
    print(f"Found {len(fold_dirs)} folds: {fold_dirs}")
    
    # Saved metrics from user's training output (optional)
    saved_metrics = None
    if args.use_saved_metrics:
        saved_metrics = {
            0: {'loss': 1.6783, 'f1': 0.7363, 'auprc': 0.7470, 'precision': 0.8190, 'recall': 0.6688, 'acc_neg': 0.9554, 'acc_pos': 0.6688},
            1: {'loss': 1.8658, 'f1': 0.7415, 'auprc': 0.6859, 'precision': 0.6794, 'recall': 0.8163, 'acc_neg': 0.9122, 'acc_pos': 0.8163},
            2: {'loss': 1.4397, 'f1': 0.7437, 'auprc': 0.7444, 'precision': 0.7759, 'recall': 0.7141, 'acc_neg': 0.9479, 'acc_pos': 0.7141},
            3: {'loss': 1.2951, 'f1': 0.7647, 'auprc': 0.7645, 'precision': 0.7666, 'recall': 0.7629, 'acc_neg': 0.9310, 'acc_pos': 0.7629},
            4: {'loss': 6.8471, 'f1': 0.6517, 'auprc': 0.5465, 'precision': 0.5099, 'recall': 0.9028, 'acc_neg': 0.5040, 'acc_pos': 0.9028},
        }
        
        # Calculate overall accuracy from acc_neg and acc_pos
        # We need to estimate class distribution to compute overall accuracy
        # For now, we'll compute it during evaluation
    
    # Evaluate each fold
    fold_results = []
    
    for fold_num, fold_dirname in enumerate(fold_dirs):
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_num}")
        print(f"{'='*60}")
        
        # Find checkpoint
        ckpt_dir = os.path.join(args.checkpoint_dir, f"fold_{fold_num}")
        
        if args.checkpoint_name:
            ckpt_path = os.path.join(ckpt_dir, args.checkpoint_name)
        else:
            # Find best checkpoint (not last.ckpt)
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') and f != 'last.ckpt']
            if not ckpts:
                print(f"Warning: No checkpoint found in {ckpt_dir}, skipping...")
                continue
            ckpt_path = os.path.join(ckpt_dir, ckpts[0])
        
        # Find test CSV
        fold_path = os.path.join(args.fold_dir, fold_dirname)
        test_csv = os.path.join(fold_path, 'test_split.csv')
        
        if not os.path.exists(test_csv):
            print(f"Warning: Test CSV not found at {test_csv}, skipping...")
            continue
        
        # Evaluate fold
        result = evaluate_fold(
            checkpoint_path=ckpt_path,
            test_csv=test_csv,
            spectrograms_root=spectrograms_root,
            x_col=x_col,
            target_size=target_size,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
        )
        
        fold_results.append(result)
        
        # Print fold metrics
        print(f"\nFold {fold_num} Metrics:")
        for key, value in result['metrics'].items():
            if not np.isnan(value):
                print(f"  {key}: {value:.4f}")
    
    # Plot precision-recall curves
    if fold_results and fold_results[0]['is_binary']:
        pr_curve_path = os.path.join(args.output_dir, "precision_recall_curves.png")
        plot_precision_recall_curves(fold_results, pr_curve_path)
    
    # Plot qualitative results
    if fold_results:
        qual_path = os.path.join(args.output_dir, "qualitative_results.png")
        plot_qualitative_results(
            fold_results,
            output_path=qual_path,
            audio_dir=args.audio_dir,
            annotations_path=args.annotations,
            class_names=class_names,
        )
    
    # Print results table
    print_results_table(fold_results, saved_metrics if args.use_saved_metrics else None)
    
    # Save results to CSV
    results_csv_path = os.path.join(args.output_dir, "cv_results.csv")
    df_data = []
    for fold_num, result in enumerate(fold_results):
        row = {'fold': fold_num}
        row.update(result['metrics'])
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(results_csv_path, index=False)
    print(f"Results saved to: {results_csv_path}")


if __name__ == "__main__":
    main()
