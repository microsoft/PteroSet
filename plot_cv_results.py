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
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import pytorch_lightning as pl

# Import from training script
from train import SpectrogramDataModule, DataModuleConfig

# Import from PytorchWildlife core library
from PytorchWildlife.models.bioacoustics import ResNetClassifier
from PytorchWildlife.utils.bioacoustics_configs import load_config


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
    
    # Model arguments
    parser.add_argument("--checkpoint_name", type=str, default=None,
                        help="Specific checkpoint filename (default: best checkpoint)")
    
    # Saved metrics (optional)
    parser.add_argument("--use_saved_metrics", action="store_true",
                        help="Use metrics provided in script rather than recomputing")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        cfg = load_config(args.config)
        spectrograms_root = cfg.paths.spectrograms_dir
        x_col = cfg.training.x_col
        target_size = cfg.training.target_size
        batch_size = cfg.training.batch_size
        num_workers = cfg.training.num_workers
        normalize = cfg.training.normalize
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
