"""
Create non-overlapping test sets for fair model evaluation.

For each fold, takes the test split and keeps only non-overlapping windows
(e.g., [0-5s], [5-10s], [10-15s], etc.) to avoid inflated metrics.

Usage:
    python create_nonoverlapping_test.py --windows data/windows_mapping_4.0overlap_segmented.json --config data/config.yaml
"""

import os
import argparse
import json
import csv
from collections import defaultdict

from PytorchWildlife.utils.bioacoustics_configs import load_config


def extract_nonoverlapping_windows(windows, window_size_samples):
    """
    Extract non-overlapping windows from a list of windows.
    
    For each sound_id, select windows that don't overlap:
    - Start at 0, then window_size, then 2*window_size, etc.
    
    Args:
        windows: List of window dicts with 'sound_id', 'start', 'end'
        window_size_samples: Size of window in samples
    
    Returns:
        List of non-overlapping windows
    """
    # Group by sound_id
    by_sound = defaultdict(list)
    for w in windows:
        by_sound[w['sound_id']].append(w)
    
    nonoverlapping = []
    
    for sound_id, sound_windows in by_sound.items():
        # Sort by start time
        sound_windows.sort(key=lambda x: x['start'])
        
        # Find non-overlapping windows
        expected_starts = set()
        
        # Determine all valid starting points (multiples of window_size)
        for w in sound_windows:
            # Check if this window starts at a multiple of window_size
            if w['start'] % window_size_samples == 0:
                expected_starts.add(w['start'])
        
        # Select windows at those starts
        for w in sound_windows:
            if w['start'] in expected_starts:
                nonoverlapping.append(w)
    
    return nonoverlapping


def process_fold_test_splits(config, windows_file, output_suffix="_nonoverlap_test"):
    """Process all existing fold test splits to create non-overlapping versions."""
    
    print(f"Loading segmented windows from: {windows_file}")
    with open(windows_file, 'r') as f:
        all_windows = json.load(f)
    
    # Create lookup by window_id
    window_lookup = {w['window_id']: w for w in all_windows}
    
    window_size_samples = int(config.audio.window_size_sec * config.audio.sample_rate)
    print(f"Window size: {config.audio.window_size_sec}s = {window_size_samples} samples")
    
    # Find all existing fold directories
    data_root = config.paths.data_root
    fold_dirs = [d for d in os.listdir(data_root) 
                 if os.path.isdir(os.path.join(data_root, d)) and d.startswith('fold_')]
    
    print(f"\nFound {len(fold_dirs)} existing folds")
    
    stats_summary = []
    
    for fold_dir in sorted(fold_dirs):
        fold_path = os.path.join(data_root, fold_dir)
        test_split_path = os.path.join(fold_path, 'test_split.csv')
        
        if not os.path.exists(test_split_path):
            print(f"  Skipping {fold_dir} - no test_split.csv")
            continue
        
        print(f"\n{'-'*60}")
        print(f"Processing: {fold_dir}")
        print(f"{'-'*60}")
        
        # Read original test split
        test_windows = []
        with open(test_split_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_windows.append({
                    'window_id': int(row['window_id']),
                    'sound_id': int(row['sound_id']),
                    'start': int(row['start']),
                    'end': int(row['end']),
                    'label': int(row['label']),
                    'spec_name': row.get('spec_name', ''),
                    'sound_filename': row.get('sound_filename', ''),
                    'project': row.get('project', ''),
                })
        
        print(f"  Original test windows: {len(test_windows)}")
        
        # Extract non-overlapping
        nonoverlap_windows = extract_nonoverlapping_windows(test_windows, window_size_samples)
        
        print(f"  Non-overlapping windows: {len(nonoverlap_windows)}")
        print(f"  Reduction: {100 * (1 - len(nonoverlap_windows)/len(test_windows)):.1f}%")
        
        # Check label distribution
        label_counts_orig = defaultdict(int)
        label_counts_new = defaultdict(int)
        for w in test_windows:
            label_counts_orig[w['label']] += 1
        for w in nonoverlap_windows:
            label_counts_new[w['label']] += 1
        
        print(f"  Original labels: {dict(label_counts_orig)}")
        print(f"  Non-overlap labels: {dict(label_counts_new)}")
        
        # Save new test split
        new_test_path = os.path.join(fold_path, f'test_split{output_suffix}.csv')
        
        fieldnames = ['window_id', 'dataset', 'sample_rate', 'sound_id', 
                     'start', 'end', 'label', 'spec_name', 'sound_filename', 'project']
        
        with open(new_test_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for w in nonoverlap_windows:
                row = {k: w.get(k, '') for k in fieldnames}
                row['sample_rate'] = config.audio.sample_rate
                row['dataset'] = ''
                writer.writerow(row)
        
        print(f"  Saved to: {new_test_path}")
        
        stats_summary.append({
            'fold': fold_dir,
            'original': len(test_windows),
            'nonoverlap': len(nonoverlap_windows),
            'labels_orig': dict(label_counts_orig),
            'labels_new': dict(label_counts_new)
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for stat in stats_summary:
        print(f"{stat['fold']}:")
        print(f"  {stat['original']} → {stat['nonoverlap']} windows")
        print(f"  Labels: {stat['labels_new']}")
    
    print(f"\n✓ Created non-overlapping test splits for {len(stats_summary)} folds")
    print(f"  Suffix: {output_suffix}.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Create non-overlapping test sets for fair evaluation"
    )
    parser.add_argument("--windows", required=True,
                       help="Path to segmented windows JSON file")
    parser.add_argument("--config", required=True,
                       help="Path to config YAML file")
    parser.add_argument("--suffix", default="_nonoverlap",
                       help="Suffix for output test files (default: _nonoverlap)")
    
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    process_fold_test_splits(
        config=config,
        windows_file=args.windows,
        output_suffix=args.suffix
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Test OLD models (current checkpoints) on new test sets")
    print("   → This gives you baseline performance without boundary-crossing artifact")
    print()
    print("2. Create new train/val splits with segmented windows:")
    print("   → python prepare_dataset.py --config data/config.yaml --steps splits")
    print("   → But first update windows file in run_splits() to use segmented version")
    print()
    print("3. Train NEW models with segmented train/val data")
    print()
    print("4. Test NEW models on same non-overlapping test sets")
    print()
    print("5. Compare OLD vs NEW model performance")
    print("="*60)


if __name__ == "__main__":
    main()
