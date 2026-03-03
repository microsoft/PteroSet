"""
Create new fold directories with:
- Train/Val: Using segmented windows (no boundary-crossing, but with overlaps within segments)
- Test: Non-overlapping windows only

Usage:
    python create_segmented_folds.py --windows data/windows_mapping_4.0overlap_segmented.json --config data/config.yaml
"""

import os
import argparse
import json
import csv
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit

from PytorchWildlife.utils.bioacoustics_configs import load_config


def extract_nonoverlapping_windows(windows, window_size_samples):
    """Extract non-overlapping windows (e.g., [0-5s], [5-10s], [10-15s])."""
    by_sound = defaultdict(list)
    for w in windows:
        by_sound[w['sound_id']].append(w)
    
    nonoverlapping = []
    
    for sound_id, sound_windows in by_sound.items():
        sound_windows.sort(key=lambda x: x['start'])
        
        # Find windows that start at multiples of window_size
        for w in sound_windows:
            if w['start'] % window_size_samples == 0:
                nonoverlapping.append(w)
    
    return nonoverlapping


def create_segmented_folds(windows_file, config, output_suffix="_segmented"):
    """Create leave-one-project-out folds with segmented train/val and non-overlapping test."""
    
    print(f"Loading segmented windows from: {windows_file}")
    with open(windows_file, 'r') as f:
        windows = json.load(f)
    
    print(f"Total segmented windows: {len(windows)}")
    
    # Load annotations
    with open(config.paths.annotations_path, 'r') as f:
        annotations = json.load(f)
    sounds = {s['id']: s for s in annotations['sounds']}
    
    # Build data with all needed fields
    from inference import spectrogram_filename
    
    data = []
    for w in windows:
        sound = sounds.get(w['sound_id'])
        if sound:
            spec_name = spectrogram_filename(sound['file_name_path'], w['start'], w['end'])
            sound_filename = os.path.basename(sound['file_name_path'])
            
            data.append({
                'window_id': w['window_id'],
                'sound_id': w['sound_id'],
                'start': w['start'],
                'end': w['end'],
                'label': w.get('label', 0),
                'spec_name': spec_name,
                'sound_filename': sound_filename,
            })
    
    # Load metadata
    metadata_path = os.path.join(config.paths.data_root, 'metadata.csv')
    print(f"Loading metadata from: {metadata_path}")
    
    audio_to_project = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_to_project[row['audio_file']] = row['project_name']
    
    # Add project
    for d in data:
        d['project'] = audio_to_project.get(d['sound_filename'])
    
    data = [d for d in data if d['project'] is not None]
    print(f"Windows with project mapping: {len(data)}")
    
    # Filter to existing spectrograms
    spectrograms_dir = config.paths.spectrograms_dir
    data = [d for d in data if os.path.exists(os.path.join(spectrograms_dir, d['spec_name']))]
    print(f"Windows with existing spectrograms: {len(data)}")
    
    # Get projects
    projects = sorted(set(d['project'] for d in data))
    print(f"\nProjects ({len(projects)}): {projects}")
    
    # Window size for non-overlapping test
    window_size_samples = int(config.audio.window_size_sec * config.audio.sample_rate)
    
    output_dir = config.paths.data_root
    fold_stats = []
    
    for fold_idx, held_out_project in enumerate(projects):
        fold_name = f"fold_{fold_idx}_{held_out_project}{output_suffix}"
        fold_dir = os.path.join(output_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        
        print(f"\n{'-'*60}")
        print(f"Fold {fold_idx}: {held_out_project}")
        print(f"{'-'*60}")
        
        # Test: all from held-out project (with overlaps first, then filter)
        test_data_all = [d for d in data if d['project'] == held_out_project]
        
        # Create non-overlapping test set
        test_data = extract_nonoverlapping_windows(test_data_all, window_size_samples)
        
        print(f"  Test: {len(test_data_all)} → {len(test_data)} (non-overlapping)")
        
        # Train/Val: remaining projects with overlaps (within segments)
        remaining_data = [d for d in data if d['project'] != held_out_project]
        
        X = list(range(len(remaining_data)))
        y = [d['label'] for d in remaining_data]
        groups = [d['sound_id'] for d in remaining_data]
        
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=config.splits.val_size,
            random_state=config.splits.random_state,
        )
        train_idx, val_idx = next(gss.split(X, y, groups=groups))
        
        train_data = [remaining_data[i] for i in train_idx]
        val_data = [remaining_data[i] for i in val_idx]
        
        # Save splits
        fieldnames = ['window_id', 'dataset', 'sample_rate', 'sound_id',
                     'start', 'end', 'label', 'spec_name', 'sound_filename', 'project']
        
        def save_csv(data, filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for d in data:
                    row = {k: d.get(k, '') for k in fieldnames}
                    row['sample_rate'] = config.audio.sample_rate
                    row['dataset'] = ''
                    writer.writerow(row)
        
        save_csv(train_data, os.path.join(fold_dir, 'train_split.csv'))
        save_csv(val_data, os.path.join(fold_dir, 'val_split.csv'))
        save_csv(test_data, os.path.join(fold_dir, 'test_split.csv'))
        
        # Statistics
        print(f"  Train: {len(train_data)} (with overlaps within segments)")
        print(f"  Val:   {len(val_data)} (with overlaps within segments)")
        print(f"  Test:  {len(test_data)} (non-overlapping)")
        
        for name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            label_counts = defaultdict(int)
            proj_counts = defaultdict(int)
            for d in split_data:
                label_counts[d['label']] += 1
                proj_counts[d['project']] += 1
            
            print(f"    {name} labels: {dict(label_counts)}")
            print(f"    {name} projects: {dict(proj_counts)}")
        
        fold_stats.append({
            'fold': fold_name,
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data)
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Created {len(projects)} new fold directories:")
    for stat in fold_stats:
        print(f"  {stat['fold']}")
        print(f"    Train: {stat['train']}, Val: {stat['val']}, Test: {stat['test']}")
    
    print(f"\n✓ New folds use:")
    print(f"  - Train/Val: Segmented windows (no boundary-crossing, with overlaps)")
    print(f"  - Test: Non-overlapping windows only")


def main():
    parser = argparse.ArgumentParser(
        description="Create segmented fold directories for NEW model training"
    )
    parser.add_argument("--windows", required=True,
                       help="Path to segmented windows JSON file")
    parser.add_argument("--config", required=True,
                       help="Path to config YAML file")
    parser.add_argument("--suffix", default="_segmented",
                       help="Suffix for fold directory names")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    create_segmented_folds(
        windows_file=args.windows,
        config=config,
        output_suffix=args.suffix
    )


if __name__ == "__main__":
    main()
