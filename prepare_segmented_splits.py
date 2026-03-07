"""
Create data splits using segmented windows (no boundary-crossing windows).

Usage:
    python prepare_segmented_splits.py --windows data/windows_mapping_4.0overlap_segmented.json --config data/config.yaml
"""

import os
import argparse
import json
from sklearn.model_selection import GroupShuffleSplit

from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config


def create_splits_from_windows(
    windows_file: str,
    config,
    output_suffix: str = "_segmented"
):
    """Create leave-one-project-out splits from a windows file."""
    
    print(f"Loading windows from: {windows_file}")
    with open(windows_file, 'r') as f:
        windows = json.load(f)
    
    print(f"Total windows: {len(windows)}")
    
    # Load annotations to map sound_id -> file path
    with open(config.paths.annotations_path, 'r') as f:
        annotations = json.load(f)
    sounds = {s['id']: s for s in annotations['sounds']}
    
    # Build list of dicts with necessary fields
    from inference import spectrogram_filename
    
    data = []
    for w in windows:
        sound = sounds.get(w['sound_id'])
        if sound:
            spec_name = spectrogram_filename(
                sound['file_name_path'],
                w['start'],
                w['end']
            )
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
    
    print(f"Processed {len(data)} windows")
    
    # Load metadata to get project mapping
    metadata_path = os.path.join(config.paths.data_root, 'metadata.csv')
    print(f"Loading metadata from: {metadata_path}")
    
    import csv
    audio_to_project = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_to_project[row['audio_file']] = row['project_name']
    
    # Add project to each window
    for d in data:
        d['project'] = audio_to_project.get(d['sound_filename'])
    
    # Filter out unmapped
    data = [d for d in data if d['project'] is not None]
    unmapped = len(windows) - len(data)
    if unmapped > 0:
        print(f"Warning: Removed {unmapped} windows without project mapping")
    
    # Filter to existing spectrograms
    spectrograms_dir = config.paths.spectrograms_dir
    data = [d for d in data if os.path.exists(os.path.join(spectrograms_dir, d['spec_name']))]
    print(f"Windows with existing spectrograms: {len(data)}")
    
    # Get unique projects
    projects = sorted(set(d['project'] for d in data))
    print(f"\nProjects ({len(projects)}): {projects}")
    
    # Create splits for each project
    output_dir = config.paths.data_root
    
    for fold_idx, held_out_project in enumerate(projects):
        fold_name = f"fold_{fold_idx}_{held_out_project}{output_suffix}"
        fold_dir = os.path.join(output_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        
        print(f"\n{'-'*50}")
        print(f"Fold {fold_idx}: held-out project = {held_out_project}")
        print(f"{'-'*50}")
        
        # Test: all from held-out project
        test_data = [d for d in data if d['project'] == held_out_project]
        
        # Remaining: split into train/val grouped by sound_id
        remaining_data = [d for d in data if d['project'] != held_out_project]
        
        # Convert to arrays for GroupShuffleSplit
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
        
        # Save as CSV
        def save_csv(data, filepath):
            import csv
            fieldnames = ['window_id', 'dataset', 'sample_rate', 'sound_id', 
                         'start', 'end', 'label', 'spec_name', 'sound_filename', 'project']
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for d in data:
                    row = {k: d.get(k, '') for k in fieldnames}
                    row['sample_rate'] = 48000
                    row['dataset'] = ''
                    writer.writerow(row)
        
        save_csv(train_data, os.path.join(fold_dir, 'train_split.csv'))
        save_csv(val_data, os.path.join(fold_dir, 'val_split.csv'))
        save_csv(test_data, os.path.join(fold_dir, 'test_split.csv'))
        
        # Statistics
        print(f"  Sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        for name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            label_counts = {}
            for d in split_data:
                label = d['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            proj_counts = {}
            for d in split_data:
                proj = d['project']
                proj_counts[proj] = proj_counts.get(proj, 0) + 1
            
            print(f"  {name:5s} labels: {label_counts}")
            print(f"  {name:5s} projects: {proj_counts}")
    
    print(f"\n{'='*60}")
    print(f"Created {len(projects)} folds under: {output_dir}")
    print(f"Fold names have suffix: {output_suffix}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Create splits using segmented windows"
    )
    parser.add_argument("--windows", required=True,
                       help="Path to windows JSON file")
    parser.add_argument("--config", required=True,
                       help="Path to config YAML file")
    parser.add_argument("--suffix", default="_segmented",
                       help="Suffix for fold directory names (default: _segmented)")
    
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    create_splits_from_windows(
        windows_file=args.windows,
        config=config,
        output_suffix=args.suffix
    )


if __name__ == "__main__":
    main()
