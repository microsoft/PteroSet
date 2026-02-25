"""
Data statistics and visualization script.

Generates:
  1. Bar plot of species occurrence counts from annotations_species.json
  2. 3x3 grid of mel spectrogram examples from the most frequent species

Usage:
    python data/data_stats.py
"""

import json
import os
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
from matplotlib.patches import Rectangle

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_PATH = os.path.join(ROOT, "annotations_species.json")
SPECTROGRAMS_DIR = os.path.join(ROOT, "spectrograms")
OUTPUT_DIR = os.path.join(ROOT, "annotations_examples")

# ── Audio / spectrogram config (must match config.yaml) ─────────────────
SR = 48000
HOP_LENGTH = 512
N_MELS = 224
FMAX = SR / 2  # 24 kHz


def load_annotations():
    with open(ANNOTATIONS_PATH, "r") as f:
        data = json.load(f)

    categories = {c["id"]: c for c in data["categories"]}
    sounds = {s["id"]: s for s in data["sounds"]}
    annotations = data["annotations"]
    return categories, sounds, annotations


def spectrogram_filename(sound_path, start_sample, end_sample):
    """Reproduce the naming convention used by prepare_dataset / inference."""
    base = os.path.splitext(os.path.basename(sound_path))[0]
    return f"{base}_{start_sample}_{end_sample}.npy"


def plot_species_bar(annotations, categories, top_n=20, suffix=""):
    """Bar plot of the top N most frequent species."""
    fname = f"species_occurrence{suffix}.png"
    out = os.path.join(OUTPUT_DIR, fname)
    
    if os.path.exists(out):
        print(f"Skipping species bar plot (already exists): {out}")
        return
    
    total_categories = len(categories)
    counts = Counter(ann["category"] for ann in annotations)
    top = counts.most_common(top_n)
    codes, values = zip(*top)

    species_labels = []
    for code in codes:
        cat = next((c for c in categories.values() if c["name"] == code), None)
        species_labels.append(cat["species"] if cat else code)

    fontsize = 9 if top_n <= 20 else 7
    label_fontsize = 8 if top_n <= 20 else 6
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
    y_pos = range(len(codes))
    bars = ax.barh(y_pos, values, color=plt.cm.magma(np.linspace(0.3, 0.85, len(codes))))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{sp}  ({cd})" for sp, cd in zip(species_labels, codes)], fontsize=fontsize)
    ax.invert_yaxis()
    ax.set_xlabel("Number of annotations")
    ax.set_title(f"Top {top_n} Species by Occurrence ({len(annotations)} total annotations, "
                 f"{total_categories} species)")

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.008, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=label_fontsize)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved species bar plot to {out}")


def find_spectrogram_for_annotation(ann, sounds):
    """Find a spectrogram file that contains the given annotation."""
    from matplotlib.patches import Rectangle

    sound = sounds[ann["sound_id"]]
    sound_path = sound["file_name_path"]
    t_min, t_max = ann["t_min"], ann["t_max"]

    # Align to 5-second windows with 1-second steps
    window_samples = int(5.0 * SR)
    step_samples = int(1.0 * SR)
    ann_centre = (t_min + t_max) / 2
    best_start = int((ann_centre - 2.5) * SR)
    best_start = max(0, best_start)
    best_start = round(best_start / step_samples) * step_samples
    end_sample = best_start + window_samples

    spec_name = spectrogram_filename(sound_path, best_start, end_sample)
    spec_path = os.path.join(SPECTROGRAMS_DIR, spec_name)

    if os.path.exists(spec_path):
        return spec_path, best_start, end_sample

    # Fallback: search for any window that overlaps the annotation
    base = os.path.splitext(os.path.basename(sound_path))[0]
    candidates = [f for f in os.listdir(SPECTROGRAMS_DIR) if f.startswith(base)]
    for cand in candidates:
        parts = cand.replace(".npy", "").split("_")
        try:
            ws = int(parts[-2])
            we = int(parts[-1])
        except (ValueError, IndexError):
            continue
        if ws / SR <= t_min and we / SR >= t_max:
            return os.path.join(SPECTROGRAMS_DIR, cand), ws, we

    return None, None, None


def plot_top_species_examples(annotations, categories, sounds):
    """Plot one spectrogram example per each of the top 9 species."""
    out = os.path.join(OUTPUT_DIR, "top_species_examples.png")
    
    if os.path.exists(out):
        print(f"Skipping top species examples (already exists): {out}")
        return
    
    from matplotlib.patches import Rectangle

    counts = Counter(ann["category"] for ann in annotations)
    top9 = counts.most_common(9)

    # Group annotations by species code
    anns_by_species = {}
    for ann in annotations:
        anns_by_species.setdefault(ann["category"], []).append(ann)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, (species_code, count) in zip(axes.flat, top9):
        cat = next((c for c in categories.values() if c["name"] == species_code), None)
        species_name = cat["species"] if cat else species_code

        # Try random annotations until we find one with a spectrogram
        candidates = anns_by_species[species_code].copy()
        random.shuffle(candidates)
        spec_path, best_start, end_sample = None, None, None
        ann = None
        for candidate in candidates:
            spec_path, best_start, end_sample = find_spectrogram_for_annotation(candidate, sounds)
            if spec_path is not None:
                ann = candidate
                break

        if ann is None:
            ax.set_title(f"{species_name}\n(no spectrogram found)", fontsize=8)
            ax.axis("off")
            continue

        sound = sounds[ann["sound_id"]]
        t_min, t_max = ann["t_min"], ann["t_max"]
        f_min, f_max = ann["f_min"], ann["f_max"]

        spec = np.load(spec_path)
        win_start_sec = best_start / SR
        win_end_sec = end_sample / SR

        librosa.display.specshow(
            spec, sr=SR, hop_length=HOP_LENGTH,
            x_axis='time', y_axis='mel',
            fmax=FMAX, cmap='magma', ax=ax
        )

        # Draw annotation rectangle (coordinates relative to window start)
        # t_min, t_max are absolute times; convert to relative
        rect_t_min = t_min - win_start_sec
        rect_t_max = t_max - win_start_sec
        rect = Rectangle(
            (rect_t_min, f_min), rect_t_max - rect_t_min, f_max - f_min,
            linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        audio_name = os.path.basename(sound["file_name_path"])
        ax.set_title(f"{species_name} ({species_code})\n{audio_name}", fontsize=8)

    fig.suptitle("Top 9 Species — One Example Each", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "top_species_examples.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved top species examples to {out}")


def plot_annotation_scenarios(annotations=None, categories=None, sounds=None):
    """Plot 6 examples showing different annotation scenarios.
    
    Uses annotations_identification.json for annotations.
    """
    out = os.path.join(OUTPUT_DIR, "annotation_scenarios.png")
    
    if os.path.exists(out):
        print(f"Skipping annotation scenarios (already exists): {out}")
        return
    
    # Load annotations_identification.json for this plot
    annotations_id_path = os.path.join(ROOT, "annotations_identification.json")
    if not os.path.exists(annotations_id_path):
        print(f"Annotations identification file not found: {annotations_id_path}")
        return
    
    with open(annotations_id_path, 'r') as f:
        data = json.load(f)
    
    categories = {c["id"]: c for c in data["categories"]}
    sounds = {s["id"]: s for s in data["sounds"]}
    annotations = data["annotations"]
    
    print(f"Loaded {len(annotations)} annotations from annotations_identification.json for scenarios plot")
    
    from matplotlib.patches import Rectangle
    
    # Get all spectrogram files
    spec_files = [f for f in os.listdir(SPECTROGRAMS_DIR) if f.endswith('.npy')]
    
    # Build a mapping: spectrogram -> list of annotations
    spec_to_anns = {}
    for spec_file in spec_files:
        parts = spec_file.replace('.npy', '').split('_')
        try:
            start_sample = int(parts[-2])
            end_sample = int(parts[-1])
        except (ValueError, IndexError):
            continue
        
        # Find the sound file this spectrogram belongs to
        base_name = '_'.join(parts[:-2])
        matching_sound = None
        for sound in sounds.values():
            if os.path.splitext(os.path.basename(sound['file_name_path']))[0] == base_name:
                matching_sound = sound
                break
        
        if matching_sound is None:
            continue
        
        win_start_sec = start_sample / SR
        win_end_sec = end_sample / SR
        
        # Find annotations that overlap this window
        overlapping_anns = []
        for ann in annotations:
            if ann['sound_id'] == matching_sound['id']:
                t_min, t_max = ann['t_min'], ann['t_max']
                # Check if annotation overlaps window
                if not (t_max < win_start_sec or t_min > win_end_sec):
                    overlapping_anns.append(ann)
        
        spec_to_anns[spec_file] = {
            'start_sample': start_sample,
            'end_sample': end_sample,
            'sound': matching_sound,
            'annotations': overlapping_anns
        }
    
    # Find examples for each scenario
    examples = {}
    
    # 1. No annotations (no birds) - preferably at 350-355 seconds
    for spec_file, info in spec_to_anns.items():
        if len(info['annotations']) == 0:
            win_start = info['start_sample'] / SR
            win_end = info['end_sample'] / SR
            # Try to find window at 350-355 seconds
            if abs(win_start - 350.0) < 1.0 and abs(win_end - 355.0) < 1.0:
                examples['no_birds'] = (spec_file, info)
                break
    
    # Fallback: any window without annotations and away from merge boundaries
    if 'no_birds' not in examples:
        for spec_file, info in spec_to_anns.items():
            if len(info['annotations']) == 0:
                win_start = info['start_sample'] / SR
                win_end = info['end_sample'] / SR
                # Avoid merge boundaries
                is_at_boundary = False
                for minute_mark in range(1, 100):
                    boundary_time = minute_mark * 60
                    if abs(win_start - (boundary_time - 3)) < 5 or abs(win_end - (boundary_time + 3)) < 5:
                        is_at_boundary = True
                        break
                
                if not is_at_boundary:
                    examples['no_birds'] = (spec_file, info)
                    break
    
    # 2. One annotation totally inside window
    for spec_file, info in spec_to_anns.items():
        if len(info['annotations']) == 1:
            ann = info['annotations'][0]
            win_start = info['start_sample'] / SR
            win_end = info['end_sample'] / SR
            if ann['t_min'] >= win_start and ann['t_max'] <= win_end:
                examples['one_inside'] = (spec_file, info)
                break
    
    # 3. One annotation partially in window
    for spec_file, info in spec_to_anns.items():
        if len(info['annotations']) == 1:
            ann = info['annotations'][0]
            win_start = info['start_sample'] / SR
            win_end = info['end_sample'] / SR
            # Partially overlapping: starts before or ends after
            if (ann['t_min'] < win_start and ann['t_max'] > win_start and ann['t_max'] < win_end) or \
               (ann['t_min'] > win_start and ann['t_min'] < win_end and ann['t_max'] > win_end):
                examples['one_partial'] = (spec_file, info)
                break
    
    # 4. One annotation bigger than window (spans entire window)
    for spec_file, info in spec_to_anns.items():
        if len(info['annotations']) == 1:
            ann = info['annotations'][0]
            win_start = info['start_sample'] / SR
            win_end = info['end_sample'] / SR
            if ann['t_min'] < win_start and ann['t_max'] > win_end:
                examples['one_spanning'] = (spec_file, info)
                break
    
    # 5. Multiple annotations
    for spec_file, info in spec_to_anns.items():
        if len(info['annotations']) >= 2:
            examples['multiple'] = (spec_file, info)
            break
    
    # 6. Nested annotations (one annotation bounding box completely contains another)
    for spec_file, info in spec_to_anns.items():
        if len(info['annotations']) >= 2:
            anns = info['annotations']
            # Check if any annotation bounding box contains another (both time and frequency)
            found_nested = False
            for i, ann1 in enumerate(anns):
                for j, ann2 in enumerate(anns):
                    if i != j:
                        # Check if ann1 completely contains ann2 (both time and frequency)
                        time_contains = ann1['t_min'] <= ann2['t_min'] and ann1['t_max'] >= ann2['t_max']
                        freq_contains = ann1['f_min'] <= ann2['f_min'] and ann1['f_max'] >= ann2['f_max']
                        if time_contains and freq_contains:
                            found_nested = True
                            break
                if found_nested:
                    break
            
            if found_nested:
                examples['nested'] = (spec_file, info)
                break
    
    # Plot the examples
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    scenario_names = [
        ('no_birds', 'No Annotations', '0 (no birds)'),
        ('one_inside', 'One Annotation Inside', '1 (birds)'),
        ('one_partial', 'One Annotation Partial', '1 (birds)'),
        ('one_spanning', 'One Annotation Spanning', '1 (birds)'),
        ('multiple', 'Multiple Annotations', '1 (birds)'),
        ('nested', 'Nested Annotations', '1 (birds)')
    ]
    
    for idx, (key, title, label) in enumerate(scenario_names):
        ax = axes[idx]
        
        if key not in examples:
            ax.set_title(f"{title}\n(Example not found)", fontsize=10)
            ax.axis('off')
            continue
        
        spec_file, info = examples[key]
        spec_path = os.path.join(SPECTROGRAMS_DIR, spec_file)
        spec = np.load(spec_path)
        
        win_start_sec = info['start_sample'] / SR
        win_end_sec = info['end_sample'] / SR
        
        librosa.display.specshow(
            spec, sr=SR, hop_length=HOP_LENGTH,
            x_axis='time', y_axis='mel',
            fmax=FMAX, cmap='magma', ax=ax
        )
        
        # Draw all annotations for this window
        for ann in info['annotations']:
            t_min, t_max = ann['t_min'], ann['t_max']
            f_min, f_max = ann['f_min'], ann['f_max']
            
            # Convert to relative coordinates
            rect_t_min = t_min - win_start_sec
            rect_t_max = t_max - win_start_sec
            
            rect = Rectangle(
                (rect_t_min, f_min), rect_t_max - rect_t_min, f_max - f_min,
                linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)
        
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Frequency (Hz)", fontsize=8)
        
        audio_name = os.path.basename(info['sound']['file_name_path'])
        n_anns = len(info['annotations'])
        ax.set_title(f"{title}\nLabel: {label} | {n_anns} annotation(s)\n{audio_name} [{win_start_sec:.1f}s - {win_end_sec:.1f}s]", 
                     fontsize=9)
    
    fig.suptitle("Annotation Scenarios for Model Training", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved annotation scenarios to {out}")


def plot_complete_audio(annotations, categories, sounds, sound_id=None):
    """Plot complete audio file as spectrogram with all annotations overlaid.
    
    Args:
        annotations: List of all annotations
        categories: Dictionary of categories
        sounds: Dictionary of sounds
        sound_id: Specific sound ID to plot. If None, picks the first sound with annotations
    """
    import librosa
    import soundfile as sf
    from matplotlib.patches import Rectangle
    
    # Select a sound to plot
    if sound_id is None:
        # Find a sound that has annotations
        sound_ann_counts = Counter(ann['sound_id'] for ann in annotations)
        if not sound_ann_counts:
            print("No annotations found to plot complete audio")
            return
        sound_id, _ = sound_ann_counts.most_common(1)[0]
    
    sound = sounds.get(sound_id)
    if sound is None:
        print(f"Sound ID {sound_id} not found")
        return
    
    audio_path = sound['file_name_path']
    audio_name = os.path.basename(audio_path)
    
    # Resolve audio path - it might be relative to project root
    if not os.path.isabs(audio_path):
        # Try relative to ROOT (current data directory)
        audio_path_candidate = os.path.join(ROOT, audio_path)
        if not os.path.exists(audio_path_candidate):
            # Try relative to parent directory (birds_bioacoustics/)
            audio_path_candidate = os.path.join(os.path.dirname(ROOT), audio_path)
        if not os.path.exists(audio_path_candidate):
            # Try treating path as if it starts with 'data/' and we're already in data/
            if audio_path.startswith('data/'):
                audio_path_candidate = os.path.join(ROOT, audio_path[5:])  # Remove 'data/' prefix
        audio_path = audio_path_candidate
    
    # Check if output already exists
    safe_name = audio_name.replace('.', '_').replace('/', '_')
    out = os.path.join(OUTPUT_DIR, f"complete_audio_{safe_name}.png")
    
    if os.path.exists(out):
        print(f"Skipping complete audio plot (already exists): {out}")
        return
    
    print(f"Plotting complete audio: {audio_name}")
    
    # Load audio file
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        print(f"Searched in: {audio_path}")
        return
    
    try:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Take first channel if stereo
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    # Resample if needed
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    
    # Compute mel spectrogram for the complete audio
    import librosa
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=2048, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Get all annotations for this sound
    sound_annotations = [ann for ann in annotations if ann['sound_id'] == sound_id]
    
    # Create figure
    duration = len(audio) / SR
    fig_width = max(15, duration / 10)  # Scale width with duration
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    # Plot spectrogram
    librosa.display.specshow(
        mel_spec_db, sr=SR, hop_length=HOP_LENGTH,
        x_axis='time', y_axis='mel',
        fmax=FMAX, cmap='magma', ax=ax
    )
    
    # Overlay all annotations
    for ann in sound_annotations:
        t_min, t_max = ann['t_min'], ann['t_max']
        f_min, f_max = ann['f_min'], ann['f_max']
        
        rect = Rectangle(
            (t_min, f_min), t_max - t_min, f_max - f_min,
            linewidth=1.0, edgecolor="cyan", facecolor="none", linestyle="--", alpha=0.8
        )
        ax.add_patch(rect)
    
    # Add colorbar
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB', label='Power (dB)')
    
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)", fontsize=10)
    ax.set_title(f"Complete Audio: {audio_name}\n"
                 f"Duration: {duration:.1f}s | Annotations: {len(sound_annotations)}", 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved complete audio plot to {out}")


def plot_audios_by_project_time_of_day(annotations=None, categories=None, sounds=None):
    """Plot one audio per project with shared x-axis showing time of day.
    
    Each audio is 480 seconds (8 minutes) representing a full day:
    10 seconds recorded every 30 minutes, stitched together.
    X-axis shows actual time of day (00:00 to 24:00).
    
    Uses annotations_identification.json for annotations.
    """
    import librosa
    import soundfile as sf
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter
    import pandas as pd
    
    out = os.path.join(OUTPUT_DIR, "audios_by_project_time_of_day.png")
    
    if os.path.exists(out):
        print(f"Skipping project audio plots (already exists): {out}")
        return
    
    # Load annotations_identification.json for this plot
    annotations_id_path = os.path.join(ROOT, "annotations_identification.json")
    if not os.path.exists(annotations_id_path):
        print(f"Annotations identification file not found: {annotations_id_path}")
        return
    
    with open(annotations_id_path, 'r') as f:
        data = json.load(f)
    
    categories = {c["id"]: c for c in data["categories"]}
    sounds = {s["id"]: s for s in data["sounds"]}
    annotations = data["annotations"]
    
    print(f"Loaded {len(annotations)} annotations from annotations_identification.json")
    
    # Load metadata to get project information
    metadata_path = os.path.join(ROOT, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return
    
    metadata = pd.read_csv(metadata_path)
    
    # Group by project and audio file
    project_audio_map = {}
    for _, row in metadata.iterrows():
        project = row['project_name']
        audio_file = row['audio_file']
        
        if project not in project_audio_map:
            project_audio_map[project] = []
        project_audio_map[project].append(audio_file)
    
    # For each project, find the sound with most annotations
    selected_sounds = {}
    for project, audio_files in project_audio_map.items():
        best_sound = None
        max_anns = -1
        best_audio_file = None
        
        for audio_file in audio_files:
            # Find matching sound by filename
            for sound_id, sound in sounds.items():
                sound_basename = os.path.basename(sound['file_name_path'])
                if sound_basename == audio_file:
                    n_anns = sum(1 for ann in annotations if ann['sound_id'] == sound_id)
                    if n_anns > max_anns:
                        max_anns = n_anns
                        best_sound = (sound_id, sound)
                        best_audio_file = audio_file
                    break
        
        if best_sound:
            selected_sounds[project] = (best_sound, best_audio_file, max_anns)
    
    if len(selected_sounds) == 0:
        print("No sounds found to plot by project")
        return
    
    # Sort projects chronologically (PAREX first if exists, then by name which includes year)
    def sort_key(item):
        project_name = item[0]
        if 'PAREX' in project_name.upper():
            return '0_' + project_name  # Ensure PAREX comes first
        return project_name
    
    sorted_projects = sorted(selected_sounds.items(), key=sort_key)
    
    print(f"Plotting {len(sorted_projects)} projects with time-of-day x-axis")
    for proj, (sound_info, audio_file, n_anns) in sorted_projects:
        print(f"  - {proj}: {audio_file} ({n_anns} annotations)")
    
    # Create subplots
    n_projects = len(sorted_projects)
    fig, axes = plt.subplots(n_projects, 1, figsize=(16, 3 * n_projects), sharex=True)
    
    if n_projects == 1:
        axes = [axes]
    
    # Conversion: each second in audio represents 3 minutes in real time
    # (480 seconds audio represents 24 hours = 1440 minutes)
    def audio_seconds_to_hours(t_seconds):
        """Convert audio time (seconds) to time of day (hours from midnight)"""
        return t_seconds * 3.0 / 60.0  # Each second = 3 minutes = 0.05 hours
    
    def format_time_of_day(seconds, pos=None):
        """Format audio seconds as HH:MM time of day"""
        hours_float = audio_seconds_to_hours(seconds)
        hours = int(hours_float)
        mins = int((hours_float % 1) * 60)
        return f"{hours:02d}:{mins:02d}"
    
    for idx, (project, (sound_info, audio_file, n_anns)) in enumerate(sorted_projects):
        ax = axes[idx]
        sound_id, sound = sound_info
        ax = axes[idx]
        
        audio_path = sound['file_name_path']
        audio_name = os.path.basename(audio_path)
        
        # Resolve audio path
        if not os.path.isabs(audio_path):
            if audio_path.startswith('data/'):
                audio_path = os.path.join(ROOT, audio_path[5:])
            else:
                audio_path = os.path.join(ROOT, audio_path)
        
        if not os.path.exists(audio_path):
            ax.text(0.5, 0.5, f"Audio file not found:\n{audio_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(f"{project}", fontsize=10, fontweight='bold')
            continue
        
        # Load and process audio
        try:
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            if sr != SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=SR, n_fft=2048, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmax=FMAX
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Get annotations for this sound
            sound_annotations = [ann for ann in annotations if ann['sound_id'] == sound_id]
            
            # Plot spectrogram with librosa (same style as complete_audio)
            librosa.display.specshow(
                mel_spec_db, sr=SR, hop_length=HOP_LENGTH,
                x_axis='time', y_axis='mel',
                fmax=FMAX, cmap='magma', ax=ax
            )
            
            # Overlay annotations (in seconds to match spectrogram coordinates)
            for ann in sound_annotations:
                t_min_audio = ann['t_min']  # Time in audio seconds
                t_max_audio = ann['t_max']
                f_min, f_max = ann['f_min'], ann['f_max']
                
                # Keep coordinates in seconds (matching spectrogram)
                rect = Rectangle(
                    (t_min_audio, f_min), t_max_audio - t_min_audio, f_max - f_min,
                    linewidth=1.0, edgecolor="cyan", facecolor="none", linestyle="--", alpha=0.8
                )
                ax.add_patch(rect)
            
            # Format axes
            ax.set_ylabel(f"{project}\nFrequency (Hz)", fontsize=9, fontweight='bold')
            ax.set_title(f"{audio_name} | {len(sound_annotations)} annotations", 
                        fontsize=9, loc='right')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading audio:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(f"{project}", fontsize=10, fontweight='bold')
    
    # Set shared x-axis with time of day labels
    # Audio is 480 seconds representing 24 hours (each second = 3 minutes)
    axes[-1].set_xlabel("Time of Day", fontsize=11, fontweight='bold')
    
    # Set x-axis limits and ticks for all axes
    tick_positions = list(range(0, 481, 10))  # 0, 10, 20, ..., 480 seconds (every 30 min)
    
    for ax in axes:
        ax.set_xlim(0, 480)
        ax.set_xticks(tick_positions)
        ax.xaxis.set_major_formatter(FuncFormatter(format_time_of_day))
    
    # Rotate labels for readability (only on bottom axis)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    fig.suptitle("Audio Spectrograms by Project - Time of Day (10s every 30 min)", 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved project audio plots to {out}")


def print_data_statistics():
    """Print comprehensive statistics about the dataset."""
    
    print("\n" + "="*80)
    print("DATA STATISTICS OVERVIEW")
    print("="*80 + "\n")
    
    # Load annotations at species level
    with open(ANNOTATIONS_PATH, "r") as f:
        data_species = json.load(f)
    categories_species = {c["id"]: c for c in data_species["categories"]}
    sounds_species = {s["id"]: s for s in data_species["sounds"]}
    annotations_species = data_species["annotations"]
    
    # Load annotations at identification level
    annotations_id_path = os.path.join(ROOT, "annotations_identification.json")
    with open(annotations_id_path, "r") as f:
        data_id = json.load(f)
    annotations_identification = data_id["annotations"]
    
    # Load metadata
    metadata_path = os.path.join(ROOT, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    
    # ── 1. AUDIO DURATION STATISTICS ──
    print("1. AUDIO DURATION STATISTICS")
    print("-" * 80)
    
    durations = [sound["duration"] for sound in sounds_species.values()]
    
    print(f"  Total number of audios: {len(durations)}")
    print(f"  Min duration:  {np.min(durations):.2f} seconds ({np.min(durations)/60:.2f} minutes)")
    print(f"  Max duration:  {np.max(durations):.2f} seconds ({np.max(durations)/60:.2f} minutes)")
    print(f"  Mean duration: {np.mean(durations):.2f} seconds ({np.mean(durations)/60:.2f} minutes)")
    print(f"  Std duration:  {np.std(durations):.2f} seconds ({np.std(durations)/60:.2f} minutes)")
    print(f"  Total duration: {np.sum(durations):.2f} seconds ({np.sum(durations)/3600:.2f} hours)")
    print()
    
    # ── 2. AUDIOS PER PROJECT ──
    print("2. AUDIOS PER PROJECT")
    print("-" * 80)
    
    project_counts = metadata['project_name'].value_counts().sort_index()
    
    for project, count in project_counts.items():
        print(f"  {project}: {count} audios")
    print(f"  Total projects: {len(project_counts)}")
    print()
    
    # ── 3. ANNOTATIONS PER AUDIO (SPECIES LEVEL) ──
    print("3. ANNOTATIONS PER AUDIO (SPECIES LEVEL)")
    print("-" * 80)
    
    # Count annotations per sound_id
    anns_per_sound_species = Counter(ann["sound_id"] for ann in annotations_species)
    ann_counts_species = list(anns_per_sound_species.values())
    
    # Include sounds with 0 annotations
    num_sounds_no_anns_species = len(sounds_species) - len(anns_per_sound_species)
    if num_sounds_no_anns_species > 0:
        ann_counts_species.extend([0] * num_sounds_no_anns_species)
    
    print(f"  Total annotations (species level): {len(annotations_species)}")
    print(f"  Total audios: {len(sounds_species)}")
    print(f"  Audios with annotations: {len(anns_per_sound_species)}")
    print(f"  Audios without annotations: {num_sounds_no_anns_species}")
    print(f"  Min annotations per audio:  {np.min(ann_counts_species):.0f}")
    print(f"  Max annotations per audio:  {np.max(ann_counts_species):.0f}")
    print(f"  Mean annotations per audio: {np.mean(ann_counts_species):.2f}")
    print(f"  Std annotations per audio:  {np.std(ann_counts_species):.2f}")
    print()
    
    # ── 4. ANNOTATIONS PER AUDIO (IDENTIFICATION LEVEL) ──
    print("4. ANNOTATIONS PER AUDIO (IDENTIFICATION LEVEL)")
    print("-" * 80)
    
    # Count annotations per sound_id at identification level
    anns_per_sound_id = Counter(ann["sound_id"] for ann in annotations_identification)
    ann_counts_id = list(anns_per_sound_id.values())
    
    # Include sounds with 0 annotations
    num_sounds_no_anns_id = len(sounds_species) - len(anns_per_sound_id)
    if num_sounds_no_anns_id > 0:
        ann_counts_id.extend([0] * num_sounds_no_anns_id)
    
    print(f"  Total annotations (identification level): {len(annotations_identification)}")
    print(f"  Total audios: {len(sounds_species)}")
    print(f"  Audios with annotations: {len(anns_per_sound_id)}")
    print(f"  Audios without annotations: {num_sounds_no_anns_id}")
    print(f"  Min annotations per audio:  {np.min(ann_counts_id):.0f}")
    print(f"  Max annotations per audio:  {np.max(ann_counts_id):.0f}")
    print(f"  Mean annotations per audio: {np.mean(ann_counts_id):.2f}")
    print(f"  Std annotations per audio:  {np.std(ann_counts_id):.2f}")
    print()
    
    # ── 5. ANNOTATION DURATION STATISTICS (SPECIES LEVEL) ──
    print("5. ANNOTATION DURATION STATISTICS (SPECIES LEVEL)")
    print("-" * 80)
    
    ann_durations_species = [ann["t_max"] - ann["t_min"] for ann in annotations_species]
    
    print(f"  Total annotations: {len(ann_durations_species)}")
    print(f"  Min duration:  {np.min(ann_durations_species):.3f} seconds")
    print(f"  Max duration:  {np.max(ann_durations_species):.3f} seconds")
    print(f"  Mean duration: {np.mean(ann_durations_species):.3f} seconds")
    print(f"  Std duration:  {np.std(ann_durations_species):.3f} seconds")
    print(f"  Median duration: {np.median(ann_durations_species):.3f} seconds")
    print()
    
    # ── 6. ANNOTATION DURATION STATISTICS (IDENTIFICATION LEVEL) ──
    print("6. ANNOTATION DURATION STATISTICS (IDENTIFICATION LEVEL)")
    print("-" * 80)
    
    ann_durations_id = [ann["t_max"] - ann["t_min"] for ann in annotations_identification]
    
    print(f"  Total annotations: {len(ann_durations_id)}")
    print(f"  Min duration:  {np.min(ann_durations_id):.3f} seconds")
    print(f"  Max duration:  {np.max(ann_durations_id):.3f} seconds")
    print(f"  Mean duration: {np.mean(ann_durations_id):.3f} seconds")
    print(f"  Std duration:  {np.std(ann_durations_id):.3f} seconds")
    print(f"  Median duration: {np.median(ann_durations_id):.3f} seconds")
    print()
    
    # ── 7. SPECIES DIVERSITY ──
    print("7. SPECIES DIVERSITY")
    print("-" * 80)
    
    print(f"  Total unique species: {len(categories_species)}")
    
    species_counts = Counter(ann["category"] for ann in annotations_species)
    print(f"  Most common species:")
    for species_code, count in species_counts.most_common(10):
        cat = next((c for c in categories_species.values() if c["name"] == species_code), None)
        species_name = cat["species"] if cat else species_code
        print(f"    {species_name} ({species_code}): {count} annotations")
    
    print()
    print("="*80)
    print("END OF STATISTICS")
    print("="*80 + "\n")


def print_statistics_by_project():
    """Print and plot comprehensive statistics grouped by project."""
    
    print("\n" + "="*80)
    print("STATISTICS BY PROJECT")
    print("="*80 + "\n")
    
    # Load annotations at species level
    with open(ANNOTATIONS_PATH, "r") as f:
        data_species = json.load(f)
    categories_species = {c["id"]: c for c in data_species["categories"]}
    sounds_species = {s["id"]: s for s in data_species["sounds"]}
    annotations_species = data_species["annotations"]
    
    # Load annotations at identification level
    annotations_id_path = os.path.join(ROOT, "annotations_identification.json")
    with open(annotations_id_path, "r") as f:
        data_id = json.load(f)
    annotations_identification = data_id["annotations"]
    
    # Load metadata
    metadata_path = os.path.join(ROOT, "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    
    # Create mapping: audio_file -> project_name
    audio_to_project = {}
    for _, row in metadata.iterrows():
        audio_to_project[row['audio_file']] = row['project_name']
    
    # Create mapping: sound_id -> project_name
    sound_id_to_project = {}
    for sound_id, sound in sounds_species.items():
        audio_file = os.path.basename(sound['file_name_path'])
        if audio_file in audio_to_project:
            sound_id_to_project[sound_id] = audio_to_project[audio_file]
    
    # Get unique projects (sorted with PAREX first)
    def sort_projects(proj):
        if 'PAREX' in proj.upper():
            return '0_' + proj  # PAREX comes first
        return '1_' + proj
    
    projects = sorted(set(audio_to_project.values()), key=sort_projects)
    
    # Initialize statistics storage
    project_stats = {proj: {
        'num_audios': 0,
        'audio_durations': [],
        'num_anns_species': 0,
        'num_anns_identification': 0,
        'ann_durations_species': [],
        'ann_durations_identification': [],
        'species_set': set(),
        'anns_per_audio_species': [],
        'anns_per_audio_identification': []
    } for proj in projects}
    
    # Collect audio statistics by project
    for sound_id, sound in sounds_species.items():
        project = sound_id_to_project.get(sound_id)
        if project:
            project_stats[project]['num_audios'] += 1
            project_stats[project]['audio_durations'].append(sound['duration'])
    
    # Collect annotation statistics (species level)
    anns_per_sound_species = {proj: Counter() for proj in projects}
    for ann in annotations_species:
        project = sound_id_to_project.get(ann['sound_id'])
        if project:
            project_stats[project]['num_anns_species'] += 1
            project_stats[project]['ann_durations_species'].append(ann['t_max'] - ann['t_min'])
            project_stats[project]['species_set'].add(ann['category'])
            anns_per_sound_species[project][ann['sound_id']] += 1
    
    # Convert annotation counts to lists
    for proj in projects:
        project_stats[proj]['anns_per_audio_species'] = list(anns_per_sound_species[proj].values())
    
    # Collect annotation statistics (identification level)
    anns_per_sound_id = {proj: Counter() for proj in projects}
    for ann in annotations_identification:
        project = sound_id_to_project.get(ann['sound_id'])
        if project:
            project_stats[project]['num_anns_identification'] += 1
            project_stats[project]['ann_durations_identification'].append(ann['t_max'] - ann['t_min'])
            anns_per_sound_id[project][ann['sound_id']] += 1
    
    # Convert annotation counts to lists
    for proj in projects:
        project_stats[proj]['anns_per_audio_identification'] = list(anns_per_sound_id[proj].values())
    
    # Print statistics by project
    for project in projects:
        stats = project_stats[project]
        
        print(f"PROJECT: {project}")
        print("-" * 80)
        
        # Audio statistics
        print(f"  Number of audios: {stats['num_audios']}")
        if stats['audio_durations']:
            print(f"  Audio duration - Min: {np.min(stats['audio_durations']):.2f}s, "
                  f"Max: {np.max(stats['audio_durations']):.2f}s, "
                  f"Mean: {np.mean(stats['audio_durations']):.2f}s, "
                  f"Std: {np.std(stats['audio_durations']):.2f}s")
            print(f"  Total audio duration: {np.sum(stats['audio_durations'])/3600:.2f} hours")
        
        # Species-level annotations
        print(f"  Annotations (species): {stats['num_anns_species']}")
        if stats['anns_per_audio_species']:
            print(f"  Annotations per audio (species) - Min: {np.min(stats['anns_per_audio_species']):.0f}, "
                  f"Max: {np.max(stats['anns_per_audio_species']):.0f}, "
                  f"Mean: {np.mean(stats['anns_per_audio_species']):.2f}, "
                  f"Std: {np.std(stats['anns_per_audio_species']):.2f}")
        
        if stats['ann_durations_species']:
            print(f"  Annotation duration (species) - Min: {np.min(stats['ann_durations_species']):.3f}s, "
                  f"Max: {np.max(stats['ann_durations_species']):.3f}s, "
                  f"Mean: {np.mean(stats['ann_durations_species']):.3f}s")
        
        # Identification-level annotations
        print(f"  Annotations (identification): {stats['num_anns_identification']}")
        if stats['anns_per_audio_identification']:
            print(f"  Annotations per audio (identification) - Min: {np.min(stats['anns_per_audio_identification']):.0f}, "
                  f"Max: {np.max(stats['anns_per_audio_identification']):.0f}, "
                  f"Mean: {np.mean(stats['anns_per_audio_identification']):.2f}, "
                  f"Std: {np.std(stats['anns_per_audio_identification']):.2f}")
        
        if stats['ann_durations_identification']:
            print(f"  Annotation duration (identification) - Min: {np.min(stats['ann_durations_identification']):.3f}s, "
                  f"Max: {np.max(stats['ann_durations_identification']):.3f}s, "
                  f"Mean: {np.mean(stats['ann_durations_identification']):.3f}s")
        
        # Species diversity
        print(f"  Unique species: {len(stats['species_set'])}")
        
        print()
    
    print("="*80 + "\n")
    
    # Create visualizations
    plot_statistics_by_project(projects, project_stats)


def plot_statistics_by_project(projects, project_stats):
    """Create comprehensive plots showing statistics by project."""
    
    out = os.path.join(OUTPUT_DIR, "statistics_by_project.png")
    
    if os.path.exists(out):
        print(f"Skipping project statistics plot (already exists): {out}")
        return
    
    # Prepare data for plotting
    num_projects = len(projects)
    
    # Create figure with subplots (3 rows x 2 cols = 6 subplots)
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35)
    
    # 1. Number of audios per project
    ax1 = fig.add_subplot(gs[0, 0])
    num_audios = [project_stats[proj]['num_audios'] for proj in projects]
    colors = plt.cm.magma(np.linspace(0.2, 0.85, num_projects))
    bars = ax1.barh(projects, num_audios, color=colors)
    ax1.set_xlabel('Number of Audios', fontsize=12, fontweight='bold')
    ax1.set_title('Number of Audios per Project', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=10)
    ax1.invert_yaxis()
    for bar, val in zip(bars, num_audios):
        ax1.text(bar.get_width() + max(num_audios)*0.01, bar.get_y() + bar.get_height()/2, 
                 str(val), va='center', fontsize=10)
    
    # 2. Total audio duration per project (hours)
    ax2 = fig.add_subplot(gs[0, 1])
    total_durations = [np.sum(project_stats[proj]['audio_durations'])/3600 for proj in projects]
    bars = ax2.barh(projects, total_durations, color=colors)
    ax2.set_xlabel('Total Duration (hours)', fontsize=12, fontweight='bold')
    ax2.set_title('Total Audio Duration per Project', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=10)
    ax2.invert_yaxis()
    for bar, val in zip(bars, total_durations):
        ax2.text(bar.get_width() + max(total_durations)*0.01, bar.get_y() + bar.get_height()/2, 
                 f'{val:.2f}h', va='center', fontsize=10)
    
    # 3. Number of species-level annotations per project
    ax3 = fig.add_subplot(gs[1, 0])
    num_anns_species = [project_stats[proj]['num_anns_species'] for proj in projects]
    bars = ax3.barh(projects, num_anns_species, color=colors)
    ax3.set_xlabel('Number of Annotations', fontsize=12, fontweight='bold')
    ax3.set_title('Species-Level Annotations per Project', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=10)
    ax3.invert_yaxis()
    for bar, val in zip(bars, num_anns_species):
        ax3.text(bar.get_width() + max(num_anns_species)*0.01, bar.get_y() + bar.get_height()/2, 
                 str(val), va='center', fontsize=10)
    
    # 4. Number of identification-level annotations per project
    ax4 = fig.add_subplot(gs[1, 1])
    num_anns_id = [project_stats[proj]['num_anns_identification'] for proj in projects]
    bars = ax4.barh(projects, num_anns_id, color=colors)
    ax4.set_xlabel('Number of Annotations', fontsize=12, fontweight='bold')
    ax4.set_title('Identification-Level Annotations per Project', fontsize=13, fontweight='bold')
    ax4.tick_params(axis='both', labelsize=10)
    ax4.invert_yaxis()
    for bar, val in zip(bars, num_anns_id):
        ax4.text(bar.get_width() + max(num_anns_id)*0.01, bar.get_y() + bar.get_height()/2, 
                 str(val), va='center', fontsize=10)
    
    # 5. Mean annotation duration (identification level)
    ax5 = fig.add_subplot(gs[2, 0])
    mean_ann_dur_id = [np.mean(project_stats[proj]['ann_durations_identification']) 
                       if project_stats[proj]['ann_durations_identification'] else 0 
                       for proj in projects]
    bars = ax5.barh(projects, mean_ann_dur_id, color=colors)
    ax5.set_xlabel('Mean Duration (seconds)', fontsize=12, fontweight='bold')
    ax5.set_title('Mean Annotation Duration by Project', fontsize=13, fontweight='bold')
    ax5.tick_params(axis='both', labelsize=10)
    ax5.invert_yaxis()
    for bar, val in zip(bars, mean_ann_dur_id):
        ax5.text(bar.get_width() + max(mean_ann_dur_id)*0.01, bar.get_y() + bar.get_height()/2, 
                 f'{val:.2f}s', va='center', fontsize=10)
    
    # 6. Number of unique species per project
    ax6 = fig.add_subplot(gs[2, 1])
    num_species = [len(project_stats[proj]['species_set']) for proj in projects]
    bars = ax6.barh(projects, num_species, color=colors)
    ax6.set_xlabel('Number of Unique Species', fontsize=12, fontweight='bold')
    ax6.set_title('Species Diversity per Project', fontsize=13, fontweight='bold')
    ax6.tick_params(axis='both', labelsize=10)
    ax6.invert_yaxis()
    for bar, val in zip(bars, num_species):
        ax6.text(bar.get_width() + max(num_species)*0.01, bar.get_y() + bar.get_height()/2, 
                 str(val), va='center', fontsize=10)
    
    fig.suptitle('Dataset Statistics by Project', fontsize=18, fontweight='bold')
    
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved project statistics plot to {out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Print comprehensive data statistics
    print_data_statistics()
    
    # Print and plot statistics by project
    print_statistics_by_project()

    categories, sounds, annotations = load_annotations()

    print(f"Total annotations: {len(annotations)}")
    print(f"Total species: {len(categories)}")
    print(f"Total sounds: {len(sounds)}")

    plot_species_bar(annotations, categories, top_n=20)
    plot_species_bar(annotations, categories, top_n=50, suffix="_top50")
    plot_top_species_examples(annotations, categories, sounds)
    plot_annotation_scenarios()  # Uses annotations_identification.json internally
    plot_complete_audio(annotations, categories, sounds)
    plot_audios_by_project_time_of_day()  # Uses annotations_identification.json internally


if __name__ == "__main__":
    main()
