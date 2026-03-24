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

# ── Mel-scale display helpers ────────────────────────────────────────────
# Precomputed mel tick values for consistent y-axis across all plots
_MEL_FREQ_TICKS_HZ = np.array([0, 1000, 2000, 4000, 8000, 16000, 24000])
_MEL_FREQ_TICKS_HZ = _MEL_FREQ_TICKS_HZ[_MEL_FREQ_TICKS_HZ <= FMAX]
_MEL_TICK_LABELS = [f"{int(f/1000)}k" if f >= 1000 else str(int(f)) for f in _MEL_FREQ_TICKS_HZ]


def _hz_to_mel_bin(freq_hz, max_freq_hz, n_mels):
    """Convert a frequency in Hz to a mel-scale bin position."""
    mel = 2595.0 * np.log10(1.0 + freq_hz / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + max_freq_hz / 700.0)
    return mel / mel_max * n_mels


# Precomputed tick positions in mel-bin space
_MEL_TICK_POSITIONS = np.array([_hz_to_mel_bin(f, FMAX, N_MELS) for f in _MEL_FREQ_TICKS_HZ])


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
        
        # Skip windows that cross a 10-second segment boundary
        if int(win_start_sec // 10) != int((win_end_sec - 1e-9) // 10):
            continue
        
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
    
    subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

    for idx, (key, title, label) in enumerate(scenario_names):
        ax = axes[idx]
        letter = subplot_labels[idx]
        
        if key not in examples:
            ax.set_title(f"{letter} {title}", fontsize=10, fontweight='bold')
            ax.text(0.5, 0.5, '(Example not found)', ha='center', va='center',
                    fontsize=9, transform=ax.transAxes)
            ax.axis('off')
            continue
        
        spec_file, info = examples[key]
        spec_path = os.path.join(SPECTROGRAMS_DIR, spec_file)
        spec = np.load(spec_path)
        
        win_start_sec = info['start_sample'] / SR
        win_end_sec = info['end_sample'] / SR
        duration_sec = win_end_sec - win_start_sec
        n_time_frames = spec.shape[-1]

        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')

        # X-axis: time in seconds
        time_ticks_sec = np.arange(0, duration_sec + 0.5, 1.0)
        time_tick_positions = time_ticks_sec / duration_sec * n_time_frames
        ax.set_xticks(time_tick_positions)
        ax.set_xticklabels([f"{t:.0f}" for t in time_ticks_sec], fontsize=7)

        # Y-axis: mel-scale frequency ticks
        ax.set_yticks(_MEL_TICK_POSITIONS)
        ax.set_yticklabels(_MEL_TICK_LABELS, fontsize=7)
        
        # Draw all annotations for this window
        for ann in info['annotations']:
            t_min, t_max = ann['t_min'], ann['t_max']
            f_min, f_max = ann['f_min'], ann['f_max']
            
            # Convert to relative coordinates in frame / mel-bin space
            rect_t_min = (t_min - win_start_sec) / duration_sec * n_time_frames
            rect_t_max = (t_max - win_start_sec) / duration_sec * n_time_frames
            y0 = _hz_to_mel_bin(f_min, FMAX, N_MELS)
            y1 = _hz_to_mel_bin(f_max, FMAX, N_MELS)
            
            rect = Rectangle(
                (rect_t_min, y0), rect_t_max - rect_t_min, y1 - y0,
                linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)
        
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Frequency (Hz)", fontsize=8)
        
        audio_name = os.path.basename(info['sound']['file_name_path'])
        n_anns = len(info['annotations'])
        ax.set_title(f"{letter} {title}", fontsize=10, fontweight='bold', pad=28)
        ax.text(0.5, 1.01, f"Label: {label} | {n_anns} annotation(s)\n"
                f"{audio_name} [{win_start_sec:.1f}s - {win_end_sec:.1f}s]",
                transform=ax.transAxes, fontsize=8, ha='center', va='bottom')
    
    fig.suptitle("Annotation Scenarios for Model Training", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved annotation scenarios to {out}")


def plot_most_annotated_audio_time_of_day():
    """Plot the single audio with the most annotations, with time-of-day x-axis.

    Same layout as :func:`plot_audios_by_project_time_of_day` but limited to the
    audio file that has the highest number of annotations across the whole dataset.

    Uses annotations_identification.json for annotations.
    """
    import librosa
    import soundfile as sf
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter
    import pandas as pd

    out = os.path.join(OUTPUT_DIR, "most_annotated_audio_time_of_day.png")

    if os.path.exists(out):
        print(f"Skipping most-annotated audio plot (already exists): {out}")
        return

    # Load annotations_identification.json
    annotations_id_path = os.path.join(ROOT, "annotations_identification.json")
    if not os.path.exists(annotations_id_path):
        print(f"Annotations identification file not found: {annotations_id_path}")
        return

    with open(annotations_id_path, 'r') as f:
        data = json.load(f)

    sounds = {s["id"]: s for s in data["sounds"]}
    annotations = data["annotations"]

    # Try loading species-level annotations for color-coded boxes
    species_annotations_path = os.path.join(ROOT, "annotations_species.json")
    species_by_coords = {}
    species_categories = {}
    if os.path.exists(species_annotations_path):
        with open(species_annotations_path, 'r') as f:
            species_data = json.load(f)
        species_categories = {c["id"]: c for c in species_data["categories"]}
        for ann in species_data["annotations"]:
            key = (ann["sound_id"], round(ann["t_min"], 6), round(ann["t_max"], 6),
                   round(ann["f_min"], 1), round(ann["f_max"], 1))
            species_by_coords[key] = ann

    # Find the sound with the most annotations
    ann_counts = Counter(ann['sound_id'] for ann in annotations)
    if not ann_counts:
        print("No annotations found")
        return

    best_sound_id, n_anns = ann_counts.most_common(1)[0]
    sound = sounds[best_sound_id]
    audio_name = os.path.basename(sound['file_name_path'])
    print(f"Most annotated audio: {audio_name} ({n_anns} annotations)")

    # Resolve audio path
    audio_path = sound['file_name_path']
    if not os.path.isabs(audio_path):
        if audio_path.startswith('data/'):
            audio_path = os.path.join(ROOT, audio_path[5:])
        else:
            audio_path = os.path.join(ROOT, audio_path)

    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    # Conversion helpers
    def audio_seconds_to_hours(t_seconds):
        return t_seconds * 3.0 / 60.0

    def format_time_of_day(seconds, pos=None):
        hours_float = audio_seconds_to_hours(seconds)
        hours = int(hours_float)
        mins = int((hours_float % 1) * 60)
        return f"{hours:02d}:{mins:02d}"

    # Load and process audio
    try:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=SR, n_fft=2048, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmax=FMAX
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        sound_annotations = [ann for ann in annotations if ann['sound_id'] == best_sound_id]

        fig, ax = plt.subplots(1, 1, figsize=(16, 4))

        n_time_frames = mel_spec_db.shape[1]
        ax.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='magma')

        # Y-axis: mel-scale frequency ticks
        ax.set_yticks(_MEL_TICK_POSITIONS)
        ax.set_yticklabels(_MEL_TICK_LABELS, fontsize=7)

        # Overlay annotations (converted to frame / mel-bin coordinates)
        # Build species color map if species annotations are available
        species_in_sound = set()
        if species_by_coords:
            for ann in sound_annotations:
                key = (ann["sound_id"], round(ann["t_min"], 6), round(ann["t_max"], 6),
                       round(ann["f_min"], 1), round(ann["f_max"], 1))
                sp_ann = species_by_coords.get(key)
                if sp_ann:
                    species_in_sound.add(sp_ann['category_id'])
        species_list = sorted(species_in_sound)
        cmap_tab = plt.cm.get_cmap('tab20', max(len(species_list), 1))
        species_color = {sid: cmap_tab(i) for i, sid in enumerate(species_list)}

        audio_duration = len(audio) / SR
        legend_handles = {}
        for ann in sound_annotations:
            t_min_audio = ann['t_min']
            t_max_audio = ann['t_max']
            f_min, f_max = ann['f_min'], ann['f_max']

            x0 = t_min_audio / audio_duration * n_time_frames
            x1 = t_max_audio / audio_duration * n_time_frames
            y0 = _hz_to_mel_bin(f_min, FMAX, N_MELS)
            y1 = _hz_to_mel_bin(f_max, FMAX, N_MELS)

            # Determine color from species annotation
            color = "cyan"
            label = "No species"
            key = (ann["sound_id"], round(ann["t_min"], 6), round(ann["t_max"], 6),
                   round(ann["f_min"], 1), round(ann["f_max"], 1))
            sp_ann = species_by_coords.get(key)
            if sp_ann and sp_ann['category_id'] in species_color:
                cat = species_categories[sp_ann['category_id']]
                color = species_color[sp_ann['category_id']]
                label = cat['name']

            rect = Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.0, edgecolor=color, facecolor="none", linestyle="--", alpha=0.8
            )
            ax.add_patch(rect)
            if label not in legend_handles:
                legend_handles[label] = rect

        ax.set_ylabel("Frequency (Hz)", fontsize=10, fontweight='bold')
        ax.set_xlabel("Time of Day", fontsize=11, fontweight='bold')

        tick_secs = [t for t in range(0, 481, 10) if t <= audio_duration]
        x_tick_positions = [t / audio_duration * n_time_frames for t in tick_secs]
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels([format_time_of_day(t) for t in tick_secs], fontsize=8,
                           rotation=45, ha='right')
        ax.set_xlim(0, n_time_frames)

        # Add species legend
        if legend_handles:
            ax.legend(legend_handles.values(), legend_handles.keys(),
                      loc='upper right', fontsize=6, ncol=2, framealpha=0.7)

        fig.suptitle(
            f"Audio and Annotations Example - {audio_name} ({n_anns} annotations)\n"
            f"Time of Day (10s every 30 min)",
            fontsize=13, fontweight='bold'
        )

        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved most-annotated audio plot to {out}")

    except Exception as e:
        print(f"Error processing audio {audio_name}: {e}")


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
    
    # Unique audio durations and counts
    duration_counts = Counter(round(d, 2) for d in durations)
    print("  Unique audio durations (seconds) and counts:")
    for dur in sorted(duration_counts):
        print(f"    {dur:.2f}s ({dur/60:.2f} min): {duration_counts[dur]} audios")
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
    
    # ── 7. ANNOTATION FREQUENCY STATISTICS (IDENTIFICATION LEVEL) ──
    print("7. ANNOTATION FREQUENCY STATISTICS (IDENTIFICATION LEVEL)")
    print("-" * 80)

    f_mins_id = [ann["f_min"] for ann in annotations_identification]
    f_maxs_id = [ann["f_max"] for ann in annotations_identification]
    f_ranges_id = [ann["f_max"] - ann["f_min"] for ann in annotations_identification]

    print(f"  Low Frequency (Hz):")
    print(f"    Min:    {np.min(f_mins_id):.1f}")
    print(f"    Max:    {np.max(f_mins_id):.1f}")
    print(f"    Mean:   {np.mean(f_mins_id):.1f}")
    print(f"    Median: {np.median(f_mins_id):.1f}")
    print(f"    Std:    {np.std(f_mins_id):.1f}")
    print(f"  High Frequency (Hz):")
    print(f"    Min:    {np.min(f_maxs_id):.1f}")
    print(f"    Max:    {np.max(f_maxs_id):.1f}")
    print(f"    Mean:   {np.mean(f_maxs_id):.1f}")
    print(f"    Median: {np.median(f_maxs_id):.1f}")
    print(f"    Std:    {np.std(f_maxs_id):.1f}")
    print(f"  Bandwidth (Hz):")
    print(f"    Min:    {np.min(f_ranges_id):.1f}")
    print(f"    Max:    {np.max(f_ranges_id):.1f}")
    print(f"    Mean:   {np.mean(f_ranges_id):.1f}")
    print(f"    Median: {np.median(f_ranges_id):.1f}")
    print(f"    Std:    {np.std(f_ranges_id):.1f}")
    print(f"  Percentiles (Low / High / Bandwidth):")
    for p in [5, 25, 50, 75, 95]:
        print(f"    P{p:2d}: {np.percentile(f_mins_id, p):8.1f} / {np.percentile(f_maxs_id, p):8.1f} / {np.percentile(f_ranges_id, p):8.1f}")
    print()

    # ── 8. SPECIES DIVERSITY ──
    print("8. SPECIES DIVERSITY")
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
    
    # Get unique projects (sorted with MAP1 first)
    def sort_projects(proj):
        if 'MAP1' in proj.upper():
            return '0_' + proj  # MAP1 comes first
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
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.1)
    
    # 1. Number of audios per project
    ax1 = fig.add_subplot(gs[0, 0])
    num_audios = [project_stats[proj]['num_audios'] for proj in projects]
    colors = plt.cm.magma(np.linspace(0.2, 0.85, num_projects))
    bars = ax1.barh(projects, num_audios, color=colors)
    ax1.set_xlabel('Number of Audios', fontsize=12)
    ax1.set_title('a) Audio Recording per Project', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=10)
    ax1.invert_yaxis()
    for bar, val in zip(bars, num_audios):
        ax1.text(bar.get_width() + max(num_audios)*0.01, bar.get_y() + bar.get_height()/2, 
                 str(val), va='center', fontsize=10)
    ax1.set_xlim(right=max(num_audios) * 1.15)
    
    # 2. Total audio duration per project (hours)
    ax2 = fig.add_subplot(gs[0, 1])
    total_durations = [np.sum(project_stats[proj]['audio_durations'])/3600 for proj in projects]
    bars = ax2.barh(projects, total_durations, color=colors)
    ax2.set_xlabel('Total Duration (hours)', fontsize=12)
    ax2.set_title('b) Total Audio Duration per Project', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=10)
    ax2.invert_yaxis()
    for bar, val in zip(bars, total_durations):
        ax2.text(bar.get_width() + max(total_durations)*0.01, bar.get_y() + bar.get_height()/2, 
                 f'{val:.2f}h', va='center', fontsize=10)
    ax2.set_xlim(right=max(total_durations) * 1.15)
    
    # 3. Mean annotation duration (identification level)
    ax3 = fig.add_subplot(gs[1, 0])
    total_ann_dur_id = [np.sum(project_stats[proj]['ann_durations_identification']) / 3600
                       if project_stats[proj]['ann_durations_identification'] else 0 
                       for proj in projects]
    bars = ax3.barh(projects, total_ann_dur_id, color=colors)
    ax3.set_xlabel('Total Annotated Duration (hours)', fontsize=12)
    ax3.set_title('c) Total Annotated Duration by Project', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=10)
    ax3.invert_yaxis()
    for bar, val in zip(bars, total_ann_dur_id):
        ax3.text(bar.get_width() + max(total_ann_dur_id)*0.01, bar.get_y() + bar.get_height()/2, 
                 f'{val:.2f}h', va='center', fontsize=10)
    ax3.set_xlim(right=max(total_ann_dur_id) * 1.15)
    
    # 4. Number of identification-level annotations per project
    ax4 = fig.add_subplot(gs[1, 1])
    num_anns_id = [project_stats[proj]['num_anns_identification'] for proj in projects]
    bars = ax4.barh(projects, num_anns_id, color=colors)
    ax4.set_xlabel('Number of Annotations', fontsize=12)
    ax4.set_title('d) Class-Level Annotations per Project', fontsize=13, fontweight='bold')
    ax4.tick_params(axis='both', labelsize=10)
    ax4.invert_yaxis()
    for bar, val in zip(bars, num_anns_id):
        ax4.text(bar.get_width() + max(num_anns_id)*0.01, bar.get_y() + bar.get_height()/2, 
                 str(val), va='center', fontsize=10)
    ax4.set_xlim(right=max(num_anns_id) * 1.15)
    
    # 5. Number of species-level annotations per project
    ax5 = fig.add_subplot(gs[2, 0])
    num_anns_species = [project_stats[proj]['num_anns_species'] for proj in projects]
    bars = ax5.barh(projects, num_anns_species, color=colors)
    ax5.set_xlabel('Number of Annotations', fontsize=12)
    ax5.set_title('e) Species-Level Annotations per Project', fontsize=13, fontweight='bold')
    ax5.tick_params(axis='both', labelsize=10)
    ax5.invert_yaxis()
    for bar, val in zip(bars, num_anns_species):
        ax5.text(bar.get_width() + max(num_anns_species)*0.01, bar.get_y() + bar.get_height()/2, 
                 str(val), va='center', fontsize=10)
    ax5.set_xlim(right=max(num_anns_species) * 1.15)
    
    # 6. Number of unique species per project
    ax6 = fig.add_subplot(gs[2, 1])
    num_species = [len(project_stats[proj]['species_set']) for proj in projects]
    bars = ax6.barh(projects, num_species, color=colors)
    ax6.set_xlabel('Number of Unique Species', fontsize=12)
    ax6.set_title('f) Species Diversity Identified per Project', fontsize=13, fontweight='bold')
    ax6.tick_params(axis='both', labelsize=10)
    ax6.invert_yaxis()
    for bar, val in zip(bars, num_species):
        ax6.text(bar.get_width() + max(num_species)*0.01, bar.get_y() + bar.get_height()/2, 
                 str(val), va='center', fontsize=10)
    ax6.set_xlim(right=max(num_species) * 1.15)
    
    fig.subplots_adjust(top=0.92) 
    fig.suptitle('Dataset Statistics by Project', fontsize=18, fontweight='bold', y=0.97)
    
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
    plot_top_species_examples(annotations, categories, sounds)
    plot_annotation_scenarios()  # Uses annotations_identification.json internally
    plot_most_annotated_audio_time_of_day()  # Single audio with most annotations


if __name__ == "__main__":
    main()
