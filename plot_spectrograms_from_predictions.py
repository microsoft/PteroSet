"""
Plot spectrograms from a predictions CSV and save them as PNG images.

Loads pre-computed .npy spectrogram files (the same used as model input)
and renders them using the plasma colormap.

Usage:
    python plot_spectrograms_from_predictions.py \
        --csv data/predictions/G010_timelapse_20250627_with_predictions.csv \
        --config data/config.yaml \
        --output_dir data/predictions/spectrograms
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def time_str_to_seconds(t: str) -> float:
    """Convert a mm:ss string to total seconds."""
    parts = t.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_spectrogram(spec: np.ndarray, start_sec: float, end_sec: float,
                     label: int, prediction: int, probability: float,
                     prediction_type: str, output_path: str,
                     max_freq_hz: float = 24000.0):
    """Plot a single spectrogram and save as PNG."""
    n_mels = spec.shape[0]
    n_time_frames = spec.shape[-1]
    duration_sec = end_sec - start_sec

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")

    # X-axis: time ticks in seconds
    time_ticks_sec = np.arange(0, duration_sec + 0.5, 1.0)
    time_tick_positions = time_ticks_sec / duration_sec * n_time_frames
    ax.set_xticks(time_tick_positions)
    ax.set_xticklabels([f"{t + start_sec:.0f}" for t in time_ticks_sec])
    ax.set_xlabel("Time (s)")

    # Y-axis: frequency in mel scale
    freq_ticks_hz = np.array([0, 1000, 2000, 4000, 8000, 16000, 24000])
    freq_ticks_hz = freq_ticks_hz[freq_ticks_hz <= max_freq_hz]
    mel_ticks = 2595.0 * np.log10(1.0 + freq_ticks_hz / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + max_freq_hz / 700.0)
    mel_tick_positions = mel_ticks / mel_max * n_mels
    freq_tick_labels = [f"{int(f/1000)}k" if f >= 1000 else str(int(f))
                        for f in freq_ticks_hz]
    ax.set_yticks(mel_tick_positions)
    ax.set_yticklabels(freq_tick_labels)
    ax.set_ylabel("Freq (Hz)")

    ax.set_title(
        f"Label: {label}  |  Pred: {prediction}  |  Prob: {probability:.4f}  |  {prediction_type}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot spectrograms from a predictions CSV"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the _with_predictions CSV")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (used for spectrograms_dir)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save PNG images (default: next to CSV)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    spectrograms_dir = cfg["paths"]["spectrograms_dir"]
    max_freq_hz = cfg["audio"]["sample_rate"] / 2.0

    df = pd.read_csv(args.csv)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.csv), "spectrograms")
    os.makedirs(args.output_dir, exist_ok=True)

    for _, row in df.iterrows():
        spec_path = os.path.join(spectrograms_dir, row["spec_name"])
        if not os.path.exists(spec_path):
            print(f"Spectrogram not found, skipping: {spec_path}")
            continue

        spec = np.load(spec_path)

        start_sec = time_str_to_seconds(str(row["start"]))
        end_sec = time_str_to_seconds(str(row["end"]))

        png_name = os.path.splitext(row["spec_name"])[0] + ".png"
        output_path = os.path.join(args.output_dir, png_name)

        plot_spectrogram(
            spec=spec,
            start_sec=start_sec,
            end_sec=end_sec,
            label=row["label"],
            prediction=row["prediction"],
            probability=row["probability"],
            prediction_type=row["prediction_type"],
            output_path=output_path,
            max_freq_hz=max_freq_hz,
        )

    print(f"Saved {len(df)} spectrograms to: {args.output_dir}")


if __name__ == "__main__":
    main()
