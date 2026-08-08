"""
Create a CSV from a windows mapping JSON file, enriched with sound paths
from an annotations JSON file.

Usage:
    python windows_to_csv.py \
        --windows /path/to/windows_mapping_1.5overlap.json \
        --annotations /path/to/annotations_filtered_dclde2026.json \
        --output windows_mapping_1.5overlap.csv
"""

import argparse
import csv
import json
import os


def windows_to_csv(windows_path, annotations_path, output_path):
    """Convert a windows mapping JSON to CSV, adding sound_path from annotations.

    Args:
        windows_path: Path to the windows mapping JSON file.
        annotations_path: Path to the annotations JSON file containing sounds.
        output_path: Path to the output CSV file.
    """
    print(f"Loading windows from: {windows_path}")
    with open(windows_path, 'r') as f:
        windows = json.load(f)
    print(f"Loaded {len(windows)} windows")

    print(f"Loading annotations from: {annotations_path}")
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Build sound_id -> sound_path lookup (skip entries missing required keys)
    sounds = {}
    for s in annotations.get('sounds', []):
        sid = s.get('id')
        path = s.get('file_name_path', '')
        if sid is not None:
            sounds[sid] = path
    print(f"Loaded {len(sounds)} sound entries")

    if not windows:
        print("No windows found; writing empty CSV.")
        with open(output_path, 'w', newline='') as f:
            f.write('')
        return

    # Collect all field names present in the windows (preserve order)
    fieldnames = list(dict.fromkeys(
        key for w in windows for key in w.keys()
    ))
    if 'sound_path' not in fieldnames:
        fieldnames.append('sound_path')

    missing_count = 0
    print(f"Writing CSV to: {output_path}")
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for w in windows:
            sound_id = w.get('sound_id')
            sound_path = sounds.get(sound_id, '')
            if not sound_path:
                missing_count += 1
            row = dict(w)
            row['sound_path'] = sound_path
            writer.writerow(row)

    if missing_count:
        print(f"Warning: {missing_count} windows had no matching sound path")
    print(f"Done. Saved {len(windows)} rows to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a windows mapping JSON to CSV with sound paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python windows_to_csv.py \\
        --windows /home/v-druizlopez/bioacoustics/orcas_dclde2026/data/windows_mapping_1.5overlap.json \\
        --annotations /home/v-druizlopez/bioacoustics/orcas_dclde2026/data/annotations_filtered_dclde2026.json \\
        --output windows_mapping_1.5overlap.csv
        """,
    )
    parser.add_argument(
        "--windows", required=True,
        help="Path to the windows mapping JSON file (e.g. windows_mapping_1.5overlap.json)"
    )
    parser.add_argument(
        "--annotations", required=True,
        help="Path to the annotations JSON file containing sound paths "
             "(e.g. annotations_filtered_dclde2026.json)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the output CSV file"
    )

    args = parser.parse_args()

    for path in (args.windows, args.annotations):
        if not os.path.exists(path):
            parser.error(f"File not found: {path}")

    windows_to_csv(args.windows, args.annotations, args.output)


if __name__ == "__main__":
    main()
