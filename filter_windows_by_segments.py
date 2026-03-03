"""
Filter windows to keep only those that don't cross 10-second segment boundaries.

Usage:
    python filter_windows_by_segments.py --input data/windows_mapping_4.0overlap.json --output data/windows_mapping_4.0overlap_segmented.json
"""

import json
import argparse


def filter_windows_by_segments(windows, segment_duration_sec=10, sample_rate=48000):
    """Filter out windows that cross segment boundaries.
    
    Args:
        windows: List of window dictionaries
        segment_duration_sec: Duration of each segment in seconds
        sample_rate: Audio sample rate
    
    Returns:
        Filtered list of windows
    """
    segment_duration_samples = segment_duration_sec * sample_rate
    
    valid_windows = []
    
    for w in windows:
        start = w['start']
        end = w['end']
        
        # Check if window crosses a segment boundary
        start_segment = start // segment_duration_samples
        end_segment = (end - 1) // segment_duration_samples
        
        if start_segment == end_segment:
            valid_windows.append(w)
    
    return valid_windows


def main():
    parser = argparse.ArgumentParser(description="Filter windows by segment boundaries")
    parser.add_argument("--input", required=True, help="Input windows JSON file")
    parser.add_argument("--output", required=True, help="Output windows JSON file")
    parser.add_argument("--segment-duration", type=int, default=10, 
                       help="Segment duration in seconds (default: 10)")
    parser.add_argument("--sample-rate", type=int, default=48000,
                       help="Audio sample rate (default: 48000)")
    
    args = parser.parse_args()
    
    print(f"Loading windows from: {args.input}")
    with open(args.input, 'r') as f:
        windows = json.load(f)
    
    print(f"Total windows: {len(windows)}")
    
    print(f"\nFiltering windows (segment duration: {args.segment_duration}s, sample rate: {args.sample_rate})")
    filtered_windows = filter_windows_by_segments(
        windows, 
        segment_duration_sec=args.segment_duration,
        sample_rate=args.sample_rate
    )
    
    print(f"Valid windows: {len(filtered_windows)}")
    print(f"Removed: {len(windows) - len(filtered_windows)} ({100 * (1 - len(filtered_windows)/len(windows)):.1f}%)")
    
    # Reassign window IDs sequentially
    for i, w in enumerate(filtered_windows):
        w['window_id'] = i
    
    print(f"\nSaving to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(filtered_windows, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()
