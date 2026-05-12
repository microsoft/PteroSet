# Phase 1 Findings — PPA4 Regression Root Cause

**Date**: 2026-04-22 → 2026-04-23 investigation
**Conclusion**: **Case C — pipeline caching bug in `prepare_dataset.py`.**

## Summary

The 1 615 PPA4 annotations that v2 correctly applied as positive labels ARE present in the current source `annotations_identification.json`. v3 still labels those windows negative because `run_windows()` in `prepare_dataset.py` short-circuits when its output file already exists on disk, so the label-derivation from the current annotations is skipped. v3 inherited stale labels from the Feb 16 cache.

## Evidence

### Artifact timeline
| File | Modified | Role |
|---|---|---|
| `data/windows_mapping_4.0overlap.json` | 2026-02-16 | Unsegmented cache, built from pre-fix annotations |
| `data/windows_mapping_4.0overlap_segmented.json` (v1) | 2026-02-25 | Derived from above → pre-fix labels |
| `data/annotations_identification.json` | **2026-03-10** | **Source updated with 1 615 PPA4 additions** |
| `data/annotations_species.json` | 2026-03-10 | Species file (does NOT contain the 1 615 — they are AVEVOC-only, no species) |
| `data/windows_mapping_4.0overlap_segmented_v2.json` | 2026-03-19 | Has fix — produced by a path that bypassed the cache |
| `data/windows_mapping_4.0overlap_segmented_v3.json` | 2026-04-15 | Re-derived via cached Feb 16 mapping → lost fix |

### Source coverage verification
For the 1 615 PPA4 windows flipped 0→1 in v1→v2:
- `annotations_identification.json`: **1 615 / 1 615 covered** (every fix window has an overlapping annotation in the source)
- `annotations_species.json`: 0 / 1 615 (species file excludes AVEVOC-only annotations)

### Label-derivation check on v3 PPA4 windows
Deriving "should be positive" from `annotations_identification.json` overlap and comparing to v3's stored label:

| Source says positive | v3 label | Count | Interpretation |
|---|---|---|---|
| True | 0 | **1 615** | Regression — source has the annotation, v3 says negative |
| True | 1 | 7 329 | Correct positive |
| False | 0 | 25 331 | Correct negative |
| False | 1 | 5 | Edge cases (same 5 that v1→v2 flipped 1→0; boundary effects) |

This is an exact match with the "regression set" identified earlier.

### Pipeline bug: cache shortcut in `run_windows()`
`prepare_dataset.py:107-111`:
```python
if os.path.exists(windows_output_path):
    print(f"Loading existing windows from: {windows_output_path}")
    with open(windows_output_path, 'r') as f:
        windows = json.load(f)
    print(f"Loaded {len(windows)} windows")
else:
    ... build_windows(annotation_file=annotation_path, ...)
```

The cache key is **only the filename** (`windows_mapping_{overlap}overlap.json`), not the annotation file mtime. Once built on Feb 16, it was never invalidated. The Mar-10 annotation update had no effect on subsequent pipeline runs.

`run_segment_windows()` (line 144 onward) reads labels from the input window dicts verbatim — it does not re-derive labels from annotations. So the stale labels propagate unchanged into the segmented mapping.

### Why v2 has the fix
Cache `windows_mapping_4.0overlap.json` retains its Feb 16 mtime — no other pipeline run rewrote it. So v2 was produced either by:
- A one-off script that re-derived labels directly against the Mar-10 annotations for the segmented file, bypassing `run_windows()`, or
- The cache was temporarily deleted on Mar 19 and rebuilt, with the rebuilt file later overwritten back to Feb 16 state (less likely).

Either way, v2's correct labels are "out-of-band" with respect to the current pipeline.

## Implications for v4 fix

- **Source is already correct** — no need to patch `annotations_identification.json`.
- **Pipeline is buggy** — the cache shortcut needs to be either removed or made mtime-aware.
- **v4 rebuild is straightforward**: invalidate the Feb-16 cache, rerun `prepare_dataset.py --steps windows segment_windows splits`. `build_windows` (PytorchWildlife) will derive labels from the current annotations JSON, including the 1 615 PPA4 fix.

## Recommended fix for the pipeline

Two options:

1. **Minimal — cache invalidation by mtime**. In `run_windows()`, compare `os.path.getmtime(annotation_path)` with `os.path.getmtime(windows_output_path)`; rebuild if annotations are newer. Small code change, preserves caching for performance.

2. **Paranoid — remove the cache entirely**. Delete the `if os.path.exists` short-circuit; always rebuild. Simplest, but slower on large datasets. Acceptable since `build_windows` runs once per version.

Option 1 is the minimum correct fix and should be applied in Phase 2.

## Action items (roll into Phase 2)

1. Back up `data/windows_mapping_4.0overlap.json` (current state: stale).
2. Patch `prepare_dataset.py:107-111` to add mtime-based cache invalidation.
3. Delete the stale cache and rerun `python prepare_dataset.py --config data/config.yaml --steps windows segment_windows splits --version v4` (or patch the default version).
4. Phase 3 verification gate: confirm PPA4 positives = 8 944 and other projects match v3.
