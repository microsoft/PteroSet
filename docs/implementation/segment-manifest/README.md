# Segment Manifest

## Purpose and rationale

PteroSet source WAVs concatenate time-lapse snapshots. Treating every project
as a sequence of non-overlapping 10-second blocks loses the actual geometry of
PPA1, whose adjacent snapshots overlap by one second. The segment
manifest records each snapshot explicitly so filtering, extraction, and review
use one shared rule:

| Project | Snapshot duration | Start-to-start stride | Geometry |
|---------|-------------------|-----------------------|----------|
| PPA1 | 10 seconds | 9 seconds | Adjacent snapshots overlap by 1 second |
| MAP1, PPA2, PPA3, PPA4 | 10 seconds | 10 seconds | Adjacent snapshots do not overlap |

For segment index `i`, the geometry is:

```text
start_sec_in_file = i * segment_stride_sec
end_sec_in_file   = start_sec_in_file + 10
```

The manifest provides a stable segment identifier and both second- and
sample-based ranges. This avoids duplicating project-specific boundary logic
in downstream tools. Rows describe complete 10-second snapshots; a trailing
interval shorter than 10 seconds is not a segment.

## Segmented-window cache correction

### Old flow and root cause

The v4 dataset did not generate segmented windows from the segment manifest.
It first loaded or generated the recording-level
`windows_mapping_4.0overlap.json`, then retained only windows contained within
the project-aware 10-second segments:

```text
annotations -> cached recording-level windows -> segment-boundary filter
            -> segmented windows -> spectrograms -> folds -> checkpoints
```

That flow made the segmented dataset depend on the completeness of an
upstream cache. The cached recording-level mapping ended before the currently
annotated duration for a subset of recordings, so filtering it could not
recover their valid tail windows. The segment geometry itself was correct:
PPA1 used a 9-second stride and all other projects used a 10-second stride.
The defect was cache invalidation and cache provenance, not the segment
containment calculation.

The shipped segmented v4 mapping therefore contains 160,244 windows. The v4
folds, spectrogram selection, and checkpoints were built from that mapping and
exclude 1,822 valid windows.

### Corrected flow

Segmented windows are now generated directly from the current annotations and
the shared segment geometry:

```text
current annotations + segment geometry -> segmented windows
                                       -> required spectrograms
                                       -> folds -> checkpoints
```

Each complete 10-second segment contributes six 5-second windows at a
1-second step. Generation no longer requires the recording-level windows
cache to contain every candidate window. Spectrogram generation consumes the
new segmented mapping so that every mapped window has its required
spectrogram.

### Verified expected counts

The following counts are derived from the current annotation sound durations
and the shared full-segment iterator:

| Project | Complete segments | Expected windows | Old v4 windows | Added |
|---------|------------------:|-----------------:|---------------:|------:|
| MAP1 | 2,208 | 13,248 | 13,018 | 230 |
| PPA1 | 5,184 | 31,104 | 31,104 | 0 |
| PPA2 | 6,576 | 39,456 | 38,947 | 509 |
| PPA3 | 7,248 | 43,488 | 42,895 | 593 |
| PPA4 | 5,795 | 34,770 | 34,280 | 490 |
| **Total** | **27,011** | **162,066** | **160,244** | **1,822** |

The expected total is 162,066, not 162,144. Of the 121 PPA4 recordings, five
have current annotated durations shorter than 480 seconds: two are 440
seconds, two are 460 seconds, and one is 470 seconds. They provide 13 fewer
complete segments, or 78 fewer windows, than the all-480-second assumption.

### What is known and what is not

Verified facts:

- the old segmented v4 artifact contains 160,244 windows;
- direct generation from current annotation durations produces 162,066;
- the 1,822-window difference consists of valid tail windows absent from the
  stale recording-level cache; and
- the five short PPA4 durations account for the 78-window difference between
  162,144 and 162,066.

The historical origin of the older duration values used when the stale cache
was created is unknown. The available artifacts establish that the cache ends
early; they do not establish whether those duration values came from an older
annotation export, audio metadata, or another preprocessing step. Do not
record a specific historical source without additional provenance evidence.

### Artifact migration

The corrected mapping is a dataset revision, not an in-place reinterpretation
of v4 results. Rebuild the segmented mapping, generate all spectrograms
required by it, regenerate folds/splits, and retrain/evaluate checkpoints.
Window IDs and downstream row assignments may change when the omitted windows
are inserted, so existing folds must not be combined with the corrected
mapping by ID. Retain old v4 checkpoints only as results for the 160,244-window
dataset; they do not include the 1,822 recovered windows.

## Time semantics

`start_sec_in_file`, `end_sec_in_file`, `start_sample`, and `end_sample` are
offsets inside the concatenated source WAV. They are **not absolute acquisition
timestamps**.

In particular, `date_recorded` alone cannot establish the real capture time of
each time-lapse snapshot. Deriving a wall-clock timestamp would require
additional acquisition schedule or per-snapshot timing metadata that is not
represented by this manifest. Consumers must not compute a capture timestamp
by adding `start_sec_in_file` to `date_recorded`.

Ranges use half-open interval notation: `[start, end)`. This makes the expected
sample count `end_sample - start_sample` and prevents adjacent non-overlapping
ranges from sharing a sample.

## CSV schema

The CSV header is fixed and ordered as follows:

```text
segment_id,sound_id,segment_index,audio_file,project,start_sec_in_file,end_sec_in_file,start_sample,end_sample,source_sample_rate,segment_duration_sec,segment_stride_sec,date_recorded,location_id,recorder_id
```

| Field | Meaning |
|-------|---------|
| `segment_id` | Unique, stable identifier used to select a segment for extraction. Treat it as opaque. |
| `sound_id` | Identifier of the source sound in the annotations JSON. |
| `segment_index` | Zero-based snapshot index within the source sound. |
| `audio_file` | Portable source WAV filename, resolved beneath `audio-root` during extraction. |
| `project` | Source project (`MAP1` or `PPA1` through `PPA4`), which selects the stride. |
| `start_sec_in_file` | Snapshot start offset in seconds within the concatenated WAV. |
| `end_sec_in_file` | Exclusive snapshot end offset in seconds within the concatenated WAV. |
| `start_sample` | Snapshot start offset in source samples. |
| `end_sample` | Exclusive snapshot end offset in source samples. |
| `source_sample_rate` | Sample rate of the source WAV in samples per second. |
| `segment_duration_sec` | Snapshot duration; `10` seconds. |
| `segment_stride_sec` | Start-to-start spacing; `9` for PPA1 and `10` for the other projects. |
| `date_recorded` | Optional recording-level metadata copied from the metadata CSV; not a per-snapshot timestamp. |
| `location_id` | Optional location metadata copied from the metadata CSV. |
| `recorder_id` | Optional recorder metadata copied from the metadata CSV. |

When no metadata CSV is supplied, or no metadata row matches a sound, the
optional metadata fields remain empty. Their absence does not change segment
geometry.

## Usage

The command-line entry point is `data/segment_manifest.py`; `--help` lists all
available options.

Generate a manifest from annotations and optional recording metadata:

```bash
python data/segment_manifest.py generate \
    --annotations data/annotations_identification.json \
    --metadata data/metadata.csv \
    --output data/segment_manifest.csv
```

Without metadata:

```bash
python data/segment_manifest.py generate \
    --annotations data/annotations_identification.json \
    --output data/segment_manifest.csv
```

Extract one segment by identifier:

```bash
python data/segment_manifest.py extract \
    --manifest data/segment_manifest.csv \
    --segment-id SEGMENT_ID \
    --audio-root data/audios_192khz \
    --output segment.wav
```

Extraction reads the sample range from the selected row rather than
reconstructing geometry from the identifier. The generator stores only the
portable WAV filename, so `audio-root` must identify the directory containing
the source WAV files.

## Validation expectations

A generator or consumer should verify:

- the CSV header exactly matches the documented schema;
- `segment_id` is non-empty and unique;
- each source sound has ordered, zero-based `segment_index` values;
- `segment_duration_sec` is 10 and `segment_stride_sec` is 9 only for PPA1,
  otherwise 10;
- second offsets follow the project stride and
  `end_sec_in_file - start_sec_in_file = 10`;
- sample offsets are non-negative integers, use the source sample rate, and
  satisfy `start_sample = start_sec_in_file * source_sample_rate`,
  `end_sample = end_sec_in_file * source_sample_rate`, and
  `start_sample < end_sample`;
- `[start_sample, end_sample)` lies within the source WAV and contains exactly
  the samples written by extraction;
- the source WAV sample rate agrees with `source_sample_rate`;
- extraction fails clearly for an unknown or duplicate segment identifier, a
  missing source WAV, malformed numeric values, or an out-of-range sample
  interval; and
- missing optional metadata produces empty fields rather than guessed values.

For PPA1, a useful geometry check is that segment 0 covers `[0, 10)`, segment 1
covers `[9, 19)`, and their one-second overlap is intentional. For another
project, segment 0 covers `[0, 10)` and segment 1 covers `[10, 20)`.

## Reviewer-response wording

Suggested concise wording for reviews or change summaries:

> Segmented windows are now generated directly from current annotations and
> project-aware segment geometry instead of filtering a recording-level
> window cache. The old cache omitted 1,822 valid tail windows, leaving the v4
> dataset and checkpoints at 160,244 windows. The corrected expected total is
> 162,066: five short PPA4 recordings account for the 78-window reduction from
> the otherwise expected 162,144. The segment math was not the fault; the
> upstream cache was stale.
