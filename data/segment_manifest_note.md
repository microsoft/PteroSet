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
