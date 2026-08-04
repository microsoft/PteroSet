"""Tests for segment manifest generation and extraction."""

import csv
import json
from pathlib import Path
import struct
import wave

import pytest

from data.segment_manifest import (
    extract_segment,
    generate_manifest,
    generate_manifest_records,
    main,
    read_manifest,
    select_segment,
)
from data.segment_utils import (
    iter_full_segments,
    window_is_contained_in_segment,
)


def _write_annotations(path: Path, sounds: list) -> None:
    path.write_text(json.dumps({"sounds": sounds, "annotations": []}))


def _write_metadata(path: Path, rows: list) -> None:
    with path.open("w", newline="") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=[
                "audio_file",
                "project_name",
                "date_recorded",
                "location_id",
                "recorder_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_wav(path: Path, sample_rate: int, samples: list) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def test_enumerates_default_and_ppa1_segment_geometry():
    default_segments = list(iter_full_segments(480, 100, "PPA2"))
    ppa1_segments = list(iter_full_segments(433, 100, "PPA1"))

    assert len(default_segments) == 48
    assert (default_segments[-1].start_sec, default_segments[-1].end_sec) == (
        470,
        480,
    )
    assert len(ppa1_segments) == 48
    assert (ppa1_segments[-1].start_sec, ppa1_segments[-1].end_sec) == (
        423,
        433,
    )
    assert len(list(iter_full_segments(479.9999995, 100, "PPA2"))) == 48
    assert len(list(iter_full_segments(479.999, 100, "PPA2"))) == 47


def test_manifest_is_deterministic_sorted_and_joins_metadata(tmp_path):
    annotations = tmp_path / "annotations.json"
    metadata = tmp_path / "metadata.csv"
    manifest_a = tmp_path / "segments-a.csv"
    manifest_b = tmp_path / "segments-b.csv"
    _write_annotations(
        annotations,
        [
            {
                "id": 10,
                "file_name_path": "/source/PPA1.wav",
                "duration": 433,
                "sample_rate": 100,
                "project": "PPA1",
            },
            {
                "id": 2,
                "file_name_path": "audios/PPA2.wav",
                "duration": 480,
                "sample_rate": 100,
                "project": "PPA2",
            },
        ],
    )
    _write_metadata(
        metadata,
        [
            {
                "audio_file": "PPA2.wav",
                "project_name": "PPA2",
                "date_recorded": "2024-02-03",
                "location_id": "location-2",
                "recorder_id": "recorder-2",
            },
            {
                "audio_file": "PPA1.wav",
                "project_name": "PPA1",
                "date_recorded": "2024-01-02",
                "location_id": "location-1",
                "recorder_id": "recorder-1",
            },
        ],
    )

    records = generate_manifest(annotations, manifest_a, metadata)
    generate_manifest(annotations, manifest_b, metadata)

    assert len(records) == 96
    assert records[0].sound_id == "2"
    assert records[0].segment_index == 0
    assert records[0].audio_file == "PPA2.wav"
    assert records[0].date_recorded == "2024-02-03"
    assert records[0].location_id == "location-2"
    assert records[0].recorder_id == "recorder-2"
    assert records[47].segment_index == 47
    assert records[48].sound_id == "10"
    assert records[-1].start_sec_in_file == 423
    assert records[-1].end_sec_in_file == 433
    assert records[-1].segment_stride_sec == 9
    assert records[0].segment_id == "segment-v1-2-0000"
    assert manifest_a.read_bytes() == manifest_b.read_bytes()


def test_duplicate_metadata_audio_file_is_rejected(tmp_path):
    annotations = tmp_path / "annotations.json"
    metadata = tmp_path / "metadata.csv"
    _write_annotations(
        annotations,
        [
            {
                "id": 1,
                "file_name_path": "audio.wav",
                "duration": 10,
                "sample_rate": 100,
                "project": "PPA2",
            }
        ],
    )
    _write_metadata(
        metadata,
        [
            {"audio_file": "first/audio.wav", "project_name": "PPA2"},
            {"audio_file": "second/audio.wav", "project_name": "PPA2"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate audio_file"):
        generate_manifest_records(annotations, metadata)


def test_conflicting_annotation_and_metadata_projects_are_rejected(tmp_path):
    annotations = tmp_path / "annotations.json"
    metadata = tmp_path / "metadata.csv"
    _write_annotations(
        annotations,
        [
            {
                "id": 1,
                "file_name_path": "audio.wav",
                "duration": 10,
                "sample_rate": 100,
                "project": "PPA1",
            }
        ],
    )
    _write_metadata(
        metadata,
        [{"audio_file": "audio.wav", "project_name": "PPA2"}],
    )

    with pytest.raises(ValueError, match="project conflicts"):
        generate_manifest_records(annotations, metadata)


def test_ambiguous_annotation_filename_metadata_join_is_rejected(tmp_path):
    annotations = tmp_path / "annotations.json"
    metadata = tmp_path / "metadata.csv"
    _write_annotations(
        annotations,
        [
            {
                "id": 1,
                "file_name_path": "first/audio.wav",
                "duration": 10,
                "sample_rate": 100,
                "project": "PPA2",
            },
            {
                "id": 2,
                "file_name_path": "second/audio.wav",
                "duration": 10,
                "sample_rate": 100,
                "project": "PPA2",
            },
        ],
    )
    _write_metadata(
        metadata,
        [{"audio_file": "audio.wav", "project_name": "PPA2"}],
    )

    with pytest.raises(ValueError, match="manifest path ambiguous"):
        generate_manifest_records(annotations, metadata)


def test_missing_required_sound_field_is_rejected(tmp_path):
    annotations = tmp_path / "annotations.json"
    _write_annotations(
        annotations,
        [
            {
                "id": 1,
                "file_name_path": "audio.wav",
                "duration": 10,
                "project": "PPA2",
            }
        ],
    )

    with pytest.raises(ValueError, match="sample_rate"):
        generate_manifest_records(annotations)


def test_extracts_exact_wav_frames_and_cli_round_trip(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    source_wav = audio_root / "source.wav"
    samples = list(range(3000))
    _write_wav(source_wav, sample_rate=100, samples=samples)

    annotations = tmp_path / "annotations.json"
    manifest = tmp_path / "segments.csv"
    output_wav = tmp_path / "segment.wav"
    _write_annotations(
        annotations,
        [
            {
                "id": "source",
                "file_name_path": "source.wav",
                "duration": 30,
                "sample_rate": 100,
                "project": "PPA2",
            }
        ],
    )

    assert (
        main(
            [
                "generate",
                "--annotations",
                str(annotations),
                "--output",
                str(manifest),
            ]
        )
        == 0
    )
    record = select_segment(read_manifest(manifest), "segment-v1-source-0001")
    assert (
        main(
            [
                "extract",
                "--manifest",
                str(manifest),
                "--segment-id",
                record.segment_id,
                "--audio-root",
                str(audio_root),
                "--output",
                str(output_wav),
            ]
        )
        == 0
    )

    with wave.open(str(output_wav), "rb") as extracted:
        assert extracted.getframerate() == 100
        assert extracted.getnframes() == 1000
        values = struct.unpack("<1000h", extracted.readframes(1000))
    assert list(values) == samples[1000:2000]


def test_sample_rate_mismatch_is_rejected(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    _write_wav(audio_root / "source.wav", sample_rate=200, samples=[0] * 2000)
    annotations = tmp_path / "annotations.json"
    _write_annotations(
        annotations,
        [
            {
                "id": 1,
                "file_name_path": "source.wav",
                "duration": 10,
                "sample_rate": 100,
                "project": "PPA2",
            }
        ],
    )
    record = generate_manifest_records(annotations)[0]

    with pytest.raises(ValueError, match="sample rate"):
        extract_segment(record, audio_root, tmp_path / "output.wav")


def test_extract_does_not_overwrite_source_wav(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    source_wav = audio_root / "source.wav"
    _write_wav(source_wav, sample_rate=100, samples=[0] * 1000)
    annotations = tmp_path / "annotations.json"
    _write_annotations(
        annotations,
        [
            {
                "id": 1,
                "file_name_path": "source.wav",
                "duration": 10,
                "sample_rate": 100,
                "project": "PPA2",
            }
        ],
    )
    record = generate_manifest_records(annotations)[0]

    with pytest.raises(ValueError, match="must not overwrite"):
        extract_segment(record, audio_root, source_wav)


def test_absolute_manifest_audio_path_cannot_ignore_audio_root(tmp_path):
    annotations = tmp_path / "annotations.json"
    _write_annotations(
        annotations,
        [
            {
                "id": 1,
                "file_name_path": "source.wav",
                "duration": 10,
                "sample_rate": 100,
                "project": "PPA2",
            }
        ],
    )
    record = generate_manifest_records(annotations)[0]
    absolute_record = type(record)(
        **{**record.__dict__, "audio_file": str(tmp_path / "source.wav")}
    )

    with pytest.raises(ValueError, match="relative to audio_root"):
        extract_segment(absolute_record, tmp_path, tmp_path / "output.wav")


def test_read_manifest_rejects_inconsistent_geometry(tmp_path):
    annotations = tmp_path / "annotations.json"
    manifest = tmp_path / "segments.csv"
    _write_annotations(
        annotations,
        [
            {
                "id": 1,
                "file_name_path": "source.wav",
                "duration": 10,
                "sample_rate": 100,
                "project": "PPA2",
            }
        ],
    )
    generate_manifest(annotations, manifest)

    rows = list(csv.DictReader(manifest.open()))
    rows[0]["end_sample"] = "999"
    with manifest.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="inconsistent sample offsets"):
        read_manifest(manifest)


def test_shared_containment_matches_default_and_overlapping_geometry():
    assert window_is_contained_in_segment(100, 900, "PPA2", 100)
    assert not window_is_contained_in_segment(900, 1100, "PPA2", 100)
    assert window_is_contained_in_segment(950, 1050, "PPA1", 100)
    assert not window_is_contained_in_segment(850, 1050, "PPA1", 100)
