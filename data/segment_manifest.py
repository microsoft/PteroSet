"""Generate and consume time-lapse segment manifests.

Manifest times are offsets within concatenated audio files, not absolute
acquisition timestamps.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union
from urllib.parse import quote
import wave

from data.segment_utils import (
    SEGMENT_DURATION_SEC,
    iter_full_segments,
    segment_stride_sec,
)


MANIFEST_VERSION = "1"
DEFAULT_MANIFEST_FILENAME = f"segment_manifest_v{MANIFEST_VERSION}.csv"
MANIFEST_FIELDS = [
    "manifest_version",
    "segment_id",
    "sound_id",
    "segment_index",
    "audio_file",
    "project",
    "start_sec_in_file",
    "end_sec_in_file",
    "start_sample",
    "end_sample",
    "source_sample_rate",
    "segment_duration_sec",
    "segment_stride_sec",
    "date_recorded",
    "location_id",
    "recorder_id",
]
REQUIRED_SOUND_FIELDS = ("id", "file_name_path", "duration", "sample_rate")


@dataclass(frozen=True)
class SegmentRecord:
    """One original time-lapse segment within a concatenated audio file."""

    manifest_version: str
    segment_id: str
    sound_id: str
    segment_index: int
    audio_file: str
    project: str
    start_sec_in_file: float
    end_sec_in_file: float
    start_sample: int
    end_sample: int
    source_sample_rate: int
    segment_duration_sec: float
    segment_stride_sec: float
    date_recorded: str
    location_id: str
    recorder_id: str


def _portable_audio_file(file_name_path: object) -> str:
    value = str(file_name_path).strip().replace("\\", "/")
    if not value:
        raise ValueError("sound file_name_path must not be empty")

    windows_path = PureWindowsPath(str(file_name_path))
    posix_path = PurePosixPath(value)
    if windows_path.is_absolute() or posix_path.is_absolute():
        return windows_path.name if windows_path.is_absolute() else posix_path.name

    parts = [part for part in posix_path.parts if part not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError(
            f"sound file_name_path must be a portable relative path: {file_name_path!r}"
        )
    return PurePosixPath(*parts).as_posix()


def _metadata_by_audio_file(
    metadata_csv: Optional[Union[str, Path]],
) -> Dict[str, Dict[str, str]]:
    if metadata_csv is None:
        return {}

    with open(metadata_csv, newline="", encoding="utf-8-sig") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if reader.fieldnames is None or "audio_file" not in reader.fieldnames:
            raise ValueError("metadata CSV must contain an audio_file column")

        metadata = {}
        for row_number, row in enumerate(reader, start=2):
            audio_file = (row.get("audio_file") or "").strip()
            if not audio_file:
                raise ValueError(
                    f"metadata CSV row {row_number} has an empty audio_file"
                )
            key = PureWindowsPath(audio_file).name
            if key in metadata:
                raise ValueError(
                    f"metadata CSV has duplicate audio_file rows for {key!r}"
                )
            metadata[key] = {
                key: (value or "").strip() for key, value in row.items() if key
            }
    return metadata


def _required_sound_value(sound: Mapping[str, object], field: str) -> object:
    if field not in sound or sound[field] is None or sound[field] == "":
        raise ValueError(f"annotation sound is missing required field {field!r}")
    return sound[field]


def _sound_sort_key(sound_id: object) -> tuple:
    if isinstance(sound_id, int) and not isinstance(sound_id, bool):
        return (0, sound_id, str(sound_id))
    text = str(sound_id)
    try:
        return (0, int(text), text)
    except ValueError:
        return (1, text, text)


def _metadata_value(
    sound: Mapping[str, object],
    metadata: Mapping[str, str],
    field: str,
    *aliases: str,
) -> str:
    for key in (field, *aliases):
        value = metadata.get(key)
        if value:
            return value
    value = sound.get(field)
    return "" if value is None else str(value)


def generate_manifest_records(
    annotations_json: Union[str, Path],
    metadata_csv: Optional[Union[str, Path]] = None,
) -> List[SegmentRecord]:
    """Build deterministic segment records from annotation sound entries."""
    with open(annotations_json, encoding="utf-8") as annotations_file:
        annotations = json.load(annotations_file)
    sounds = annotations.get("sounds")
    if not isinstance(sounds, list):
        raise ValueError("annotation JSON must contain a sounds list")

    metadata_by_file = _metadata_by_audio_file(metadata_csv)
    seen_sound_ids = set()
    seen_metadata_join_keys = set()
    records = []
    for sound in sounds:
        if not isinstance(sound, dict):
            raise ValueError("each annotation sound must be an object")
        for field in REQUIRED_SOUND_FIELDS:
            _required_sound_value(sound, field)

        sound_id_value = sound["id"]
        sound_id = str(sound_id_value)
        if sound_id in seen_sound_ids:
            raise ValueError(f"annotation JSON has duplicate sound id {sound_id!r}")
        seen_sound_ids.add(sound_id)

        audio_file = _portable_audio_file(sound["file_name_path"])
        metadata_join_key = PurePosixPath(audio_file).name
        if metadata_csv is not None and metadata_join_key in seen_metadata_join_keys:
            raise ValueError(
                "annotation sounds contain duplicate audio filenames, making "
                f"the metadata join ambiguous for {metadata_join_key!r}"
            )
        seen_metadata_join_keys.add(metadata_join_key)
        metadata = metadata_by_file.get(metadata_join_key, {})
        project = _metadata_value(sound, metadata, "project", "project_name")
        if not project:
            raise ValueError(
                f"sound {sound_id!r} has no project in annotations or metadata"
            )

        try:
            duration_sec = float(sound["duration"])
            sample_rate_value = float(sound["sample_rate"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"sound {sound_id!r} has invalid duration or sample_rate"
            ) from error
        if not sample_rate_value.is_integer():
            raise ValueError(f"sound {sound_id!r} sample_rate must be an integer")
        source_sample_rate = int(sample_rate_value)
        stride_sec = segment_stride_sec(project)

        for bounds in iter_full_segments(
            duration_sec=duration_sec,
            source_sample_rate=source_sample_rate,
            project=project,
        ):
            records.append(
                SegmentRecord(
                    manifest_version=MANIFEST_VERSION,
                    segment_id=(
                        f"segment-v{MANIFEST_VERSION}-"
                        f"{quote(sound_id, safe='')}-{bounds.segment_index:04d}"
                    ),
                    sound_id=sound_id,
                    segment_index=bounds.segment_index,
                    audio_file=audio_file,
                    project=project,
                    start_sec_in_file=bounds.start_sec,
                    end_sec_in_file=bounds.end_sec,
                    start_sample=bounds.start_sample,
                    end_sample=bounds.end_sample,
                    source_sample_rate=source_sample_rate,
                    segment_duration_sec=SEGMENT_DURATION_SEC,
                    segment_stride_sec=stride_sec,
                    date_recorded=_metadata_value(
                        sound, metadata, "date_recorded"
                    ),
                    location_id=_metadata_value(sound, metadata, "location_id"),
                    recorder_id=_metadata_value(sound, metadata, "recorder_id"),
                )
            )

    records.sort(
        key=lambda record: (
            _sound_sort_key(record.sound_id),
            record.segment_index,
        )
    )
    return records


def write_manifest(
    records: Iterable[SegmentRecord], output_csv: Union[str, Path]
) -> None:
    """Write records as a deterministic, versioned CSV manifest."""
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=MANIFEST_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def generate_manifest(
    annotations_json: Union[str, Path],
    output_csv: Union[str, Path],
    metadata_csv: Optional[Union[str, Path]] = None,
) -> List[SegmentRecord]:
    """Generate and write a segment manifest, returning its records."""
    records = generate_manifest_records(annotations_json, metadata_csv)
    write_manifest(records, output_csv)
    return records


def read_manifest(manifest_csv: Union[str, Path]) -> List[SegmentRecord]:
    """Load and validate segment records from a manifest CSV."""
    with open(manifest_csv, newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing = [
            field
            for field in MANIFEST_FIELDS
            if field not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"manifest CSV is missing fields: {', '.join(missing)}")

        records = []
        for row_number, row in enumerate(reader, start=2):
            try:
                record = SegmentRecord(
                    manifest_version=row["manifest_version"],
                    segment_id=row["segment_id"],
                    sound_id=row["sound_id"],
                    segment_index=int(row["segment_index"]),
                    audio_file=row["audio_file"],
                    project=row["project"],
                    start_sec_in_file=float(row["start_sec_in_file"]),
                    end_sec_in_file=float(row["end_sec_in_file"]),
                    start_sample=int(row["start_sample"]),
                    end_sample=int(row["end_sample"]),
                    source_sample_rate=int(row["source_sample_rate"]),
                    segment_duration_sec=float(row["segment_duration_sec"]),
                    segment_stride_sec=float(row["segment_stride_sec"]),
                    date_recorded=row["date_recorded"],
                    location_id=row["location_id"],
                    recorder_id=row["recorder_id"],
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"manifest CSV row {row_number} contains invalid values"
                ) from error
            if record.manifest_version != MANIFEST_VERSION:
                raise ValueError(
                    f"unsupported manifest version {record.manifest_version!r}"
                )
            records.append(record)
    return records


def select_segment(
    records: Sequence[SegmentRecord], segment_id: str
) -> SegmentRecord:
    """Select exactly one manifest segment by its stable ID."""
    matches = [record for record in records if record.segment_id == segment_id]
    if not matches:
        raise ValueError(f"segment_id {segment_id!r} was not found")
    if len(matches) > 1:
        raise ValueError(f"segment_id {segment_id!r} is duplicated")
    return matches[0]


def _audio_path(audio_root: Union[str, Path], audio_file: str) -> Path:
    portable = audio_file.replace("\\", "/")
    relative = PurePosixPath(portable)
    if relative.is_absolute() or PureWindowsPath(audio_file).is_absolute():
        raise ValueError("manifest audio_file must be relative to audio_root")
    if not relative.parts or ".." in relative.parts:
        raise ValueError("manifest audio_file must be a normalized relative path")

    root = Path(audio_root).resolve()
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("manifest audio_file resolves outside audio_root") from error
    return path


def extract_segment(
    record: SegmentRecord,
    audio_root: Union[str, Path],
    output_wav: Union[str, Path],
) -> Path:
    """Extract one manifest segment from a PCM WAV with exact frame boundaries."""
    audio_path = _audio_path(audio_root, record.audio_file)
    expected_frames = record.end_sample - record.start_sample
    if record.start_sample < 0 or expected_frames <= 0:
        raise ValueError("manifest segment sample boundaries are invalid")

    with wave.open(str(audio_path), "rb") as source:
        if source.getframerate() != record.source_sample_rate:
            raise ValueError(
                "WAV sample rate does not match manifest source_sample_rate"
            )
        if record.end_sample > source.getnframes():
            raise ValueError("manifest segment sample boundaries exceed WAV frames")
        source.setpos(record.start_sample)
        frames = source.readframes(expected_frames)
        expected_bytes = (
            expected_frames * source.getnchannels() * source.getsampwidth()
        )
        if len(frames) != expected_bytes:
            raise ValueError("WAV extraction returned an incomplete frame range")
        channels = source.getnchannels()
        sample_width = source.getsampwidth()

    output_path = Path(output_wav)
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(record.source_sample_rate)
        output.setcomptype("NONE", "not compressed")
        output.writeframes(frames)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate manifests of original 10-second time-lapse segment offsets "
            "or extract a segment from a concatenated WAV."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="generate a versioned segment-offset CSV"
    )
    generate_parser.add_argument("--annotations", required=True)
    generate_parser.add_argument(
        "--output",
        default=DEFAULT_MANIFEST_FILENAME,
        help=f"output CSV (default: {DEFAULT_MANIFEST_FILENAME})",
    )
    generate_parser.add_argument("--metadata")

    extract_parser = subparsers.add_parser(
        "extract", help="extract one manifest segment to a WAV"
    )
    extract_parser.add_argument("--manifest", required=True)
    extract_parser.add_argument("--segment-id", required=True)
    extract_parser.add_argument("--audio-root", required=True)
    extract_parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the segment manifest command-line interface."""
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        records = generate_manifest(args.annotations, args.output, args.metadata)
        print(f"Wrote {len(records)} segments to {args.output}")
    else:
        record = select_segment(read_manifest(args.manifest), args.segment_id)
        output_path = extract_segment(record, args.audio_root, args.output)
        print(f"Extracted {record.segment_id} to {output_path}")
    return 0


if __name__ == "__main__":
    main()
