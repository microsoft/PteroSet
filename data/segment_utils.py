"""Shared geometry helpers for concatenated time-lapse segments."""

from dataclasses import dataclass
import math
from typing import Iterator


SEGMENT_DURATION_SEC = 10.0
PPA1_SEGMENT_STRIDE_SEC = 9.0
SEGMENT_DURATION_TOLERANCE_SEC = 1e-6


@dataclass(frozen=True)
class SegmentBounds:
    """Offsets for one original time-lapse segment within a concatenated file."""

    segment_index: int
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int


def segment_stride_sec(
    project: str, segment_duration_sec: float = SEGMENT_DURATION_SEC
) -> float:
    """Return the segment-start stride for a project."""
    return (
        PPA1_SEGMENT_STRIDE_SEC
        if project == "PPA1"
        else float(segment_duration_sec)
    )


def iter_full_segments(
    duration_sec: float,
    source_sample_rate: int,
    project: str,
    segment_duration_sec: float = SEGMENT_DURATION_SEC,
    tolerance_sec: float = SEGMENT_DURATION_TOLERANCE_SEC,
) -> Iterator[SegmentBounds]:
    """Yield full original segments whose end is within the file duration."""
    duration_sec = float(duration_sec)
    segment_duration_sec = float(segment_duration_sec)
    tolerance_sec = float(tolerance_sec)
    if not math.isfinite(duration_sec) or duration_sec < 0:
        raise ValueError("duration_sec must be a finite non-negative number")
    if not math.isfinite(segment_duration_sec) or segment_duration_sec <= 0:
        raise ValueError("segment_duration_sec must be a finite positive number")
    if not math.isfinite(tolerance_sec) or tolerance_sec < 0:
        raise ValueError("tolerance_sec must be a finite non-negative number")
    if (
        isinstance(source_sample_rate, bool)
        or int(source_sample_rate) != source_sample_rate
    ):
        raise ValueError("source_sample_rate must be a positive integer")
    source_sample_rate = int(source_sample_rate)
    if source_sample_rate <= 0:
        raise ValueError("source_sample_rate must be a positive integer")

    stride_sec = segment_stride_sec(project, segment_duration_sec)
    segment_index = 0
    while True:
        start_sec = segment_index * stride_sec
        end_sec = start_sec + segment_duration_sec
        if end_sec > duration_sec + tolerance_sec:
            break
        yield SegmentBounds(
            segment_index=segment_index,
            start_sec=start_sec,
            end_sec=end_sec,
            start_sample=round(start_sec * source_sample_rate),
            end_sample=round(end_sec * source_sample_rate),
        )
        segment_index += 1


def window_is_contained_in_segment(
    start_sample: int,
    end_sample: int,
    project: str,
    sample_rate: int,
    segment_duration_sec: float = SEGMENT_DURATION_SEC,
) -> bool:
    """Return whether a sample window fits wholly within one original segment."""
    stride_samples = round(
        segment_stride_sec(project, segment_duration_sec) * sample_rate
    )
    segment_duration_samples = round(segment_duration_sec * sample_rate)
    segment_index = start_sample // stride_samples

    for candidate_index in (segment_index, segment_index - 1):
        if candidate_index < 0:
            continue
        segment_start = candidate_index * stride_samples
        segment_end = segment_start + segment_duration_samples
        if start_sample >= segment_start and end_sample <= segment_end:
            return True
    return False
