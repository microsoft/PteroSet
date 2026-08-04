"""
Extract Perch v2 embeddings for the species-classification train/val/test splits.

Why
---
``train_perch_logreg.py`` needs a frozen 1536-dim embedding per window, not raw
audio. Perch v2's SavedModel is CUDA-only (XLA-compiled with
``platforms=[CUDA]``), so the CUDA/cuDNN libraries pip-installs into
``site-packages/nvidia/*`` must be on ``LD_LIBRARY_PATH`` *before* the Python
process's dynamic linker starts resolving TensorFlow's shared libraries -- an
``os.environ[...] = ...`` after ``import tensorflow`` is too late. This module
adapts the working re-exec shim from
``../orcas_dclde2026/eval_perch_ecotype.py`` / ``extract_perch_3class.py``,
but fixes their one dangerous shortcut: those scripts fall back to a
zero-filled ("silent") waveform when a window's audio fails to load, which
silently manufactures a fake no-bird training example. Here, any audio load,
preprocessing, or model-inference failure is recorded (never zero-filled) and
counted against a global and a per-split failure ceiling that defaults to
zero -- the run aborts loudly the instant either ceiling is exceeded.

What
----
Two mutually exclusive modes:

- ``--download-v2``: fetch the Perch v2 SavedModel from Kaggle
  (``google/bird-vocalization-classifier/tensorFlow2/perch_v2``) into
  ``--model-dir`` (default ``checkpoints/perch/model_v2``) and exit.
- ``--extract``: for each split in ``--splits`` (default ``train val test``),
  load ``{split}_split.csv`` from ``--split-dir`` (default
  ``data/splits_species_v1``), validate its schema and identity columns,
  batch-run Perch v2 on every window's audio, and write
  ``{split}_emb.npz`` + a deterministic ``embedding_manifest.json`` to
  ``--output-dir`` (default ``data/embeddings/perch_v2/species_v1``).

How
---
Identity discipline matches ``prepare_species_splits.py``: every gate and
join uses the integer columns (``window_id``, ``sound_id``, ``start``,
``end``, ``sample_rate``); seconds are computed exactly once, transiently,
immediately before the ``librosa.load(..., offset=start/sample_rate,
duration=(end-start)/sample_rate)`` call, and are never persisted, compared,
or hashed.

CSV schema validation (unique ``window_id``, exact 5 s duration, integer
identity fields, allowed ``label_state`` values, ``target_vector`` binary and
consistent with ``target_codes`` and ``class_list.json`` order, no_bird
all-zero / species_window nonzero, referenced audio file exists) is an
unconditional gate that raises before any extraction begins -- it is a data-
integrity check, not an "audio failure," so it is never subject to
``--max-failures``.

Audio *load* failures (a window's own recording is missing/corrupt) and
*model-inference* failures (a batch forward pass errors or returns a
malformed output) are the two failure classes the ceilings apply to. A
failed window is dropped from that split's NPZ (arrays stay index-aligned
for the windows that succeeded) and recorded in
``embedding_manifest.json``'s ``excluded`` map with its identity and reason.

Every output write (NPZ, manifest) is atomic: written to a temp file in the
same directory, then ``os.replace``'d over the final path, so a killed
process never leaves a half-written artifact. The manifest is rewritten
after every split completes, so an interrupted multi-split run retains the
splits it already finished.
"""

from __future__ import annotations

# ── CUDA-libs bootstrap: must run before any TensorFlow import ──────────────
#
# Perch v2's SavedModel is XLA-compiled for CUDA only and refuses to run on
# CPU. The dynamic loader resolves LD_LIBRARY_PATH once, at process start, so
# setting it from Python after TensorFlow (or anything that imports it) has
# already loaded does nothing. `_ensure_cuda_libs_in_ldlibpath` re-execs this
# same process with the pip-installed `nvidia/*/lib` directories prepended to
# LD_LIBRARY_PATH, idempotently (guarded by a marker env var so it re-execs at
# most once). It must be called at the very start of `main()`, before any
# other code path can trigger a TensorFlow import -- never at module import
# time, so importing this module (e.g. from tests) never re-execs the
# interpreter.
import os
import site
import sys
from pathlib import Path as _Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

_CUDA_BOOTSTRAP_MARKER = "_PERCH_CUDA_LDLP_BOOTSTRAPPED"


def _ensure_cuda_libs_in_ldlibpath() -> None:
    """Re-exec the process with pip-installed CUDA lib dirs on LD_LIBRARY_PATH.

    Idempotent: does nothing if already re-exec'd (marker env var) or if no
    ``nvidia`` package tree is found in any site-packages directory (e.g. a
    CPU-only environment -- the caller is still responsible for the hard GPU
    check after this returns; this function only makes the libraries
    reachable, it does not guarantee a GPU is present).
    """
    if os.environ.get(_CUDA_BOOTSTRAP_MARKER) == "1":
        return
    nv_libs: list[str] = []
    for sp in site.getsitepackages():
        nv_root = _Path(sp) / "nvidia"
        if not nv_root.is_dir():
            continue
        for sub in sorted(nv_root.iterdir()):
            lib = sub / "lib"
            if lib.is_dir():
                nv_libs.append(str(lib))
    if not nv_libs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d in current for d in nv_libs):
        return
    new_ldlp = ":".join(nv_libs + ([current] if current else []))
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = new_ldlp
    new_env[_CUDA_BOOTSTRAP_MARKER] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], new_env)


# ── end CUDA-libs bootstrap ──────────────────────────────────────────────────

import argparse
import csv
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PERCH_SR = 32_000
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(PERCH_SR * WINDOW_SEC)  # 160,000
EMBEDDING_DIM = 1536
TARGET_PEAK = 0.25
# Bounded read-length rounding tolerance (in target-sample-rate samples): a
# resample of an exact 5.0 s request can legitimately land a handful of
# samples short/long of WINDOW_SAMPLES due to floating-point ratio rounding.
# Anything beyond this is treated as a meaningful truncation, not rounding.
ROUNDING_TOLERANCE_SAMPLES = 8

KAGGLE_SLUG = "google/bird-vocalization-classifier/tensorFlow2/perch_v2"
DEFAULT_MODEL_DIR = "checkpoints/perch/model_v2"
DEFAULT_SPLIT_DIR = "data/splits_species_v1"
DEFAULT_OUTPUT_DIR = "data/embeddings/perch_v2/species_v1"
DEFAULT_AUDIO_ROOT = "data/audios_48khz"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_FAILURES = 0
DEFAULT_MAX_FAILURES_PER_SPLIT = 0
UNVERIFIED_LICENSE = "UNVERIFIED"
# Verified 2026-07-29 against the two upstream sources directly (not
# invented, and not a Kaggle-specific license string -- Kaggle mirrors the
# same weights but states no license of its own):
#   weights: https://huggingface.co/cgeorgiaw/Perch/raw/main/README.md
#            ("license: apache-2.0" in the model-card front matter)
#   code:    https://github.com/google-research/perch-hoplite
#            (LICENSE file: Apache License, Version 2.0)
DEFAULT_WEIGHTS_LICENSE = "Apache-2.0"
DEFAULT_WEIGHTS_LICENSE_SOURCE_URL = "https://huggingface.co/cgeorgiaw/Perch/raw/main/README.md"
DEFAULT_CODE_LICENSE = "Apache-2.0"
DEFAULT_CODE_LICENSE_SOURCE_URL = "https://github.com/google-research/perch-hoplite"
# "log each ~1000 windows" during the sound_id-grouped decode phase.
PROGRESS_LOG_INTERVAL = 1000

SPLIT_NAMES = ("train", "val", "test")

REQUIRED_SPLIT_COLUMNS = (
    "window_id",
    "dataset",
    "sample_rate",
    "sound_id",
    "start",
    "end",
    "project",
    "is_canonical",
    "label_state",
    "spec_name",
    "sound_filename",
    "target_codes",
    "target_vector",
)

NO_BIRD = "no_bird"
SPECIES_WINDOW = "species_window"
ALLOWED_LABEL_STATES = frozenset({NO_BIRD, SPECIES_WINDOW})

_STRICT_INT_RE = re.compile(r"^-?\d+$")


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitRow:
    """One validated, identity-checked row from a ``{split}_split.csv``.

    ``original_index`` is the 0-based position of this row in its source CSV
    (post-header, pre-``--limit``), independent of any later sound_id-grouped
    processing order. ``validate_split_rows`` always returns rows already
    sorted by ``original_index`` (it assigns them while iterating the CSV in
    file order), and every downstream consumer (``extract_split``'s
    sound_id-grouped decode phase, its original-row-order inference-batching
    phase, and the final NPZ) preserves that order -- ``original_index`` is
    the authoritative sort key a caller can use to re-verify or restore it.
    """

    window_id: int
    dataset: str
    sample_rate: int
    sound_id: int
    start: int
    end: int
    project: str
    is_canonical: bool
    label_state: str
    spec_name: str
    sound_filename: str
    sound_filepath: str
    target_codes: Tuple[str, ...]
    target_vector: Tuple[int, ...]
    original_index: int


@dataclass
class FailureRecord:
    """One audio-load or model-inference failure, identified for the manifest."""

    split: str
    window_id: int
    sound_id: int
    reason: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "sound_id": self.sound_id,
            "reason": self.reason,
            "detail": self.detail,
        }


class FailureCeilingExceeded(RuntimeError):
    """Raised the instant a global or per-split failure ceiling is exceeded."""


@dataclass
class SplitProgress:
    """Mutable extraction progress for one split, preserved across an abort."""

    succeeded_rows: List[SplitRow] = field(default_factory=list)
    succeeded_embeddings: List[np.ndarray] = field(default_factory=list)
    split_failures: List[FailureRecord] = field(default_factory=list)
    n_padded: int = 0


# --------------------------------------------------------------------------
# Hashing / identity helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file, streamed to bound memory use."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_model_dir(model_dir: Path) -> str:
    """Deterministic sha256 over a sorted listing of a SavedModel dir's files.

    Combines each file's POSIX-relative path and content hash so both a
    renamed/added/removed file and a changed-content file change the result.
    """
    model_dir = Path(model_dir)
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(model_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def get_git_commit(repo_dir: Path) -> Tuple[Optional[str], bool]:
    """Return ``(commit_sha, dirty)``. ``commit_sha`` is ``None`` when dirty.

    A commit sha alone does not describe uncommitted working-tree changes, so
    a dirty tree records ``git_commit = None`` plus ``git_dirty = True``
    rather than a sha that silently overstates reproducibility.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None, False
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_dir), stderr=subprocess.DEVNULL
        ).decode()
        dirty = bool(status.strip())
    except Exception:
        dirty = False
    return (None if dirty else sha), dirty


def get_dependency_versions() -> Dict[str, Optional[str]]:
    """Resolved package versions via importlib.metadata (no forced imports)."""
    names = ("tensorflow", "scikit-learn", "kagglehub", "numpy", "librosa", "pandas")
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


# --------------------------------------------------------------------------
# CSV / class-list validation
# --------------------------------------------------------------------------


def parse_strict_int(value: str, field_name: str, window_id: Any = None) -> int:
    """Parse ``value`` as an int, rejecting float-looking strings like "3.0"."""
    stripped = value.strip()
    if not _STRICT_INT_RE.match(stripped):
        raise ValueError(
            f"window_id={window_id}: field {field_name!r} is not a strict integer: {value!r}"
        )
    return int(stripped)


def parse_bool01(value: str, field_name: str, window_id: Any = None) -> bool:
    stripped = value.strip()
    if stripped not in ("0", "1"):
        raise ValueError(
            f"window_id={window_id}: field {field_name!r} must be '0' or '1', got {value!r}"
        )
    return stripped == "1"


def parse_target_codes(raw: str) -> Tuple[str, ...]:
    """Semicolon-separated species codes; empty string -> empty tuple."""
    stripped = raw.strip()
    if not stripped:
        return ()
    return tuple(code.strip() for code in stripped.split(";"))


def parse_target_vector(raw: str, k: int, window_id: Any = None) -> Tuple[int, ...]:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"window_id={window_id}: target_vector is not valid JSON: {raw!r}") from exc
    if not isinstance(values, list) or len(values) != k:
        raise ValueError(
            f"window_id={window_id}: target_vector must be a length-{k} JSON list, got {raw!r}"
        )
    result = []
    for v in values:
        if v not in (0, 1):
            raise ValueError(
                f"window_id={window_id}: target_vector must be binary, found {v!r} in {raw!r}"
            )
        result.append(int(v))
    return tuple(result)


def load_class_list(path: Path) -> List[str]:
    """Load ``class_list.json`` and return codes ordered by ``index``.

    Validates that ``index`` is a contiguous 0..len(codes)-1 permutation, so
    downstream ``target_vector`` positions are unambiguous.
    """
    with open(path, "r") as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: class_list.json must be a non-empty JSON list")
    by_index: Dict[int, str] = {}
    for entry in entries:
        if "index" not in entry or "code" not in entry:
            raise ValueError(f"{path}: class_list.json entries need 'index' and 'code': {entry!r}")
        by_index[int(entry["index"])] = str(entry["code"])
    expected_indices = set(range(len(entries)))
    if set(by_index.keys()) != expected_indices:
        raise ValueError(
            f"{path}: class_list.json 'index' values must be exactly 0..{len(entries) - 1}"
        )
    return [by_index[i] for i in range(len(entries))]


def validate_header(fieldnames: Optional[Sequence[str]], csv_path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"{csv_path}: empty CSV, no header row")
    present = set(fieldnames)
    required = set(REQUIRED_SPLIT_COLUMNS)
    missing = required - present
    extra = present - required
    if missing or extra:
        raise ValueError(
            f"{csv_path}: header does not match required columns exactly. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


def validate_split_rows(
    csv_path: Path,
    split_name: str,
    class_codes: Sequence[str],
    audio_root: Path,
) -> List[SplitRow]:
    """Load and fully validate one ``{split}_split.csv``.

    This is an unconditional, fail-loud gate for malformed identity/label
    data -- a data-integrity bug, distinct from an audio-load or
    model-inference failure -- so it is never subject to ``--max-failures``.
    """
    k = len(class_codes)
    code_to_index = {code: i for i, code in enumerate(class_codes)}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        validate_header(reader.fieldnames, csv_path)
        raw_rows = list(reader)

    rows: List[SplitRow] = []
    seen_window_ids: set = set()
    for original_index, raw in enumerate(raw_rows):
        window_id = parse_strict_int(raw["window_id"], "window_id")
        if window_id in seen_window_ids:
            raise ValueError(f"{csv_path}: duplicate window_id {window_id}")
        seen_window_ids.add(window_id)

        sound_id = parse_strict_int(raw["sound_id"], "sound_id", window_id)
        start = parse_strict_int(raw["start"], "start", window_id)
        end = parse_strict_int(raw["end"], "end", window_id)
        sample_rate = parse_strict_int(raw["sample_rate"], "sample_rate", window_id)
        is_canonical = parse_bool01(raw["is_canonical"], "is_canonical", window_id)

        if start < 0 or end <= start:
            raise ValueError(f"window_id={window_id}: non-positive duration ({start}..{end})")
        expected_span = int(round(WINDOW_SEC * sample_rate))
        if end - start != expected_span:
            raise ValueError(
                f"window_id={window_id}: duration is not exactly {WINDOW_SEC}s at "
                f"sample_rate={sample_rate} (got {end - start} samples, expected {expected_span})"
            )

        label_state = raw["label_state"].strip()
        if label_state not in ALLOWED_LABEL_STATES:
            raise ValueError(
                f"window_id={window_id}: label_state {label_state!r} not in {sorted(ALLOWED_LABEL_STATES)}"
            )

        target_codes = parse_target_codes(raw["target_codes"])
        target_vector = parse_target_vector(raw["target_vector"], k, window_id)

        positive_idxs = {i for i, v in enumerate(target_vector) if v == 1}
        if label_state == NO_BIRD:
            if positive_idxs:
                raise ValueError(f"window_id={window_id}: no_bird row has a nonzero target_vector")
            if target_codes:
                raise ValueError(f"window_id={window_id}: no_bird row has nonempty target_codes")
        else:  # SPECIES_WINDOW
            if not positive_idxs:
                raise ValueError(f"window_id={window_id}: species_window row has an all-zero target_vector")

        code_idxs = set()
        for code in target_codes:
            if code not in code_to_index:
                raise ValueError(
                    f"window_id={window_id}: target_codes contains unknown class code {code!r}"
                )
            code_idxs.add(code_to_index[code])
        if code_idxs != positive_idxs:
            raise ValueError(
                f"window_id={window_id}: target_codes {sorted(target_codes)} do not match the "
                f"1-bit positions of target_vector"
            )

        sound_filename = raw["sound_filename"].strip()
        filename_path = Path(sound_filename)
        if (
            not sound_filename
            or filename_path.is_absolute()
            or filename_path.name != sound_filename
            or ".." in filename_path.parts
        ):
            raise ValueError(
                f"window_id={window_id}: sound_filename must be a bare filename within "
                f"audio_root, got {sound_filename!r}"
            )
        resolved_audio_root = Path(audio_root).resolve()
        resolved_sound_path = (resolved_audio_root / sound_filename).resolve()
        if not resolved_sound_path.is_relative_to(resolved_audio_root):
            raise ValueError(
                f"window_id={window_id}: sound_filename escapes audio_root: {sound_filename!r}"
            )
        sound_filepath = str(resolved_sound_path)
        if not os.path.isfile(sound_filepath):
            raise ValueError(
                f"window_id={window_id}: sound file not found: {sound_filepath} "
                f"(sound_filename={sound_filename!r})"
            )

        rows.append(
            SplitRow(
                window_id=window_id,
                dataset=raw["dataset"].strip(),
                sample_rate=sample_rate,
                sound_id=sound_id,
                start=start,
                end=end,
                project=raw["project"].strip(),
                is_canonical=is_canonical,
                label_state=label_state,
                spec_name=raw["spec_name"].strip(),
                sound_filename=sound_filename,
                sound_filepath=sound_filepath,
                target_codes=target_codes,
                target_vector=target_vector,
                original_index=original_index,
            )
        )

    print(f"  [{split_name}] validated {len(rows):,} rows from {csv_path}")
    return rows


# --------------------------------------------------------------------------
# Audio preprocessing
# --------------------------------------------------------------------------


def compute_offset_duration_sec(start: int, end: int, sample_rate: int) -> Tuple[float, float]:
    """Compute (offset_sec, duration_sec) transiently -- never persisted."""
    return start / sample_rate, (end - start) / sample_rate


def preprocess_waveform(
    y: np.ndarray,
    *,
    window_samples: int = WINDOW_SAMPLES,
    tolerance_samples: int = ROUNDING_TOLERANCE_SAMPLES,
    target_peak: float = TARGET_PEAK,
) -> Tuple[np.ndarray, int]:
    """Pad/truncate to exactly ``window_samples`` and peak-normalize.

    Only *bounded read-length rounding* (within ``tolerance_samples``) is
    padded/truncated silently; anything beyond that tolerance is a meaningful
    truncation and raises. Silence (an all-zero segment) is a valid no-bird
    waveform and is returned unmodified (not treated as a failure). Returns
    ``(waveform, n_padded)`` where ``n_padded`` is the number of zero samples
    appended (0 if none).
    """
    y = np.asarray(y, dtype=np.float32)
    n = y.shape[0]
    deficit = window_samples - n
    if deficit > tolerance_samples:
        raise ValueError(
            f"audio segment too short: got {n} samples, expected {window_samples} "
            f"(deficit {deficit} exceeds rounding tolerance {tolerance_samples} -- "
            "this is a meaningful truncation, not bounded read-length rounding)"
        )
    excess = n - window_samples
    if excess > tolerance_samples:
        raise ValueError(
            f"audio segment too long: got {n} samples, expected {window_samples} "
            f"(excess {excess} exceeds rounding tolerance {tolerance_samples})"
        )

    n_padded = 0
    if deficit > 0:
        y = np.pad(y, (0, deficit))
        n_padded = int(deficit)
    y = y[:window_samples].astype(np.float32)
    if y.shape[0] != window_samples:
        raise AssertionError("internal error: post pad/truncate length mismatch")

    if not np.all(np.isfinite(y)):
        raise ValueError("audio segment contains non-finite values")

    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak > 0:
        y = (y * (target_peak / peak)).astype(np.float32)
    # peak == 0: genuine silence, a valid no-bird waveform -- leave as zeros.

    return y, n_padded


def load_audio_segment(
    filepath: str,
    start: int,
    end: int,
    sample_rate: int,
    *,
    target_sr: int = PERCH_SR,
    target_peak: float = TARGET_PEAK,
    window_samples: int = WINDOW_SAMPLES,
    tolerance_samples: int = ROUNDING_TOLERANCE_SAMPLES,
) -> Tuple[np.ndarray, int]:
    """Load, resample, and preprocess one window's audio.

    ``offset``/``duration`` seconds are computed transiently here, exactly
    once, from the integer ``start``/``end``/``sample_rate`` identity columns
    -- never persisted or compared elsewhere.
    """
    import librosa

    offset_sec, duration_sec = compute_offset_duration_sec(start, end, sample_rate)
    if duration_sec <= 0:
        raise ValueError(f"non-positive duration computed for {filepath}: {duration_sec}s")
    y, _ = librosa.load(filepath, sr=target_sr, offset=offset_sec, duration=duration_sec, mono=True)
    return preprocess_waveform(
        y,
        window_samples=window_samples,
        tolerance_samples=tolerance_samples,
        target_peak=target_peak,
    )


def group_rows_by_sound_id(rows: Sequence[SplitRow]) -> "OrderedDict[int, List[int]]":
    """Map ``sound_id`` -> original-row indices, in first-encounter order.

    Grouping (not sorting) lets every window belonging to one audio file
    share a single decode+resample pass while the caller still assembles
    results back in the split CSV's original row order -- the group's key
    order is irrelevant, only each group's *contents* matter, since the
    caller re-walks ``rows`` by original index afterward.
    """
    groups: "OrderedDict[int, List[int]]" = OrderedDict()
    for i, row in enumerate(rows):
        groups.setdefault(row.sound_id, []).append(i)
    return groups


def load_full_audio(filepath: str, target_sr: int = PERCH_SR) -> np.ndarray:
    """Load and resample one sound file's *entire* waveform, once.

    Every window sharing a ``sound_id`` slices this same in-memory array
    instead of each re-decoding the file via its own
    ``librosa.load(..., offset=..., duration=...)`` call -- for a heavily
    windowed dataset (many, often overlapping, windows per file) the
    dominant extraction cost is redundant decode/seek work, not the forward
    pass itself.
    """
    import librosa

    y, _ = librosa.load(filepath, sr=target_sr, mono=True)
    return np.asarray(y, dtype=np.float32)


def slice_window_from_full_audio(
    y_full: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
    *,
    target_sr: int = PERCH_SR,
    target_peak: float = TARGET_PEAK,
    window_samples: int = WINDOW_SAMPLES,
    tolerance_samples: int = ROUNDING_TOLERANCE_SAMPLES,
) -> Tuple[np.ndarray, int]:
    """Slice one window out of an already-loaded, already-resampled waveform.

    ``offset``/``duration`` seconds are computed transiently here, exactly
    once, from the integer ``start``/``end``/``sample_rate`` identity
    columns -- never persisted or compared elsewhere -- matching the
    single-window ``load_audio_segment`` path's identity discipline exactly,
    just applied to an in-memory slice instead of a fresh file read.
    """
    offset_sec, duration_sec = compute_offset_duration_sec(start, end, sample_rate)
    if duration_sec <= 0:
        raise ValueError(f"non-positive duration computed: {duration_sec}s")
    start_idx = int(round(offset_sec * target_sr))
    end_idx = start_idx + int(round(duration_sec * target_sr))
    if start_idx < 0:
        raise ValueError(f"negative slice start computed: offset_sec={offset_sec}")
    y = y_full[start_idx:end_idx]
    return preprocess_waveform(
        y,
        window_samples=window_samples,
        tolerance_samples=tolerance_samples,
        target_peak=target_peak,
    )


def preflight_validate_audio_files(filepaths: Sequence[str]) -> None:
    """Header-only (no full decode) existence/readability check over unique audio files.

    Run over the de-duplicated set of every ``sound_filepath`` referenced by
    every requested split, *before* GPU acquisition or model loading, so a
    missing/corrupt file is caught immediately rather than partway through a
    long extraction run. Uses ``soundfile.info`` (a lightweight header read)
    rather than a full ``librosa.load`` decode of every file. Raises
    ``ValueError`` listing every failing path on the first pass over all of
    them (not just the first failure), so a single run surfaces every
    problem at once.
    """
    import soundfile as sf

    unique_paths = sorted(set(filepaths))
    failures: List[str] = []
    for path in unique_paths:
        try:
            info = sf.info(path)
        except Exception as exc:  # noqa: BLE001 - any header-read failure is a preflight failure
            failures.append(f"{path}: {exc}")
            continue
        if info.frames <= 0:
            failures.append(f"{path}: audio file has zero frames")

    if failures:
        raise ValueError(
            f"preflight audio validation failed for {len(failures)}/{len(unique_paths)} unique file(s):\n"
            + "\n".join(failures)
        )


def format_eta(seconds: float) -> str:
    """Format a non-negative duration in seconds as ``H:MM:SS``."""
    total = int(round(max(0.0, float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def report_progress(
    split_name: str,
    n_processed: int,
    n_total: int,
    elapsed_sec: float,
    interval: int = PROGRESS_LOG_INTERVAL,
) -> None:
    """Print progress + throughput + ETA roughly every ``interval`` rows, and at the end."""
    if n_total <= 0 or (n_processed != n_total and n_processed % interval != 0):
        return
    rate = n_processed / elapsed_sec if elapsed_sec > 0 else 0.0
    eta_sec = (n_total - n_processed) / rate if rate > 0 else 0.0
    print(
        f"  [{split_name}] decoded {n_processed:,}/{n_total:,} windows "
        f"({rate:.1f}/s, elapsed {format_eta(elapsed_sec)}, ETA {format_eta(eta_sec)})"
    )


# --------------------------------------------------------------------------
# Model loading / signature inspection
# --------------------------------------------------------------------------


def select_embedding_key(structured_outputs: Mapping[str, Any], expected_dim: int = EMBEDDING_DIM) -> str:
    """Pick the serving_default output whose last dim is exactly ``expected_dim``.

    Hard-asserts on the exact expected embedding dimensionality -- no "pick
    the smallest non-logit output" heuristic that could silently select the
    wrong tensor if the model's output ordering or shapes ever change.
    Raises if zero or more than one output matches.
    """
    candidates = []
    for name, spec in structured_outputs.items():
        shape = getattr(spec, "shape", None)
        if shape is None:
            continue
        rank = getattr(shape, "rank", None)
        if rank != 2:
            continue
        dim = shape[-1]
        if dim == expected_dim:
            candidates.append(name)
    if len(candidates) == 0:
        available = {
            name: (getattr(getattr(spec, "shape", None), "rank", None), getattr(spec, "shape", None))
            for name, spec in structured_outputs.items()
        }
        raise ValueError(
            f"No serving_default output with a rank-2 shape and last dim == {expected_dim} "
            f"found. Available outputs: {available}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple serving_default outputs have last dim == {expected_dim}: {candidates}. "
            "Cannot disambiguate the embedding output automatically."
        )
    return candidates[0]


def load_perch_model(model_dir: Path):
    import tensorflow as tf

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Perch v2 SavedModel not found at {model_dir}. Run with --download-v2 first."
        )
    print(f"Loading Perch v2 SavedModel from: {model_dir}")
    model = tf.saved_model.load(str(model_dir))
    print("Perch v2 loaded.")
    return model


def inspect_model_outputs(model) -> str:
    """Print serving_default's structured outputs and return the embedding key."""
    sig = model.signatures["serving_default"]
    print("\nPerch v2 serving_default structured outputs:")
    for name, spec in sig.structured_outputs.items():
        print(f"  {name}: shape={spec.shape}, dtype={spec.dtype}")
    emb_key = select_embedding_key(sig.structured_outputs)
    print(f"  -> selected {emb_key!r} as the {EMBEDDING_DIM}-dim embedding output")
    return emb_key


def download_perch_v2(dest_dir: Path, kaggle_slug: str = KAGGLE_SLUG) -> None:
    import kagglehub

    print(f"Downloading Perch v2 from Kaggle: {kaggle_slug}")
    cache_path = Path(kagglehub.model_download(kaggle_slug))
    print(f"Kaggle cache path: {cache_path}")

    dest_dir = Path(dest_dir)
    if cache_path.resolve() != dest_dir.resolve():
        dest_dir.mkdir(parents=True, exist_ok=True)
        for item in cache_path.iterdir():
            dst = dest_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
    print(f"Perch v2 saved to: {dest_dir}")


def validate_license_metadata(
    weights_license: str,
    weights_license_source_url: str,
    code_license: str,
    code_license_source_url: str,
) -> None:
    """Require explicit, sourced license metadata before download/extraction."""
    fields = {
        "weights license": weights_license,
        "weights license source URL": weights_license_source_url,
        "code license": code_license,
        "code license source URL": code_license_source_url,
    }
    invalid = [
        name
        for name, value in fields.items()
        if not value.strip() or value.strip().upper() == UNVERIFIED_LICENSE
    ]
    if invalid:
        raise SystemExit(
            "Perch license metadata is not verified: "
            + ", ".join(invalid)
            + ". Supply reviewed license values and source URLs before downloading or extracting."
        )


def require_gpu_or_exit() -> List[str]:
    """Hard-fail if no GPU is visible after the CUDA bootstrap.

    Perch v2's SavedModel is CUDA-only; this never falls back to CPU.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit(
            "No GPU visible to TensorFlow after the CUDA library bootstrap. Perch v2's "
            "SavedModel is XLA-compiled for CUDA only and does not run on CPU; this "
            "script refuses to silently fall back to CPU. Fix the CUDA driver/library "
            "setup (or move extraction to a GPU host) and rerun."
        )
    return [g.name for g in gpus]


def validate_embedding_batch_shape(
    embeddings: np.ndarray, expected_batch_size: int, expected_dim: int = EMBEDDING_DIM
) -> None:
    """Validate a model output batch: shape [B, dim], finite, convertible to float32."""
    if embeddings.ndim != 2:
        raise ValueError(f"expected a rank-2 embedding batch, got shape {embeddings.shape}")
    if embeddings.shape[0] != expected_batch_size:
        raise ValueError(
            f"embedding batch size {embeddings.shape[0]} does not match input batch size {expected_batch_size}"
        )
    if embeddings.shape[1] != expected_dim:
        raise ValueError(f"embedding dim {embeddings.shape[1]} != expected {expected_dim}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("model produced non-finite embedding values")


def run_model_batch(model, emb_key: str, batch: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    sig = model.signatures["serving_default"]
    outputs = sig(inputs=tf.constant(batch))
    embeddings = np.asarray(outputs[emb_key].numpy())
    validate_embedding_batch_shape(embeddings, expected_batch_size=batch.shape[0])
    return embeddings.astype(np.float32)


# --------------------------------------------------------------------------
# Failure-ceiling enforcement
# --------------------------------------------------------------------------


def check_failure_ceilings(
    global_failures: int, split_failures: int, max_failures: int, max_failures_per_split: int
) -> None:
    """Raise the instant either ceiling is exceeded (checked after every failure)."""
    if global_failures > max_failures:
        raise FailureCeilingExceeded(
            f"global failure ceiling exceeded: {global_failures} failures > max_failures={max_failures}"
        )
    if split_failures > max_failures_per_split:
        raise FailureCeilingExceeded(
            f"per-split failure ceiling exceeded: {split_failures} failures > "
            f"max_failures_per_split={max_failures_per_split}"
        )


# --------------------------------------------------------------------------
# NPZ assembly / atomic IO
# --------------------------------------------------------------------------


def to_unicode_array(values: Sequence[str]) -> np.ndarray:
    """Fixed-width unicode array (no allow_pickle needed), even when empty."""
    if len(values) == 0:
        return np.array([], dtype="<U1")
    maxlen = max(1, max(len(v) for v in values))
    return np.array(list(values), dtype=f"<U{maxlen}")


def build_npz_arrays(rows: Sequence[SplitRow], embeddings: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    """Assemble the fixed NPZ schema for the rows that succeeded extraction."""
    n = len(rows)
    if embeddings.shape != (n, EMBEDDING_DIM):
        raise ValueError(
            f"embeddings shape {embeddings.shape} does not match {n} succeeded rows x {EMBEDDING_DIM}"
        )
    target_vector = np.zeros((n, k), dtype=np.uint8)
    for i, row in enumerate(rows):
        if len(row.target_vector) != k:
            raise ValueError(f"window_id={row.window_id}: target_vector length != {k}")
        target_vector[i, :] = np.asarray(row.target_vector, dtype=np.uint8)

    return {
        "embeddings": embeddings.astype(np.float32),
        "target_vector": target_vector,
        "target_codes": to_unicode_array([";".join(r.target_codes) for r in rows]),
        "label_state": to_unicode_array([r.label_state for r in rows]),
        "window_id": np.array([r.window_id for r in rows], dtype=np.int64),
        "sound_id": np.array([r.sound_id for r in rows], dtype=np.int64),
        "start": np.array([r.start for r in rows], dtype=np.int64),
        "end": np.array([r.end for r in rows], dtype=np.int64),
        "sample_rate": np.array([r.sample_rate for r in rows], dtype=np.int32),
        "sound_filepath": to_unicode_array([r.sound_filepath for r in rows]),
        "sound_filename": to_unicode_array([r.sound_filename for r in rows]),
        "dataset": to_unicode_array([r.dataset for r in rows]),
        "project": to_unicode_array([r.project for r in rows]),
        "is_canonical": np.array([1 if r.is_canonical else 0 for r in rows], dtype=np.uint8),
    }


NPZ_REQUIRED_KEYS = (
    "embeddings",
    "target_vector",
    "target_codes",
    "label_state",
    "window_id",
    "sound_id",
    "start",
    "end",
    "sample_rate",
    "sound_filepath",
    "sound_filename",
    "dataset",
    "project",
    "is_canonical",
)


def validate_npz_schema(arrays: Mapping[str, np.ndarray], k: int) -> None:
    """Validate an in-memory NPZ arrays mapping against the fixed schema."""
    missing = set(NPZ_REQUIRED_KEYS) - set(arrays.keys())
    if missing:
        raise ValueError(f"NPZ arrays missing required keys: {sorted(missing)}")

    n = arrays["embeddings"].shape[0]
    if arrays["embeddings"].shape != (n, EMBEDDING_DIM):
        raise ValueError(f"'embeddings' must be shape (N, {EMBEDDING_DIM}), got {arrays['embeddings'].shape}")
    if arrays["embeddings"].dtype != np.float32:
        raise ValueError(f"'embeddings' must be float32, got {arrays['embeddings'].dtype}")

    if arrays["target_vector"].shape != (n, k):
        raise ValueError(f"'target_vector' must be shape (N, {k}), got {arrays['target_vector'].shape}")
    if arrays["target_vector"].dtype != np.uint8:
        raise ValueError(f"'target_vector' must be uint8, got {arrays['target_vector'].dtype}")

    int64_fields = ("window_id", "sound_id", "start", "end")
    for field_name in int64_fields:
        arr = arrays[field_name]
        if arr.shape != (n,):
            raise ValueError(f"{field_name!r} must be shape (N,), got {arr.shape}")
        if arr.dtype != np.int64:
            raise ValueError(f"{field_name!r} must be int64, got {arr.dtype}")

    if arrays["sample_rate"].dtype != np.int32:
        raise ValueError(f"'sample_rate' must be int32, got {arrays['sample_rate'].dtype}")
    if arrays["is_canonical"].dtype != np.uint8:
        raise ValueError(f"'is_canonical' must be uint8, got {arrays['is_canonical'].dtype}")

    string_fields = ("target_codes", "label_state", "sound_filepath", "sound_filename", "dataset", "project")
    for field_name in string_fields:
        arr = arrays[field_name]
        if arr.shape != (n,):
            raise ValueError(f"{field_name!r} must be shape (N,), got {arr.shape}")
        if arr.dtype.kind != "U":
            raise ValueError(f"{field_name!r} must be a fixed-width unicode array, got dtype {arr.dtype}")


def save_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write NPZ to a temp file in the same directory, then os.replace over path.

    Uses an open file handle (not a bare filename) with ``np.savez`` -- passing
    a filename without a ``.npz`` suffix causes numpy to silently append one,
    which would break the intended atomic-rename target.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def assert_npz_matches_rows(npz_path: Path, rows: Sequence[SplitRow], k: int) -> None:
    """Post-write round-trip check: NPZ identity/target fields exactly match ``rows``.

    ``rows`` must be the exact ordered sequence of succeeded ``SplitRow``
    objects used to build the NPZ (``progress.succeeded_rows``) -- this
    asserts both row-for-row order (the NPZ was never silently reordered
    between build and disk) and, keyed by ``window_id``, that every
    identity/target field written to disk matches the source CSV row.
    Raises ``AssertionError`` on any mismatch; never silently repairs data.
    """
    with np.load(npz_path) as npz:
        arrays = {name: npz[name] for name in NPZ_REQUIRED_KEYS}
        n = arrays["embeddings"].shape[0]
        if n != len(rows):
            raise AssertionError(f"{npz_path}: NPZ has {n} rows, expected {len(rows)} succeeded rows")

        npz_window_ids = [int(w) for w in arrays["window_id"]]
        expected_window_ids = [r.window_id for r in rows]
        if npz_window_ids != expected_window_ids:
            raise AssertionError(
                f"{npz_path}: NPZ window_id order does not exactly match succeeded-row order "
                f"(got {npz_window_ids[:5]}..., expected {expected_window_ids[:5]}...)"
            )

        for i, row in enumerate(rows):
            if int(arrays["sound_id"][i]) != row.sound_id:
                raise AssertionError(f"window_id={row.window_id}: NPZ sound_id mismatch")
            if int(arrays["start"][i]) != row.start:
                raise AssertionError(f"window_id={row.window_id}: NPZ start mismatch")
            if int(arrays["end"][i]) != row.end:
                raise AssertionError(f"window_id={row.window_id}: NPZ end mismatch")
            if int(arrays["sample_rate"][i]) != row.sample_rate:
                raise AssertionError(f"window_id={row.window_id}: NPZ sample_rate mismatch")
            expected_is_canonical = 1 if row.is_canonical else 0
            if int(arrays["is_canonical"][i]) != expected_is_canonical:
                raise AssertionError(f"window_id={row.window_id}: NPZ is_canonical mismatch")
            if str(arrays["label_state"][i]) != row.label_state:
                raise AssertionError(f"window_id={row.window_id}: NPZ label_state mismatch")
            if str(arrays["dataset"][i]) != row.dataset:
                raise AssertionError(f"window_id={row.window_id}: NPZ dataset mismatch")
            if str(arrays["project"][i]) != row.project:
                raise AssertionError(f"window_id={row.window_id}: NPZ project mismatch")
            if str(arrays["sound_filename"][i]) != row.sound_filename:
                raise AssertionError(f"window_id={row.window_id}: NPZ sound_filename mismatch")
            if str(arrays["sound_filepath"][i]) != row.sound_filepath:
                raise AssertionError(f"window_id={row.window_id}: NPZ sound_filepath mismatch")

            expected_codes = ";".join(row.target_codes)
            if str(arrays["target_codes"][i]) != expected_codes:
                raise AssertionError(f"window_id={row.window_id}: NPZ target_codes mismatch")

            if len(row.target_vector) != k:
                raise AssertionError(f"window_id={row.window_id}: source target_vector length != {k}")
            npz_vector = [int(v) for v in arrays["target_vector"][i]]
            if npz_vector != list(row.target_vector):
                raise AssertionError(f"window_id={row.window_id}: NPZ target_vector mismatch")


def write_manifest_atomic(path: Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


class OutputDirectoryLock:
    """Exclusive process lock preventing concurrent manifest read-modify-write races."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.handle = None

    def __enter__(self) -> "OutputDirectoryLock":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / ".embedding_extraction.lock"
        self.handle = open(lock_path, "a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()}\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


# --------------------------------------------------------------------------
# Manifest construction
# --------------------------------------------------------------------------


def build_embedding_manifest(
    *,
    created_at_utc: str,
    git_commit: Optional[str],
    git_dirty: bool,
    model_name: str,
    kaggle_slug: str,
    model_local_dir: str,
    model_content_sha256: str,
    embedding_key: str,
    embedding_dim: int,
    weights_license: str,
    weights_license_source_url: str,
    code_license: str,
    code_license_source_url: str,
    extractor_source_sha256: str,
    split_manifest_sha256: str,
    class_list_path: str,
    class_list_sha256: str,
    class_list_n_classes: int,
    split_csv_sha256: Mapping[str, str],
    extraction_params: Mapping[str, Any],
    dependency_versions: Mapping[str, Optional[str]],
    device_info: Mapping[str, Any],
    global_max_failures: int,
    per_split_max_failures: int,
    counts: Mapping[str, Mapping[str, int]],
    padding: Mapping[str, int],
    excluded: Mapping[str, Sequence[Mapping[str, Any]]],
    limited: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    """Build the deterministic (excluding ``created_at_utc``) manifest dict."""
    return {
        "created_at_utc": created_at_utc,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "model": {
            "name": model_name,
            "kaggle_slug": kaggle_slug,
            "local_dir": model_local_dir,
            "embedding_key": embedding_key,
            "embedding_dim": embedding_dim,
            "content_sha256": model_content_sha256,
        },
        "licenses": {
            "weights_license": weights_license,
            "weights_license_source_url": weights_license_source_url,
            "code_license": code_license,
            "code_license_source_url": code_license_source_url,
        },
        "class_list": {
            "path": class_list_path,
            "sha256": class_list_sha256,
            "n_classes": class_list_n_classes,
        },
        "identity_hash_inputs": {
            "extractor_source_sha256": extractor_source_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "class_list_sha256": class_list_sha256,
            "split_csv_sha256": dict(split_csv_sha256),
            "model_local_dir_sha256": model_content_sha256,
            "extraction_params": dict(extraction_params),
        },
        "dependency_versions": dict(dependency_versions),
        "device": dict(device_info),
        "failure_ceiling": {
            "global_max_failures": global_max_failures,
            "per_split_max_failures": per_split_max_failures,
            "note": "Defaults are pre-registered. Any reviewed override must be set "
            "before extraction and recorded verbatim.",
        },
        "counts": {name: dict(c) for name, c in counts.items()},
        "padding": dict(padding),
        "excluded": {name: list(entries) for name, entries in excluded.items()},
        "limited": limited,
        "limit": limit,
    }


def seed_manifest_accumulators(
    existing_manifest: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, List[Dict[str, Any]]]]:
    """Seed ``counts``/``padding``/``excluded`` from a prior manifest on disk.

    Supports item 8's documented recovery path: running with a subset of
    ``--splits`` (e.g. ``--splits val test`` now, ``--splits train`` later)
    must preserve the untouched splits' previously-recorded manifest entries
    -- entries here are only overwritten (in the caller) for splits actually
    re-extracted in the current invocation.
    """
    counts: Dict[str, Dict[str, int]] = {
        name: dict(c) for name, c in existing_manifest.get("counts", {}).items()
    }
    padding: Dict[str, int] = dict(existing_manifest.get("padding", {}))
    excluded: Dict[str, List[Dict[str, Any]]] = {
        name: list(entries) for name, entries in existing_manifest.get("excluded", {}).items()
    }
    return counts, padding, excluded


def existing_split_output_is_valid(
    npz_path: Path,
    manifest_path: Path,
    split_name: str,
    expected_window_ids: Sequence[int],
    expected_identity_hash_inputs: Mapping[str, Any],
    limited: bool,
) -> bool:
    """Check whether an existing NPZ + manifest can satisfy ``--resume``.

    Requires: not limited (limited outputs never satisfy full completion,
    and the current run is never resumed if it is itself limited, to avoid
    ambiguity about which limit produced the cached output); identical
    identity_hash_inputs (split_manifest/class_list/model-dir hashes and
    extraction params); matching requested row count; and an exact
    succeeded-window_id-set match against the CSV's rows minus manifest-
    recorded exclusions -- never a row-count-only check.
    """
    if limited:
        return False
    if not Path(npz_path).is_file() or not Path(manifest_path).is_file():
        return False
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if manifest.get("limited"):
        return False
    manifest_identity = manifest.get("identity_hash_inputs", {})
    manifest_base = {k: v for k, v in manifest_identity.items() if k != "split_csv_sha256"}
    expected_base = {
        k: v for k, v in expected_identity_hash_inputs.items() if k != "split_csv_sha256"
    }
    if manifest_base != expected_base:
        return False
    manifest_split_hashes = manifest_identity.get("split_csv_sha256", {})
    expected_split_hashes = expected_identity_hash_inputs.get("split_csv_sha256", {})
    if manifest_split_hashes.get(split_name) != expected_split_hashes.get(split_name):
        return False

    split_counts = manifest.get("counts", {}).get(split_name)
    if not split_counts or split_counts.get("requested") != len(expected_window_ids):
        return False

    excluded_ids = {
        int(entry["window_id"]) for entry in manifest.get("excluded", {}).get(split_name, [])
    }
    expected_succeeded_ids = set(int(w) for w in expected_window_ids) - excluded_ids

    try:
        with np.load(npz_path) as npz:
            if "window_id" not in npz:
                return False
            existing_ids = set(int(w) for w in npz["window_id"])
    except Exception:
        return False

    return existing_ids == expected_succeeded_ids


# --------------------------------------------------------------------------
# Extraction orchestration
# --------------------------------------------------------------------------


def extract_split(
    model,
    emb_key: str,
    rows: Sequence[SplitRow],
    split_name: str,
    batch_size: int,
    max_failures: int,
    max_failures_per_split: int,
    global_failures: List[FailureRecord],
    progress: SplitProgress,
) -> None:
    """Stream one sound file at a time through decode, slicing, and inference.

    Each source recording is decoded/resampled once. Its windows are sliced
    and inferred in bounded batches before the full waveform is released,
    avoiding the ~60 GB waveform cache that would result from decoding every
    train window before inference. Successful rows are reordered to the
    source CSV order before output.
    """
    n = len(rows)
    if n and [r.original_index for r in rows] != list(range(rows[0].original_index, rows[0].original_index + n)):
        raise AssertionError(
            "extract_split requires rows sorted by contiguous original_index "
            "(validate_split_rows' contract); order-preservation guarantees do not hold otherwise"
        )
    groups = group_rows_by_sound_id(rows)
    n_processed = 0
    start_time = time.monotonic()
    pending_rows: List[SplitRow] = []
    pending_audio: List[np.ndarray] = []

    def flush_pending_batch() -> None:
        if not pending_rows:
            return
        batch_rows = list(pending_rows)
        batch_array = np.stack(pending_audio, axis=0)
        pending_rows.clear()
        pending_audio.clear()
        try:
            embeddings = run_model_batch(model, emb_key, batch_array)
        except Exception as exc:  # noqa: BLE001 - recorded, never zero-filled
            for row in batch_rows:
                fr = FailureRecord(
                    split_name,
                    row.window_id,
                    row.sound_id,
                    "model_inference_error",
                    str(exc),
                )
                progress.split_failures.append(fr)
                global_failures.append(fr)
            check_failure_ceilings(
                len(global_failures),
                len(progress.split_failures),
                max_failures,
                max_failures_per_split,
            )
            return
        progress.succeeded_rows.extend(batch_rows)
        progress.succeeded_embeddings.append(embeddings)

    for indices in groups.values():
        first_row = rows[indices[0]]
        try:
            y_full = load_full_audio(first_row.sound_filepath)
        except Exception as exc:  # noqa: BLE001 - any failure is recorded, never zero-filled
            for i in indices:
                row = rows[i]
                fr = FailureRecord(split_name, row.window_id, row.sound_id, "audio_load_error", str(exc))
                progress.split_failures.append(fr)
                global_failures.append(fr)
                check_failure_ceilings(
                    len(global_failures), len(progress.split_failures), max_failures, max_failures_per_split
                )
                n_processed += 1
                report_progress(split_name, n_processed, n, time.monotonic() - start_time)
            continue

        for i in indices:
            row = rows[i]
            try:
                audio, n_padded = slice_window_from_full_audio(
                    y_full, row.start, row.end, row.sample_rate
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never zero-filled
                fr = FailureRecord(
                    split_name, row.window_id, row.sound_id, "audio_load_error", str(exc)
                )
                progress.split_failures.append(fr)
                global_failures.append(fr)
                check_failure_ceilings(
                    len(global_failures),
                    len(progress.split_failures),
                    max_failures,
                    max_failures_per_split,
                )
                n_processed += 1
                report_progress(split_name, n_processed, n, time.monotonic() - start_time)
                continue
            progress.n_padded += n_padded
            pending_rows.append(row)
            pending_audio.append(audio)
            n_processed += 1
            report_progress(split_name, n_processed, n, time.monotonic() - start_time)
            if len(pending_rows) == batch_size:
                flush_pending_batch()

        del y_full

    flush_pending_batch()

    if progress.succeeded_rows:
        flat_embeddings = np.concatenate(progress.succeeded_embeddings, axis=0)
        order = np.argsort(
            np.asarray([row.original_index for row in progress.succeeded_rows]),
            kind="stable",
        )
        progress.succeeded_rows = [progress.succeeded_rows[int(i)] for i in order]
        progress.succeeded_embeddings = [flat_embeddings[order]]


def process_and_save_split(
    model,
    emb_key: str,
    rows: Sequence[SplitRow],
    split_name: str,
    batch_size: int,
    max_failures: int,
    max_failures_per_split: int,
    global_failures: List[FailureRecord],
    progress: SplitProgress,
    out_path: Path,
    k: int,
) -> None:
    """Run :func:`extract_split`, then build/validate/save its NPZ, then round-trip-assert it.

    If ``extract_split`` raises ``FailureCeilingExceeded`` it propagates
    unchanged and ``out_path`` is never written -- ``progress`` still holds
    an accurate partial record for the caller to fold into the manifest
    (item 3: a ceiling-exceeded abort must never leave behind a final NPZ).
    On success, the NPZ is written atomically and then immediately read back
    and asserted to exactly match ``progress.succeeded_rows`` (item 2).
    """
    extract_split(
        model,
        emb_key,
        rows,
        split_name,
        batch_size,
        max_failures,
        max_failures_per_split,
        global_failures,
        progress,
    )
    embeddings = (
        np.concatenate(progress.succeeded_embeddings, axis=0)
        if progress.succeeded_embeddings
        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    )
    arrays = build_npz_arrays(progress.succeeded_rows, embeddings, k)
    validate_npz_schema(arrays, k)
    save_npz_atomic(out_path, arrays)
    assert_npz_matches_rows(out_path, progress.succeeded_rows, k)


def _run_extraction_locked(args: argparse.Namespace) -> None:
    """Validate everything up-front, then load the model and extract.

    Ordering is deliberate: every requested split's CSV rows are fully
    validated (:func:`validate_split_rows`) and every unique audio file they
    reference is preflight-checked (:func:`preflight_validate_audio_files`)
    *before* the GPU is acquired or the (large, slow-to-load) SavedModel is
    read -- so a data problem is caught in seconds, not partway through a
    long extraction run.

    Known operational limitation (documented, not a defect): there is no
    intra-split (row-level) checkpoint/shard-resume. The supported recovery
    path for an interrupted or partially-failed run is (a) per-split atomic
    NPZ + manifest writes (a completed split's output is never touched again
    unless re-requested) and (b) invoking this script with a subset of
    ``--splits`` (e.g. ``--splits val test`` now, ``--splits train`` later)
    -- the manifest's ``counts``/``padding``/``excluded`` sections are seeded
    from the existing manifest on disk and only overwritten for the splits
    actually requested in the current invocation, so unrelated splits'
    recorded results are preserved across separate invocations.
    """
    repo_dir = Path(__file__).resolve().parent
    model_dir = Path(args.model_dir)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    audio_root = Path(args.audio_root)
    class_list_path = Path(args.class_list) if args.class_list else split_dir / "class_list.json"
    split_manifest_path = (
        Path(args.split_manifest) if args.split_manifest else split_dir / "split_manifest.json"
    )

    class_codes = load_class_list(class_list_path)
    k = len(class_codes)
    print(f"Loaded class_list.json with {k} classes from {class_list_path}")

    split_csv_paths = {name: split_dir / f"{name}_split.csv" for name in SPLIT_NAMES}

    # --- Preflight (item 1): validate every requested split's CSV rows and every
    # unique audio file they reference, before any GPU/model work begins. ---
    print(f"\n{'=' * 60}\nPreflight validation\n{'=' * 60}")
    all_rows: Dict[str, List[SplitRow]] = {}
    for split_name in args.splits:
        csv_path = split_csv_paths[split_name]
        if not csv_path.is_file():
            print(f"  Skipping {split_name}: {csv_path} not found")
            continue
        print(f"  Validating {split_name} split CSV rows ({csv_path})...")
        rows = validate_split_rows(csv_path, split_name, class_codes, audio_root)
        if args.limit is not None:
            rows = rows[: args.limit]
            print(f"    --limit {args.limit}: restricting {split_name} to {len(rows)} rows (smoke-test only)")
        all_rows[split_name] = rows
        print(f"    {split_name}: {len(rows)} rows validated OK")

    unique_audio_paths = sorted({r.sound_filepath for rows in all_rows.values() for r in rows})
    print(f"  Validating {len(unique_audio_paths)} unique audio file(s) (header-only, no full decode)...")
    preflight_validate_audio_files(unique_audio_paths)
    print("  Preflight OK: all CSV rows and all unique audio files validated.")

    require_gpu_or_exit()

    model = load_perch_model(model_dir)
    emb_key = inspect_model_outputs(model)

    print("Hashing model directory (one-time, may take a moment for a large SavedModel)...")
    model_content_sha256 = hash_model_dir(model_dir)
    class_list_sha256 = sha256_file(class_list_path)
    split_manifest_sha256 = sha256_file(split_manifest_path)
    extractor_source_sha256 = sha256_file(Path(__file__).resolve())
    git_commit, git_dirty = get_git_commit(repo_dir)
    dependency_versions = get_dependency_versions()

    import tensorflow as tf

    device_info = {
        "tensorflow_version": tf.__version__,
        "gpu_devices": [g.name for g in tf.config.list_physical_devices("GPU")],
    }

    extraction_params = {
        "target_sample_rate": PERCH_SR,
        "window_sec": WINDOW_SEC,
        "target_peak": TARGET_PEAK,
        "batch_size": args.batch_size,
        "rounding_tolerance_samples": ROUNDING_TOLERANCE_SAMPLES,
    }

    split_csv_sha256 = {
        name: sha256_file(path) for name, path in split_csv_paths.items() if path.is_file()
    }

    identity_hash_inputs = {
        "extractor_source_sha256": extractor_source_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "class_list_sha256": class_list_sha256,
        "split_csv_sha256": split_csv_sha256,
        "model_local_dir_sha256": model_content_sha256,
        "extraction_params": extraction_params,
    }

    limited = args.limit is not None
    manifest_path = output_dir / "embedding_manifest.json"
    existing_manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r") as f:
                existing_manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            existing_manifest = {}

    counts, padding, excluded = seed_manifest_accumulators(existing_manifest)

    global_failures: List[FailureRecord] = []

    for split_name, rows in all_rows.items():
        print(f"\n{'=' * 60}\nSplit: {split_name}\n{'=' * 60}")

        out_path = output_dir / f"{split_name}_emb.npz"
        expected_window_ids = [r.window_id for r in rows]

        if args.resume and existing_split_output_is_valid(
            out_path, manifest_path, split_name, expected_window_ids, identity_hash_inputs, limited
        ):
            print(f"  [{split_name}] existing {out_path} fully validated -- resuming (skipping re-extraction)")
            continue

        # A failed re-extraction must not leave a stale artifact from a prior
        # successful run that no longer matches the newly-written manifest.
        out_path.unlink(missing_ok=True)
        progress = SplitProgress()
        try:
            process_and_save_split(
                model,
                emb_key,
                rows,
                split_name,
                args.batch_size,
                args.max_failures,
                args.max_failures_per_split,
                global_failures,
                progress,
                out_path,
                k,
            )
        except FailureCeilingExceeded as exc:
            counts[split_name] = {
                "requested": len(rows),
                "succeeded": len(progress.succeeded_rows),
                "excluded": len(progress.split_failures),
            }
            padding[split_name] = progress.n_padded
            excluded[split_name] = [fr.to_dict() for fr in progress.split_failures]
            manifest = build_embedding_manifest(
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                git_commit=git_commit,
                git_dirty=git_dirty,
                model_name="perch_v2",
                kaggle_slug=args.kaggle_slug,
                model_local_dir=str(model_dir),
                model_content_sha256=model_content_sha256,
                embedding_key=emb_key,
                embedding_dim=EMBEDDING_DIM,
                weights_license=args.weights_license,
                weights_license_source_url=args.weights_license_source_url,
                code_license=args.code_license,
                code_license_source_url=args.code_license_source_url,
                extractor_source_sha256=extractor_source_sha256,
                split_manifest_sha256=split_manifest_sha256,
                class_list_path=str(class_list_path),
                class_list_sha256=class_list_sha256,
                class_list_n_classes=k,
                split_csv_sha256=split_csv_sha256,
                extraction_params=extraction_params,
                dependency_versions=dependency_versions,
                device_info=device_info,
                global_max_failures=args.max_failures,
                per_split_max_failures=args.max_failures_per_split,
                counts=counts,
                padding=padding,
                excluded=excluded,
                limited=limited,
                limit=args.limit,
            )
            write_manifest_atomic(manifest_path, manifest)
            raise SystemExit(f"Aborting extraction: {exc}") from exc

        print(f"  [{split_name}] saved {len(progress.succeeded_rows):,} embeddings to {out_path}")

        counts[split_name] = {
            "requested": len(rows),
            "succeeded": len(progress.succeeded_rows),
            "excluded": len(progress.split_failures),
        }
        padding[split_name] = progress.n_padded
        excluded[split_name] = [fr.to_dict() for fr in progress.split_failures]

        manifest = build_embedding_manifest(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit,
            git_dirty=git_dirty,
            model_name="perch_v2",
            kaggle_slug=args.kaggle_slug,
            model_local_dir=str(model_dir),
            model_content_sha256=model_content_sha256,
            embedding_key=emb_key,
            embedding_dim=EMBEDDING_DIM,
            weights_license=args.weights_license,
            weights_license_source_url=args.weights_license_source_url,
            code_license=args.code_license,
            code_license_source_url=args.code_license_source_url,
            extractor_source_sha256=extractor_source_sha256,
            split_manifest_sha256=split_manifest_sha256,
            class_list_path=str(class_list_path),
            class_list_sha256=class_list_sha256,
            class_list_n_classes=k,
            split_csv_sha256=split_csv_sha256,
            extraction_params=extraction_params,
            dependency_versions=dependency_versions,
            device_info=device_info,
            global_max_failures=args.max_failures,
            per_split_max_failures=args.max_failures_per_split,
            counts=counts,
            padding=padding,
            excluded=excluded,
            limited=limited,
            limit=args.limit,
        )
        # Written after every split so an interrupted multi-split run retains progress.
        write_manifest_atomic(manifest_path, manifest)

    print(f"\nDone. Manifest: {manifest_path}")


def run_extraction(args: argparse.Namespace) -> None:
    """Serialize extraction within one output directory."""
    with OutputDirectoryLock(Path(args.output_dir)):
        _run_extraction_locked(args)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--download-v2", action="store_true", help="Download Perch v2 from Kaggle into --model-dir and exit"
    )
    mode.add_argument(
        "--extract", action="store_true", help="Extract train/val/test embeddings from the split CSVs"
    )

    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-dir", type=str, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audio-root", type=str, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--class-list", type=str, default=None, help="Default: {split-dir}/class_list.json")
    parser.add_argument(
        "--split-manifest", type=str, default=None, help="Default: {split-dir}/split_manifest.json"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--splits", nargs="+", default=list(SPLIT_NAMES), choices=list(SPLIT_NAMES), help="Splits to extract"
    )
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    parser.add_argument("--max-failures-per-split", type=int, default=DEFAULT_MAX_FAILURES_PER_SPLIT)
    parser.add_argument("--kaggle-slug", type=str, default=KAGGLE_SLUG)
    parser.add_argument(
        "--weights-license",
        type=str,
        default=DEFAULT_WEIGHTS_LICENSE,
        help="Perch v2 weights license string, verified against --weights-license-source-url "
        "(never invented; override only with a freshly reviewed value)",
    )
    parser.add_argument(
        "--weights-license-source-url",
        type=str,
        default=DEFAULT_WEIGHTS_LICENSE_SOURCE_URL,
        help="Source URL the weights license was verified against (recorded verbatim in the "
        "manifest; this is the upstream model-card mirror, not a Kaggle-specific license claim)",
    )
    parser.add_argument(
        "--code-license",
        type=str,
        default=DEFAULT_CODE_LICENSE,
        help="Source-code license string, verified against --code-license-source-url "
        "(never invented; override only with a freshly reviewed value)",
    )
    parser.add_argument(
        "--code-license-source-url",
        type=str,
        default=DEFAULT_CODE_LICENSE_SOURCE_URL,
        help="Source URL the code license was verified against (recorded verbatim in the manifest)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Restrict each split to the first N rows (smoke/test only; marks the manifest limited=true)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a split only if its existing NPZ+manifest fully validate against current inputs",
    )
    return parser


def main() -> None:
    _ensure_cuda_libs_in_ldlibpath()

    parser = build_parser()
    args = parser.parse_args()

    validate_license_metadata(
        args.weights_license,
        args.weights_license_source_url,
        args.code_license,
        args.code_license_source_url,
    )

    if args.download_v2:
        download_perch_v2(Path(args.model_dir), kaggle_slug=args.kaggle_slug)
        return

    run_extraction(args)


if __name__ == "__main__":
    main()
