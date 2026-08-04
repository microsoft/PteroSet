"""Tests for extract_perch_embeddings.py using synthetic data only.

Covers: CSV/class-list schema validation, no_bird/species_window target
consistency, audio preprocessing (monkeypatched librosa + a real temp WAV),
exact offset/sample conversion, failure-ceiling behavior (no zero fallback),
NPZ schema validation, atomic save, deterministic hashes/model-directory
hash, manifest determinism and counts, embedding-key selection with fake
signature objects, output-shape rejection, and CLI parsing/help.

TensorFlow, kagglehub, GPU hardware, and the Perch SavedModel are never
required to run this file -- ``extract_perch_embeddings`` only imports those
inside functions that these tests do not call.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import extract_perch_embeddings as epe


SR = 48000
WINDOW_SPAN = 5 * SR  # 240,000 samples at 48 kHz, matches the dataset's 5 s windows


def make_class_codes(n=4):
    return [f"CODE{i}" for i in range(n)]


def write_class_list(path: Path, codes):
    entries = [{"index": i, "code": c, "species": f"Species {i}"} for i, c in enumerate(codes)]
    path.write_text(json.dumps(entries))


def make_csv_row(
    window_id,
    sound_id=1,
    start=0,
    end=WINDOW_SPAN,
    sample_rate=SR,
    dataset="MAP1",
    project="MAP1",
    is_canonical="1",
    label_state="no_bird",
    sound_filename="a.wav",
    target_codes="",
    target_vector=None,
    k=4,
):
    if target_vector is None:
        target_vector = [0] * k
    return {
        "window_id": str(window_id),
        "dataset": dataset,
        "sample_rate": str(sample_rate),
        "sound_id": str(sound_id),
        "start": str(start),
        "end": str(end),
        "project": project,
        "is_canonical": is_canonical,
        "label_state": label_state,
        "spec_name": f"spec_{window_id}.npy",
        "sound_filename": sound_filename,
        "target_codes": target_codes,
        "target_vector": json.dumps(target_vector),
    }


def write_split_csv(path: Path, rows):
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epe.REQUIRED_SPLIT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------
# Class-list loading
# --------------------------------------------------------------------------


def test_load_class_list_orders_by_index(tmp_path):
    path = tmp_path / "class_list.json"
    entries = [
        {"index": 2, "code": "CCC", "species": "C"},
        {"index": 0, "code": "AAA", "species": "A"},
        {"index": 1, "code": "BBB", "species": "B"},
    ]
    path.write_text(json.dumps(entries))
    codes = epe.load_class_list(path)
    assert codes == ["AAA", "BBB", "CCC"]


def test_load_class_list_rejects_non_contiguous_index(tmp_path):
    path = tmp_path / "class_list.json"
    entries = [{"index": 0, "code": "AAA", "species": "A"}, {"index": 2, "code": "BBB", "species": "B"}]
    path.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="index"):
        epe.load_class_list(path)


def test_load_class_list_rejects_empty(tmp_path):
    path = tmp_path / "class_list.json"
    path.write_text(json.dumps([]))
    with pytest.raises(ValueError):
        epe.load_class_list(path)


# --------------------------------------------------------------------------
# Strict integer / boolean parsing
# --------------------------------------------------------------------------


def test_parse_strict_int_accepts_clean_integers():
    assert epe.parse_strict_int("123", "window_id") == 123
    assert epe.parse_strict_int(" 45 ", "window_id") == 45


@pytest.mark.parametrize("bad", ["3.0", "abc", "", "1e3", "12,3"])
def test_parse_strict_int_rejects_non_integers(bad):
    with pytest.raises(ValueError):
        epe.parse_strict_int(bad, "window_id")


def test_parse_bool01():
    assert epe.parse_bool01("1", "is_canonical") is True
    assert epe.parse_bool01("0", "is_canonical") is False
    with pytest.raises(ValueError):
        epe.parse_bool01("2", "is_canonical")


# --------------------------------------------------------------------------
# target_codes / target_vector parsing
# --------------------------------------------------------------------------


def test_parse_target_codes_empty_and_multi():
    assert epe.parse_target_codes("") == ()
    assert epe.parse_target_codes("A;B") == ("A", "B")


def test_parse_target_vector_valid():
    assert epe.parse_target_vector("[0, 1, 0]", 3) == (0, 1, 0)


def test_parse_target_vector_rejects_wrong_length():
    with pytest.raises(ValueError):
        epe.parse_target_vector("[0, 1]", 3)


def test_parse_target_vector_rejects_non_binary():
    with pytest.raises(ValueError):
        epe.parse_target_vector("[0, 2, 0]", 3)


# --------------------------------------------------------------------------
# Full split-CSV validation
# --------------------------------------------------------------------------


@pytest.fixture
def audio_root(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    (root / "a.wav").write_bytes(b"fake-wav-bytes")
    return root


def test_validate_split_rows_happy_path(tmp_path, audio_root):
    codes = make_class_codes(4)
    class_list_path = tmp_path / "class_list.json"
    write_class_list(class_list_path, codes)

    rows = [
        make_csv_row(1, sound_id=10, label_state="no_bird", target_vector=[0, 0, 0, 0]),
        make_csv_row(
            2,
            sound_id=10,
            start=WINDOW_SPAN,
            end=2 * WINDOW_SPAN,
            label_state="species_window",
            target_codes="CODE1",
            target_vector=[0, 1, 0, 0],
        ),
    ]
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)

    parsed = epe.validate_split_rows(csv_path, "train", codes, audio_root)
    assert len(parsed) == 2
    assert parsed[0].label_state == "no_bird"
    assert parsed[1].target_codes == ("CODE1",)
    assert str(audio_root / "a.wav") == parsed[0].sound_filepath


def test_validate_split_rows_rejects_duplicate_window_id(tmp_path, audio_root):
    codes = make_class_codes(4)
    rows = [make_csv_row(1), make_csv_row(1)]
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)
    with pytest.raises(ValueError, match="duplicate window_id"):
        epe.validate_split_rows(csv_path, "train", codes, audio_root)


def test_validate_split_rows_rejects_wrong_duration(tmp_path, audio_root):
    codes = make_class_codes(4)
    rows = [make_csv_row(1, end=WINDOW_SPAN - 1)]
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)
    with pytest.raises(ValueError, match="duration"):
        epe.validate_split_rows(csv_path, "train", codes, audio_root)


def test_validate_split_rows_rejects_bad_label_state(tmp_path, audio_root):
    codes = make_class_codes(4)
    rows = [make_csv_row(1, label_state="something_else")]
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)
    with pytest.raises(ValueError, match="label_state"):
        epe.validate_split_rows(csv_path, "train", codes, audio_root)


def test_validate_split_rows_rejects_no_bird_with_nonzero_vector(tmp_path, audio_root):
    codes = make_class_codes(4)
    rows = [make_csv_row(1, label_state="no_bird", target_vector=[1, 0, 0, 0])]
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)
    with pytest.raises(ValueError, match="nonzero"):
        epe.validate_split_rows(csv_path, "train", codes, audio_root)


def test_validate_split_rows_rejects_species_window_all_zero(tmp_path, audio_root):
    codes = make_class_codes(4)
    rows = [make_csv_row(1, label_state="species_window", target_vector=[0, 0, 0, 0])]
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)
    with pytest.raises(ValueError, match="all-zero"):
        epe.validate_split_rows(csv_path, "train", codes, audio_root)


def test_validate_split_rows_rejects_target_codes_vector_mismatch(tmp_path, audio_root):
    codes = make_class_codes(4)
    rows = [
        make_csv_row(
            1, label_state="species_window", target_codes="CODE0", target_vector=[0, 1, 0, 0]
        )
    ]
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)
    with pytest.raises(ValueError, match="do not match"):
        epe.validate_split_rows(csv_path, "train", codes, audio_root)


def test_validate_split_rows_rejects_unknown_target_code(tmp_path, audio_root):
    codes = make_class_codes(4)
    rows = [
        make_csv_row(
            1, label_state="species_window", target_codes="NOTREAL", target_vector=[1, 0, 0, 0]
        )
    ]
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)
    with pytest.raises(ValueError, match="unknown class code"):
        epe.validate_split_rows(csv_path, "train", codes, audio_root)


def test_validate_split_rows_rejects_missing_audio_file(tmp_path):
    codes = make_class_codes(4)
    missing_root = tmp_path / "no_audio_here"
    missing_root.mkdir()
    rows = [make_csv_row(1)]
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)
    with pytest.raises(ValueError, match="sound file not found"):
        epe.validate_split_rows(csv_path, "train", codes, missing_root)


@pytest.mark.parametrize("sound_filename", ["/tmp/outside.wav", "../outside.wav", "subdir/a.wav"])
def test_validate_split_rows_rejects_audio_path_escape(tmp_path, sound_filename):
    codes = make_class_codes(4)
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    rows = [make_csv_row(1, sound_filename=sound_filename)]
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)
    with pytest.raises(ValueError, match="bare filename"):
        epe.validate_split_rows(csv_path, "train", codes, audio_root)


def test_validate_split_rows_rejects_float_looking_window_id(tmp_path, audio_root):
    codes = make_class_codes(4)
    rows = [make_csv_row(1)]
    rows[0]["window_id"] = "1.0"
    csv_path = tmp_path / "train_split.csv"
    write_split_csv(csv_path, rows)
    with pytest.raises(ValueError):
        epe.validate_split_rows(csv_path, "train", codes, audio_root)


def test_validate_header_rejects_missing_or_extra_columns(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        epe.validate_header(["window_id", "dataset"], tmp_path / "x.csv")
    with pytest.raises(ValueError, match="extra"):
        epe.validate_header(list(epe.REQUIRED_SPLIT_COLUMNS) + ["extra_col"], tmp_path / "x.csv")


def test_validate_header_rejects_empty_csv(tmp_path):
    with pytest.raises(ValueError, match="empty CSV"):
        epe.validate_header(None, tmp_path / "x.csv")


# --------------------------------------------------------------------------
# Audio preprocessing (pure numpy)
# --------------------------------------------------------------------------


def test_preprocess_waveform_exact_length_normalizes_peak():
    y = np.zeros(epe.WINDOW_SAMPLES, dtype=np.float32)
    y[100] = 2.0
    out, n_padded = epe.preprocess_waveform(y)
    assert n_padded == 0
    assert out.shape == (epe.WINDOW_SAMPLES,)
    assert out.dtype == np.float32
    assert np.isclose(np.abs(out).max(), epe.TARGET_PEAK, atol=1e-6)
    assert np.all(np.isfinite(out))


def test_preprocess_waveform_silence_is_valid_not_a_failure():
    y = np.zeros(epe.WINDOW_SAMPLES, dtype=np.float32)
    out, n_padded = epe.preprocess_waveform(y)
    assert n_padded == 0
    assert np.all(out == 0.0)
    assert out.dtype == np.float32


def test_preprocess_waveform_pads_within_tolerance():
    short = epe.WINDOW_SAMPLES - 3
    y = np.ones(short, dtype=np.float32) * 0.5
    out, n_padded = epe.preprocess_waveform(y)
    assert n_padded == 3
    assert out.shape == (epe.WINDOW_SAMPLES,)
    assert np.all(out[short:] == 0.0)


def test_preprocess_waveform_rejects_meaningful_truncation():
    too_short = epe.WINDOW_SAMPLES - epe.ROUNDING_TOLERANCE_SAMPLES - 1
    y = np.ones(too_short, dtype=np.float32)
    with pytest.raises(ValueError, match="meaningful truncation"):
        epe.preprocess_waveform(y)


def test_preprocess_waveform_rejects_meaningful_excess():
    too_long = epe.WINDOW_SAMPLES + epe.ROUNDING_TOLERANCE_SAMPLES + 1
    y = np.ones(too_long, dtype=np.float32)
    with pytest.raises(ValueError, match="too long"):
        epe.preprocess_waveform(y)


def test_preprocess_waveform_rejects_non_finite():
    y = np.ones(epe.WINDOW_SAMPLES, dtype=np.float32)
    y[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        epe.preprocess_waveform(y)


def test_compute_offset_duration_sec_exact_conversion():
    offset, duration = epe.compute_offset_duration_sec(start=240000, end=480000, sample_rate=48000)
    assert offset == pytest.approx(5.0)
    assert duration == pytest.approx(5.0)


def test_load_audio_segment_calls_librosa_with_exact_offset_duration(monkeypatch):
    captured = {}

    class FakeLibrosa:
        @staticmethod
        def load(filepath, sr, offset, duration, mono):
            captured["filepath"] = filepath
            captured["sr"] = sr
            captured["offset"] = offset
            captured["duration"] = duration
            captured["mono"] = mono
            return np.zeros(int(sr * duration), dtype=np.float32), sr

    monkeypatch.setitem(sys.modules, "librosa", FakeLibrosa())

    epe.load_audio_segment("some/file.wav", start=96000, end=336000, sample_rate=48000)

    assert captured["filepath"] == "some/file.wav"
    assert captured["sr"] == epe.PERCH_SR
    assert captured["offset"] == pytest.approx(2.0)
    assert captured["duration"] == pytest.approx(5.0)
    assert captured["mono"] is True


def test_load_audio_segment_with_real_librosa_and_temp_wav(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    sr = 48000
    duration_s = 6.0
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path / "tone.wav"
    soundfile.write(wav_path, tone, sr)

    y, n_padded = epe.load_audio_segment(str(wav_path), start=0, end=5 * sr, sample_rate=sr)
    assert y.shape == (epe.WINDOW_SAMPLES,)
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))
    assert n_padded >= 0


def test_full_audio_slice_matches_direct_window_load_within_tolerance(tmp_path):
    """The optimized production path must remain numerically close to Orcas' direct load."""
    soundfile = pytest.importorskip("soundfile")
    sr = 48000
    duration_s = 12.0
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    waveform = (
        0.08 * np.sin(2 * np.pi * 440 * t)
        + 0.03 * np.sin(2 * np.pi * 1733 * t)
    ).astype(np.float32)
    wav_path = tmp_path / "two_tones.wav"
    soundfile.write(wav_path, waveform, sr)

    start, end = 2 * sr, 7 * sr
    direct, _ = epe.load_audio_segment(str(wav_path), start, end, sr)
    full = epe.load_full_audio(str(wav_path))
    sliced, _ = epe.slice_window_from_full_audio(full, start, end, sr)

    assert np.max(np.abs(direct - sliced)) < 0.01
    assert np.corrcoef(direct, sliced)[0, 1] > 0.99999


# --------------------------------------------------------------------------
# Embedding-key selection (fake signature objects, no TensorFlow)
# --------------------------------------------------------------------------


class FakeShape:
    def __init__(self, dims):
        self._dims = dims
        self.rank = len(dims)

    def __getitem__(self, idx):
        return self._dims[idx]


class FakeOutputSpec:
    def __init__(self, dims, dtype="float32"):
        self.shape = FakeShape(dims)
        self.dtype = dtype


def test_select_embedding_key_picks_the_1536_output():
    outputs = {
        "output_0": FakeOutputSpec([None, 10932]),
        "output_1": FakeOutputSpec([None, 1536]),
    }
    assert epe.select_embedding_key(outputs) == "output_1"


def test_select_embedding_key_raises_if_none_match():
    outputs = {"output_0": FakeOutputSpec([None, 10932]), "output_1": FakeOutputSpec([None, 1280])}
    with pytest.raises(ValueError, match="No serving_default output"):
        epe.select_embedding_key(outputs)


def test_select_embedding_key_raises_if_ambiguous():
    outputs = {"output_0": FakeOutputSpec([None, 1536]), "output_1": FakeOutputSpec([None, 1536])}
    with pytest.raises(ValueError, match="Multiple serving_default outputs"):
        epe.select_embedding_key(outputs)


def test_select_embedding_key_ignores_non_rank2_outputs():
    outputs = {
        "output_0": FakeOutputSpec([None, 5, 1536]),  # rank 3, must be ignored
        "output_1": FakeOutputSpec([None, 1536]),
    }
    assert epe.select_embedding_key(outputs) == "output_1"


# --------------------------------------------------------------------------
# Output-shape validation / rejection
# --------------------------------------------------------------------------


def test_validate_embedding_batch_shape_accepts_valid():
    emb = np.zeros((8, epe.EMBEDDING_DIM), dtype=np.float32)
    epe.validate_embedding_batch_shape(emb, expected_batch_size=8)


def test_validate_embedding_batch_shape_rejects_wrong_dim():
    emb = np.zeros((8, 1280), dtype=np.float32)
    with pytest.raises(ValueError, match="embedding dim"):
        epe.validate_embedding_batch_shape(emb, expected_batch_size=8)


def test_validate_embedding_batch_shape_rejects_wrong_batch_size():
    emb = np.zeros((7, epe.EMBEDDING_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="batch size"):
        epe.validate_embedding_batch_shape(emb, expected_batch_size=8)


def test_validate_embedding_batch_shape_rejects_non_finite():
    emb = np.zeros((8, epe.EMBEDDING_DIM), dtype=np.float32)
    emb[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        epe.validate_embedding_batch_shape(emb, expected_batch_size=8)


def test_validate_embedding_batch_shape_rejects_rank1():
    emb = np.zeros((epe.EMBEDDING_DIM,), dtype=np.float32)
    with pytest.raises(ValueError, match="rank-2"):
        epe.validate_embedding_batch_shape(emb, expected_batch_size=1)


# --------------------------------------------------------------------------
# Failure-ceiling enforcement (no zero fallback)
# --------------------------------------------------------------------------


def test_check_failure_ceilings_passes_within_bounds():
    epe.check_failure_ceilings(global_failures=0, split_failures=0, max_failures=0, max_failures_per_split=0)


def test_check_failure_ceilings_raises_on_global_ceiling():
    with pytest.raises(epe.FailureCeilingExceeded, match="global failure ceiling"):
        epe.check_failure_ceilings(global_failures=1, split_failures=0, max_failures=0, max_failures_per_split=0)


def test_check_failure_ceilings_raises_on_per_split_ceiling():
    with pytest.raises(epe.FailureCeilingExceeded, match="per-split failure ceiling"):
        epe.check_failure_ceilings(global_failures=0, split_failures=1, max_failures=5, max_failures_per_split=0)


def test_check_failure_ceilings_allows_nonzero_override():
    epe.check_failure_ceilings(global_failures=2, split_failures=2, max_failures=5, max_failures_per_split=5)


def test_extract_split_aborts_on_first_audio_failure_default_ceiling(tmp_path):
    codes = make_class_codes(2)
    rows = [
        epe.SplitRow(
            window_id=1,
            dataset="MAP1",
            sample_rate=SR,
            sound_id=1,
            start=0,
            end=WINDOW_SPAN,
            project="MAP1",
            is_canonical=True,
            label_state="no_bird",
            spec_name="s1.npy",
            sound_filename="missing.wav",
            sound_filepath=str(tmp_path / "missing.wav"),
            target_codes=(),
            target_vector=(0, 0),
            original_index=0,
        )
    ]
    global_failures = []
    progress = epe.SplitProgress()
    with pytest.raises(epe.FailureCeilingExceeded):
        epe.extract_split(
            model=None,
            emb_key="output_1",
            rows=rows,
            split_name="train",
            batch_size=64,
            max_failures=0,
            max_failures_per_split=0,
            global_failures=global_failures,
            progress=progress,
        )
    # No embedding was ever fabricated for the failed row.
    assert progress.succeeded_rows == []
    assert progress.succeeded_embeddings == []
    assert len(progress.split_failures) == 1
    assert progress.split_failures[0].window_id == 1
    assert progress.split_failures[0].reason == "audio_load_error"


def test_extract_split_with_override_excludes_failed_row_keeps_alignment(tmp_path, monkeypatch):
    codes = make_class_codes(2)

    good_wav = tmp_path / "good.wav"
    good_wav.write_bytes(b"placeholder")

    # Distinct sound_id per row (distinct source files) so the sound_id-grouped
    # decode phase does not merge these two windows into a single (failing) group.
    rows = [
        epe.SplitRow(
            window_id=1,
            dataset="MAP1",
            sample_rate=SR,
            sound_id=1,
            start=0,
            end=WINDOW_SPAN,
            project="MAP1",
            is_canonical=True,
            label_state="no_bird",
            spec_name="s1.npy",
            sound_filename="missing.wav",
            sound_filepath=str(tmp_path / "missing.wav"),
            target_codes=(),
            target_vector=(0, 0),
            original_index=0,
        ),
        epe.SplitRow(
            window_id=2,
            dataset="MAP1",
            sample_rate=SR,
            sound_id=2,
            start=0,
            end=WINDOW_SPAN,
            project="MAP1",
            is_canonical=True,
            label_state="no_bird",
            spec_name="s2.npy",
            sound_filename="good.wav",
            sound_filepath=str(good_wav),
            target_codes=(),
            target_vector=(0, 0),
            original_index=1,
        ),
    ]

    def fake_load_full_audio(filepath, target_sr=epe.PERCH_SR):
        if "missing" in filepath:
            raise FileNotFoundError(filepath)
        return np.zeros(epe.WINDOW_SAMPLES, dtype=np.float32)

    def fake_run_model_batch(model, emb_key, batch):
        return np.ones((batch.shape[0], epe.EMBEDDING_DIM), dtype=np.float32)

    monkeypatch.setattr(epe, "load_full_audio", fake_load_full_audio)
    monkeypatch.setattr(epe, "run_model_batch", fake_run_model_batch)

    global_failures = []
    progress = epe.SplitProgress()
    epe.extract_split(
        model=object(),
        emb_key="output_1",
        rows=rows,
        split_name="train",
        batch_size=64,
        max_failures=1,
        max_failures_per_split=1,
        global_failures=global_failures,
        progress=progress,
    )

    assert len(progress.succeeded_rows) == 1
    assert progress.succeeded_rows[0].window_id == 2
    assert len(progress.split_failures) == 1
    assert progress.split_failures[0].window_id == 1
    embeddings = np.concatenate(progress.succeeded_embeddings, axis=0)
    assert embeddings.shape == (1, epe.EMBEDDING_DIM)


def test_extract_split_records_model_inference_failure(monkeypatch, tmp_path):
    good_wav = tmp_path / "good.wav"
    good_wav.write_bytes(b"placeholder")
    rows = [
        epe.SplitRow(
            window_id=1,
            dataset="MAP1",
            sample_rate=SR,
            sound_id=1,
            start=0,
            end=WINDOW_SPAN,
            project="MAP1",
            is_canonical=True,
            label_state="no_bird",
            spec_name="s1.npy",
            sound_filename="good.wav",
            sound_filepath=str(good_wav),
            target_codes=(),
            target_vector=(0, 0),
            original_index=0,
        )
    ]

    def fake_load_full_audio(filepath, target_sr=epe.PERCH_SR):
        return np.zeros(epe.WINDOW_SAMPLES, dtype=np.float32)

    def fake_run_model_batch(model, emb_key, batch):
        raise RuntimeError("model blew up")

    monkeypatch.setattr(epe, "load_full_audio", fake_load_full_audio)
    monkeypatch.setattr(epe, "run_model_batch", fake_run_model_batch)

    global_failures = []
    progress = epe.SplitProgress()
    with pytest.raises(epe.FailureCeilingExceeded):
        epe.extract_split(
            model=object(),
            emb_key="output_1",
            rows=rows,
            split_name="train",
            batch_size=64,
            max_failures=0,
            max_failures_per_split=0,
            global_failures=global_failures,
            progress=progress,
        )
    assert progress.succeeded_rows == []
    assert len(progress.split_failures) == 1
    assert progress.split_failures[0].reason == "model_inference_error"


def test_extract_split_decodes_shared_sound_file_exactly_once(monkeypatch, tmp_path):
    """Item 6: multiple overlapping windows on one sound_id share a single decode."""
    shared_wav = tmp_path / "shared.wav"
    shared_wav.write_bytes(b"placeholder")

    n_windows = 5
    rows = [
        epe.SplitRow(
            window_id=100 + i,
            dataset="MAP1",
            sample_rate=SR,
            sound_id=1,
            start=i * 1000,
            end=i * 1000 + WINDOW_SPAN,
            project="MAP1",
            is_canonical=True,
            label_state="no_bird",
            spec_name=f"s{i}.npy",
            sound_filename="shared.wav",
            sound_filepath=str(shared_wav),
            target_codes=(),
            target_vector=(0, 0),
            original_index=i,
        )
        for i in range(n_windows)
    ]

    call_count = {"n": 0}

    def counting_load_full_audio(filepath, target_sr=epe.PERCH_SR):
        call_count["n"] += 1
        # Large enough to cover every window's slice (start offset up to
        # (n_windows-1)*1000 samples plus one full window, resampled to target_sr).
        return np.zeros(int(SR * 20), dtype=np.float32)

    def fake_run_model_batch(model, emb_key, batch):
        return np.ones((batch.shape[0], epe.EMBEDDING_DIM), dtype=np.float32)

    monkeypatch.setattr(epe, "load_full_audio", counting_load_full_audio)
    monkeypatch.setattr(epe, "run_model_batch", fake_run_model_batch)

    global_failures = []
    progress = epe.SplitProgress()
    epe.extract_split(
        model=object(),
        emb_key="output_1",
        rows=rows,
        split_name="train",
        batch_size=64,
        max_failures=0,
        max_failures_per_split=0,
        global_failures=global_failures,
        progress=progress,
    )

    assert call_count["n"] == 1, "shared sound_id must be decoded exactly once, not once per window"
    assert len(progress.succeeded_rows) == n_windows


def test_extract_split_preserves_original_row_order_with_interleaved_sound_ids(monkeypatch, tmp_path):
    """Even though decoding is grouped by sound_id, output order matches input row order."""
    wav_a = tmp_path / "a.wav"
    wav_a.write_bytes(b"placeholder")
    wav_b = tmp_path / "b.wav"
    wav_b.write_bytes(b"placeholder")

    # Interleave sound_id 1/2/1/2 in original row order.
    rows = [
        epe.SplitRow(
            window_id=w,
            dataset="MAP1",
            sample_rate=SR,
            sound_id=sid,
            start=0,
            end=WINDOW_SPAN,
            project="MAP1",
            is_canonical=True,
            label_state="no_bird",
            spec_name=f"s{w}.npy",
            sound_filename=path.name,
            sound_filepath=str(path),
            target_codes=(),
            target_vector=(0, 0),
            original_index=idx,
        )
        for idx, (w, sid, path) in enumerate(
            [(1, 1, wav_a), (2, 2, wav_b), (3, 1, wav_a), (4, 2, wav_b)]
        )
    ]

    def fake_load_full_audio(filepath, target_sr=epe.PERCH_SR):
        return np.zeros(epe.WINDOW_SAMPLES, dtype=np.float32)

    def fake_run_model_batch(model, emb_key, batch):
        return np.ones((batch.shape[0], epe.EMBEDDING_DIM), dtype=np.float32)

    monkeypatch.setattr(epe, "load_full_audio", fake_load_full_audio)
    monkeypatch.setattr(epe, "run_model_batch", fake_run_model_batch)

    global_failures = []
    progress = epe.SplitProgress()
    epe.extract_split(
        model=object(),
        emb_key="output_1",
        rows=rows,
        split_name="train",
        batch_size=2,
        max_failures=0,
        max_failures_per_split=0,
        global_failures=global_failures,
        progress=progress,
    )

    assert [r.window_id for r in progress.succeeded_rows] == [1, 2, 3, 4]


def test_group_rows_by_sound_id_groups_and_preserves_first_encounter_order():
    rows = [
        make_split_row_with_sound_id(window_id=1, sound_id=10, original_index=0),
        make_split_row_with_sound_id(window_id=2, sound_id=20, original_index=1),
        make_split_row_with_sound_id(window_id=3, sound_id=10, original_index=2),
    ]
    groups = epe.group_rows_by_sound_id(rows)
    assert list(groups.keys()) == [10, 20]
    assert groups[10] == [0, 2]
    assert groups[20] == [1]


# --------------------------------------------------------------------------
# NPZ assembly / schema / atomic save
# --------------------------------------------------------------------------


def make_split_row(window_id, label_state="no_bird", target_vector=(0, 0), target_codes=(), original_index=None):
    return epe.SplitRow(
        window_id=window_id,
        dataset="MAP1",
        sample_rate=SR,
        sound_id=1,
        start=0,
        end=WINDOW_SPAN,
        project="MAP1",
        is_canonical=True,
        label_state=label_state,
        spec_name=f"s{window_id}.npy",
        sound_filename="a.wav",
        sound_filepath="/tmp/does/not/matter/a.wav",
        target_codes=target_codes,
        target_vector=target_vector,
        original_index=window_id if original_index is None else original_index,
    )


def make_split_row_with_sound_id(window_id, sound_id, original_index):
    return epe.SplitRow(
        window_id=window_id,
        dataset="MAP1",
        sample_rate=SR,
        sound_id=sound_id,
        start=0,
        end=WINDOW_SPAN,
        project="MAP1",
        is_canonical=True,
        label_state="no_bird",
        spec_name=f"s{window_id}.npy",
        sound_filename="a.wav",
        sound_filepath="/tmp/does/not/matter/a.wav",
        target_codes=(),
        target_vector=(0, 0),
        original_index=original_index,
    )


def test_build_npz_arrays_and_validate_schema():
    rows = [make_split_row(1), make_split_row(2, label_state="species_window", target_vector=(1, 0), target_codes=("CODE0",))]
    embeddings = np.random.rand(2, epe.EMBEDDING_DIM).astype(np.float32)
    arrays = epe.build_npz_arrays(rows, embeddings, k=2)
    epe.validate_npz_schema(arrays, k=2)
    assert arrays["window_id"].dtype == np.int64
    assert arrays["sample_rate"].dtype == np.int32
    assert arrays["is_canonical"].dtype == np.uint8
    assert arrays["target_vector"].dtype == np.uint8
    assert arrays["target_codes"].dtype.kind == "U"
    assert list(arrays["target_codes"]) == ["", "CODE0"]


def test_build_npz_arrays_rejects_embedding_shape_mismatch():
    rows = [make_split_row(1)]
    embeddings = np.zeros((1, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="embeddings shape"):
        epe.build_npz_arrays(rows, embeddings, k=2)


def test_validate_npz_schema_rejects_missing_key():
    arrays = {"embeddings": np.zeros((1, epe.EMBEDDING_DIM), dtype=np.float32)}
    with pytest.raises(ValueError, match="missing required keys"):
        epe.validate_npz_schema(arrays, k=2)


def test_validate_npz_schema_rejects_wrong_dtype():
    rows = [make_split_row(1)]
    embeddings = np.zeros((1, epe.EMBEDDING_DIM), dtype=np.float32)
    arrays = epe.build_npz_arrays(rows, embeddings, k=2)
    arrays["window_id"] = arrays["window_id"].astype(np.int32)
    with pytest.raises(ValueError, match="int64"):
        epe.validate_npz_schema(arrays, k=2)


def test_to_unicode_array_handles_empty_and_varied_lengths():
    arr = epe.to_unicode_array([])
    assert arr.shape == (0,)
    arr2 = epe.to_unicode_array(["a", "bbbb", ""])
    assert arr2.dtype.kind == "U"
    assert list(arr2) == ["a", "bbbb", ""]


def test_save_npz_atomic_writes_valid_npz_and_no_temp_left(tmp_path):
    rows = [make_split_row(1)]
    embeddings = np.zeros((1, epe.EMBEDDING_DIM), dtype=np.float32)
    arrays = epe.build_npz_arrays(rows, embeddings, k=2)
    out_path = tmp_path / "train_emb.npz"

    epe.save_npz_atomic(out_path, arrays)

    assert out_path.is_file()
    leftover = list(tmp_path.glob(".*tmp*"))
    assert leftover == []
    with np.load(out_path) as npz:
        assert npz["embeddings"].shape == (1, epe.EMBEDDING_DIM)
        assert int(npz["window_id"][0]) == 1


def test_save_npz_atomic_does_not_leave_temp_on_failure(tmp_path, monkeypatch):
    rows = [make_split_row(1)]
    embeddings = np.zeros((1, epe.EMBEDDING_DIM), dtype=np.float32)
    arrays = epe.build_npz_arrays(rows, embeddings, k=2)
    out_path = tmp_path / "train_emb.npz"

    def boom(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(epe.np, "savez", boom)
    with pytest.raises(RuntimeError):
        epe.save_npz_atomic(out_path, arrays)
    assert not out_path.exists()
    assert list(tmp_path.glob(".*tmp*")) == []


# --------------------------------------------------------------------------
# NPZ post-write round-trip assertion (item 2)
# --------------------------------------------------------------------------


def test_assert_npz_matches_rows_passes_on_correct_npz(tmp_path):
    rows = [
        make_split_row(1, label_state="species_window", target_vector=(1, 0), target_codes=("CODE0",)),
        make_split_row(2),
    ]
    embeddings = np.random.rand(2, epe.EMBEDDING_DIM).astype(np.float32)
    arrays = epe.build_npz_arrays(rows, embeddings, k=2)
    out_path = tmp_path / "train_emb.npz"
    epe.save_npz_atomic(out_path, arrays)
    epe.assert_npz_matches_rows(out_path, rows, k=2)  # must not raise


def test_assert_npz_matches_rows_detects_row_count_mismatch(tmp_path):
    rows = [make_split_row(1), make_split_row(2)]
    embeddings = np.zeros((2, epe.EMBEDDING_DIM), dtype=np.float32)
    arrays = epe.build_npz_arrays(rows, embeddings, k=2)
    out_path = tmp_path / "train_emb.npz"
    epe.save_npz_atomic(out_path, arrays)
    with pytest.raises(AssertionError, match="NPZ has"):
        epe.assert_npz_matches_rows(out_path, rows[:1], k=2)


def test_assert_npz_matches_rows_detects_order_mismatch(tmp_path):
    rows = [make_split_row(1), make_split_row(2)]
    embeddings = np.zeros((2, epe.EMBEDDING_DIM), dtype=np.float32)
    arrays = epe.build_npz_arrays(rows, embeddings, k=2)
    out_path = tmp_path / "train_emb.npz"
    epe.save_npz_atomic(out_path, arrays)
    with pytest.raises(AssertionError, match="order"):
        epe.assert_npz_matches_rows(out_path, list(reversed(rows)), k=2)


def test_assert_npz_matches_rows_detects_field_mismatch(tmp_path):
    rows = [make_split_row(1)]
    embeddings = np.zeros((1, epe.EMBEDDING_DIM), dtype=np.float32)
    arrays = epe.build_npz_arrays(rows, embeddings, k=2)
    out_path = tmp_path / "train_emb.npz"
    epe.save_npz_atomic(out_path, arrays)

    tampered_row = make_split_row(1)
    tampered_row = epe.SplitRow(
        **{**tampered_row.__dict__, "sound_id": tampered_row.sound_id + 1},
    )
    with pytest.raises(AssertionError, match="sound_id mismatch"):
        epe.assert_npz_matches_rows(out_path, [tampered_row], k=2)


def test_assert_npz_matches_rows_verifies_is_canonical_round_trip(tmp_path):
    """Item 5: is_canonical True/False survives build -> save -> reload exactly."""
    row_true = make_split_row(1)
    row_false = epe.SplitRow(**{**row_true.__dict__, "window_id": 2, "is_canonical": False})
    rows = [row_true, row_false]
    embeddings = np.zeros((2, epe.EMBEDDING_DIM), dtype=np.float32)
    arrays = epe.build_npz_arrays(rows, embeddings, k=2)
    assert list(arrays["is_canonical"]) == [1, 0]
    assert arrays["is_canonical"].dtype == np.uint8
    out_path = tmp_path / "train_emb.npz"
    epe.save_npz_atomic(out_path, arrays)
    epe.assert_npz_matches_rows(out_path, rows, k=2)  # must not raise
    with np.load(out_path) as npz:
        assert npz["is_canonical"].dtype == np.uint8
        assert list(npz["is_canonical"]) == [1, 0]


# --------------------------------------------------------------------------
# Preflight audio validation (item 1)
# --------------------------------------------------------------------------


def test_preflight_validate_audio_files_passes_for_real_readable_files(tmp_path):
    import soundfile as sf

    wav_path = tmp_path / "ok.wav"
    sf.write(str(wav_path), np.zeros(1000, dtype=np.float32), 16000)
    epe.preflight_validate_audio_files([str(wav_path)])  # must not raise


def test_preflight_validate_audio_files_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.wav"
    with pytest.raises(ValueError, match="preflight audio validation failed"):
        epe.preflight_validate_audio_files([str(missing)])


def test_preflight_validate_audio_files_raises_on_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not actually audio data")
    with pytest.raises(ValueError, match="preflight audio validation failed"):
        epe.preflight_validate_audio_files([str(corrupt)])


def test_preflight_validate_audio_files_reports_every_failure_not_just_first(tmp_path):
    missing1 = tmp_path / "missing1.wav"
    missing2 = tmp_path / "missing2.wav"
    with pytest.raises(ValueError, match="2/2"):
        epe.preflight_validate_audio_files([str(missing1), str(missing2)])


def test_preflight_validate_audio_files_dedupes_repeated_paths(tmp_path, monkeypatch):
    wav_path = tmp_path / "ok.wav"

    call_count = {"n": 0}

    class FakeInfo:
        frames = 1000

    def fake_sf_info(path):
        call_count["n"] += 1
        return FakeInfo()

    import soundfile as sf

    monkeypatch.setattr(sf, "info", fake_sf_info)
    epe.preflight_validate_audio_files([str(wav_path), str(wav_path), str(wav_path)])
    assert call_count["n"] == 1


# --------------------------------------------------------------------------
# process_and_save_split: corrupt/missing audio + max_failures=0 -> no NPZ (item 3)
# --------------------------------------------------------------------------


def test_process_and_save_split_aborts_with_no_npz_written_on_corrupt_audio(tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not actually audio data")
    rows = [
        epe.SplitRow(
            window_id=1,
            dataset="MAP1",
            sample_rate=SR,
            sound_id=1,
            start=0,
            end=WINDOW_SPAN,
            project="MAP1",
            is_canonical=True,
            label_state="no_bird",
            spec_name="s1.npy",
            sound_filename="corrupt.wav",
            sound_filepath=str(corrupt),
            target_codes=(),
            target_vector=(0, 0),
            original_index=0,
        )
    ]
    out_path = tmp_path / "train_emb.npz"
    global_failures = []
    progress = epe.SplitProgress()

    with pytest.raises(epe.FailureCeilingExceeded):
        epe.process_and_save_split(
            model=None,
            emb_key="output_1",
            rows=rows,
            split_name="train",
            batch_size=64,
            max_failures=0,
            max_failures_per_split=0,
            global_failures=global_failures,
            progress=progress,
            out_path=out_path,
            k=2,
        )

    assert not out_path.exists(), "no final NPZ may be written when max_failures=0 aborts extraction"
    assert len(progress.split_failures) == 1
    assert progress.succeeded_rows == []


def test_process_and_save_split_writes_and_round_trip_asserts_on_success(monkeypatch, tmp_path):
    good_wav = tmp_path / "good.wav"
    good_wav.write_bytes(b"placeholder")
    rows = [
        epe.SplitRow(
            window_id=1,
            dataset="MAP1",
            sample_rate=SR,
            sound_id=1,
            start=0,
            end=WINDOW_SPAN,
            project="MAP1",
            is_canonical=True,
            label_state="no_bird",
            spec_name="s1.npy",
            sound_filename="good.wav",
            sound_filepath=str(good_wav),
            target_codes=(),
            target_vector=(0, 0),
            original_index=0,
        )
    ]

    def fake_load_full_audio(filepath, target_sr=epe.PERCH_SR):
        return np.zeros(epe.WINDOW_SAMPLES, dtype=np.float32)

    def fake_run_model_batch(model, emb_key, batch):
        return np.ones((batch.shape[0], epe.EMBEDDING_DIM), dtype=np.float32)

    monkeypatch.setattr(epe, "load_full_audio", fake_load_full_audio)
    monkeypatch.setattr(epe, "run_model_batch", fake_run_model_batch)

    out_path = tmp_path / "train_emb.npz"
    global_failures = []
    progress = epe.SplitProgress()
    epe.process_and_save_split(
        model=object(),
        emb_key="output_1",
        rows=rows,
        split_name="train",
        batch_size=64,
        max_failures=0,
        max_failures_per_split=0,
        global_failures=global_failures,
        progress=progress,
        out_path=out_path,
        k=2,
    )
    assert out_path.is_file()
    with np.load(out_path) as npz:
        assert npz["window_id"][0] == 1


# --------------------------------------------------------------------------
# Progress / ETA reporting (item 7)
# --------------------------------------------------------------------------


def test_format_eta_formats_hms():
    assert epe.format_eta(0) == "0:00:00"
    assert epe.format_eta(5) == "0:00:05"
    assert epe.format_eta(65) == "0:01:05"
    assert epe.format_eta(3661) == "1:01:01"


def test_format_eta_clamps_negative_to_zero():
    assert epe.format_eta(-5) == "0:00:00"


def test_report_progress_only_prints_at_interval_and_at_end(capsys):
    epe.report_progress("train", 1, 2500, elapsed_sec=1.0, interval=1000)
    assert capsys.readouterr().out == ""

    epe.report_progress("train", 1000, 2500, elapsed_sec=1.0, interval=1000)
    out = capsys.readouterr().out
    assert "1,000/2,500" in out
    assert "ETA" in out

    epe.report_progress("train", 2500, 2500, elapsed_sec=2.5, interval=1000)
    out = capsys.readouterr().out
    assert "2,500/2,500" in out


# --------------------------------------------------------------------------
# group_rows_by_sound_id / file-grouped decode helpers (item 6)
# --------------------------------------------------------------------------


def test_slice_window_from_full_audio_matches_expected_slice_bounds():
    y_full = np.linspace(0, 1, num=int(epe.PERCH_SR * 10), dtype=np.float32)
    audio, n_padded = epe.slice_window_from_full_audio(y_full, start=0, end=WINDOW_SPAN, sample_rate=SR)
    assert audio.shape == (epe.WINDOW_SAMPLES,)
    assert n_padded == 0


def test_write_manifest_atomic_roundtrip(tmp_path):
    manifest = {"a": 1, "b": [1, 2, 3]}
    path = tmp_path / "embedding_manifest.json"
    epe.write_manifest_atomic(path, manifest)
    assert path.is_file()
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == manifest
    assert list(tmp_path.glob(".*tmp*")) == []


# --------------------------------------------------------------------------
# Hashing determinism
# --------------------------------------------------------------------------


def test_sha256_file_is_deterministic(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello world")
    assert epe.sha256_file(p) == epe.sha256_file(p)


def test_sha256_file_changes_with_content(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello world")
    h1 = epe.sha256_file(p)
    p.write_text("hello world!")
    h2 = epe.sha256_file(p)
    assert h1 != h2


def test_hash_model_dir_deterministic_and_content_sensitive(tmp_path):
    model_dir = tmp_path / "model_v2"
    (model_dir / "variables").mkdir(parents=True)
    (model_dir / "saved_model.pb").write_bytes(b"abc")
    (model_dir / "variables" / "vars.data").write_bytes(b"123")

    h1 = epe.hash_model_dir(model_dir)
    h2 = epe.hash_model_dir(model_dir)
    assert h1 == h2

    (model_dir / "variables" / "vars.data").write_bytes(b"456")
    h3 = epe.hash_model_dir(model_dir)
    assert h1 != h3


def test_hash_model_dir_sensitive_to_renamed_file(tmp_path):
    model_dir = tmp_path / "model_v2"
    model_dir.mkdir()
    (model_dir / "a.pb").write_bytes(b"abc")
    h1 = epe.hash_model_dir(model_dir)

    (model_dir / "a.pb").rename(model_dir / "b.pb")
    h2 = epe.hash_model_dir(model_dir)
    assert h1 != h2


# --------------------------------------------------------------------------
# Manifest construction / determinism
# --------------------------------------------------------------------------


def _manifest_kwargs(created_at_utc="2024-01-01T00:00:00Z"):
    return dict(
        created_at_utc=created_at_utc,
        git_commit="abc123",
        git_dirty=False,
        model_name="perch_v2",
        kaggle_slug=epe.KAGGLE_SLUG,
        model_local_dir="checkpoints/perch/model_v2",
        model_content_sha256="deadbeef",
        embedding_key="output_1",
        embedding_dim=1536,
        weights_license="UNVERIFIED",
        weights_license_source_url="https://example.invalid/weights-license",
        code_license="UNVERIFIED",
        code_license_source_url="https://example.invalid/code-license",
        extractor_source_sha256="source-hash",
        split_manifest_sha256="s1",
        class_list_path="data/splits_species_v1/class_list.json",
        class_list_sha256="s2",
        class_list_n_classes=68,
        split_csv_sha256={"train": "t", "val": "v", "test": "te"},
        extraction_params={"target_sample_rate": 32000},
        dependency_versions={"tensorflow": "2.21.0"},
        device_info={"gpu_devices": ["GPU:0"]},
        global_max_failures=0,
        per_split_max_failures=0,
        counts={"train": {"requested": 1, "succeeded": 1, "excluded": 0}},
        padding={"train": 0},
        excluded={"train": []},
        limited=False,
        limit=None,
    )


def test_build_embedding_manifest_deterministic_excluding_timestamp():
    m1 = epe.build_embedding_manifest(**_manifest_kwargs(created_at_utc="A"))
    m2 = epe.build_embedding_manifest(**_manifest_kwargs(created_at_utc="B"))
    m1.pop("created_at_utc")
    m2.pop("created_at_utc")
    assert m1 == m2


def test_build_embedding_manifest_records_failure_ceiling_note():
    m = epe.build_embedding_manifest(**_manifest_kwargs())
    assert m["failure_ceiling"]["global_max_failures"] == 0
    assert m["failure_ceiling"]["per_split_max_failures"] == 0
    assert "note" in m["failure_ceiling"]
    assert m["licenses"]["weights_license"] == "UNVERIFIED"
    assert m["model"]["embedding_dim"] == 1536


def test_build_embedding_manifest_records_license_source_urls():
    """Licenses must be traceable to a real source URL, never invented outright."""
    m = epe.build_embedding_manifest(**_manifest_kwargs())
    assert m["licenses"]["weights_license_source_url"] == "https://example.invalid/weights-license"
    assert m["licenses"]["code_license_source_url"] == "https://example.invalid/code-license"


def test_build_parser_default_license_fields_are_verified_apache2():
    """Defaults must be the verified Apache-2.0 licenses, not a placeholder sentinel."""
    parser = epe.build_parser()
    args = parser.parse_args(["--extract"])
    assert args.weights_license == "Apache-2.0"
    assert args.code_license == "Apache-2.0"
    assert args.weights_license_source_url == "https://huggingface.co/cgeorgiaw/Perch/raw/main/README.md"
    assert args.code_license_source_url == "https://github.com/google-research/perch-hoplite"
    # No Kaggle-specific license terms invented anywhere in the defaults.
    assert "kaggle" not in args.weights_license_source_url.lower()
    assert "kaggle" not in args.code_license_source_url.lower()


def test_validate_license_metadata_rejects_unverified_or_missing_fields():
    with pytest.raises(SystemExit, match="not verified"):
        epe.validate_license_metadata(
            "UNVERIFIED",
            epe.DEFAULT_WEIGHTS_LICENSE_SOURCE_URL,
            epe.DEFAULT_CODE_LICENSE,
            epe.DEFAULT_CODE_LICENSE_SOURCE_URL,
        )
    with pytest.raises(SystemExit, match="not verified"):
        epe.validate_license_metadata(
            epe.DEFAULT_WEIGHTS_LICENSE,
            "",
            epe.DEFAULT_CODE_LICENSE,
            epe.DEFAULT_CODE_LICENSE_SOURCE_URL,
        )


def test_validate_license_metadata_accepts_sourced_values():
    epe.validate_license_metadata(
        epe.DEFAULT_WEIGHTS_LICENSE,
        epe.DEFAULT_WEIGHTS_LICENSE_SOURCE_URL,
        epe.DEFAULT_CODE_LICENSE,
        epe.DEFAULT_CODE_LICENSE_SOURCE_URL,
    )


def test_build_embedding_manifest_counts_match_input():
    m = epe.build_embedding_manifest(**_manifest_kwargs())
    assert m["counts"]["train"] == {"requested": 1, "succeeded": 1, "excluded": 0}


def test_build_embedding_manifest_records_explicit_class_list_section():
    m = epe.build_embedding_manifest(**_manifest_kwargs())
    assert m["class_list"] == {
        "path": "data/splits_species_v1/class_list.json",
        "sha256": "s2",
        "n_classes": 68,
    }
    # Also mirrored inside identity_hash_inputs for the resume-hash comparison.
    assert m["identity_hash_inputs"]["class_list_sha256"] == "s2"
    assert m["identity_hash_inputs"]["extractor_source_sha256"] == "source-hash"


def test_seed_manifest_accumulators_preserves_untouched_splits():
    """Item 8: running with a --splits subset must not lose prior splits' entries."""
    existing_manifest = {
        "counts": {"val": {"requested": 10, "succeeded": 10, "excluded": 0}},
        "padding": {"val": 3},
        "excluded": {"val": []},
    }
    counts, padding, excluded = epe.seed_manifest_accumulators(existing_manifest)
    assert counts == {"val": {"requested": 10, "succeeded": 10, "excluded": 0}}
    assert padding == {"val": 3}
    assert excluded == {"val": []}

    # Simulate a later invocation that only re-extracts "train": val's entry
    # must survive unless the caller explicitly overwrites it.
    counts["train"] = {"requested": 5, "succeeded": 5, "excluded": 0}
    assert counts["val"] == {"requested": 10, "succeeded": 10, "excluded": 0}


def test_seed_manifest_accumulators_empty_manifest_yields_empty_dicts():
    counts, padding, excluded = epe.seed_manifest_accumulators({})
    assert counts == {}
    assert padding == {}
    assert excluded == {}


# --------------------------------------------------------------------------
# Resume validation
# --------------------------------------------------------------------------


def test_existing_split_output_is_valid_true_when_everything_matches(tmp_path):
    identity = {"a": 1}
    manifest = {
        "limited": False,
        "identity_hash_inputs": identity,
        "counts": {"train": {"requested": 2, "succeeded": 2, "excluded": 0}},
        "excluded": {"train": []},
    }
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    npz_path = tmp_path / "train_emb.npz"
    np.savez(npz_path, window_id=np.array([1, 2], dtype=np.int64))

    assert epe.existing_split_output_is_valid(
        npz_path, manifest_path, "train", [1, 2], identity, limited=False
    )


def test_existing_split_output_is_valid_false_when_window_ids_differ(tmp_path):
    identity = {"a": 1}
    manifest = {
        "limited": False,
        "identity_hash_inputs": identity,
        "counts": {"train": {"requested": 2, "succeeded": 2, "excluded": 0}},
        "excluded": {"train": []},
    }
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    npz_path = tmp_path / "train_emb.npz"
    np.savez(npz_path, window_id=np.array([1, 3], dtype=np.int64))

    assert not epe.existing_split_output_is_valid(
        npz_path, manifest_path, "train", [1, 2], identity, limited=False
    )


def test_existing_split_output_is_valid_false_when_hashes_differ(tmp_path):
    manifest = {
        "limited": False,
        "identity_hash_inputs": {"a": 1},
        "counts": {"train": {"requested": 2, "succeeded": 2, "excluded": 0}},
        "excluded": {"train": []},
    }
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    npz_path = tmp_path / "train_emb.npz"
    np.savez(npz_path, window_id=np.array([1, 2], dtype=np.int64))

    assert not epe.existing_split_output_is_valid(
        npz_path, manifest_path, "train", [1, 2], {"a": 2}, limited=False
    )


def test_existing_split_output_ignores_unrelated_split_csv_hash_change(tmp_path):
    identity = {
        "extractor_source_sha256": "source",
        "split_manifest_sha256": "manifest",
        "class_list_sha256": "classes",
        "model_local_dir_sha256": "model",
        "extraction_params": {"batch_size": 64},
        "split_csv_sha256": {"train": "train-a", "val": "val-a"},
    }
    manifest = {
        "limited": False,
        "identity_hash_inputs": identity,
        "counts": {"val": {"requested": 2, "succeeded": 2, "excluded": 0}},
        "excluded": {"val": []},
    }
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    npz_path = tmp_path / "val_emb.npz"
    np.savez(npz_path, window_id=np.array([1, 2], dtype=np.int64))
    expected = {**identity, "split_csv_sha256": {"train": "train-b", "val": "val-a"}}
    assert epe.existing_split_output_is_valid(
        npz_path, manifest_path, "val", [1, 2], expected, limited=False
    )


def test_existing_split_output_is_valid_false_when_limited():
    assert not epe.existing_split_output_is_valid(
        Path("/nonexistent"), Path("/nonexistent"), "train", [1, 2], {}, limited=True
    )


def test_existing_split_output_is_valid_false_when_manifest_marks_limited(tmp_path):
    identity = {"a": 1}
    manifest = {
        "limited": True,
        "identity_hash_inputs": identity,
        "counts": {"train": {"requested": 2, "succeeded": 2, "excluded": 0}},
        "excluded": {"train": []},
    }
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    npz_path = tmp_path / "train_emb.npz"
    np.savez(npz_path, window_id=np.array([1, 2], dtype=np.int64))
    assert not epe.existing_split_output_is_valid(
        npz_path, manifest_path, "train", [1, 2], identity, limited=False
    )


def test_existing_split_output_is_valid_accounts_for_excluded_rows(tmp_path):
    identity = {"a": 1}
    manifest = {
        "limited": False,
        "identity_hash_inputs": identity,
        "counts": {"train": {"requested": 3, "succeeded": 2, "excluded": 1}},
        "excluded": {"train": [{"window_id": 3, "sound_id": 1, "reason": "audio_load_error", "detail": "x"}]},
    }
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    npz_path = tmp_path / "train_emb.npz"
    np.savez(npz_path, window_id=np.array([1, 2], dtype=np.int64))
    assert epe.existing_split_output_is_valid(
        npz_path, manifest_path, "train", [1, 2, 3], identity, limited=False
    )


def test_existing_split_output_is_valid_false_when_row_count_differs(tmp_path):
    identity = {"a": 1}
    manifest = {
        "limited": False,
        "identity_hash_inputs": identity,
        "counts": {"train": {"requested": 2, "succeeded": 2, "excluded": 0}},
        "excluded": {"train": []},
    }
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    npz_path = tmp_path / "train_emb.npz"
    np.savez(npz_path, window_id=np.array([1, 2], dtype=np.int64))
    # Same window_ids in the NPZ, but the CSV now requests 3 rows: row-count
    # mismatch alone must be enough to reject, even if window_id sets overlap.
    assert not epe.existing_split_output_is_valid(
        npz_path, manifest_path, "train", [1, 2, 3], identity, limited=False
    )


def test_existing_split_output_is_valid_false_when_npz_missing(tmp_path):
    identity = {"a": 1}
    manifest = {
        "limited": False,
        "identity_hash_inputs": identity,
        "counts": {"train": {"requested": 2, "succeeded": 2, "excluded": 0}},
        "excluded": {"train": []},
    }
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    npz_path = tmp_path / "does_not_exist.npz"
    assert not epe.existing_split_output_is_valid(
        npz_path, manifest_path, "train", [1, 2], identity, limited=False
    )


# --------------------------------------------------------------------------
# Git commit helper
# --------------------------------------------------------------------------


def test_get_git_commit_not_a_repo_returns_none(tmp_path):
    sha, dirty = epe.get_git_commit(tmp_path)
    assert sha is None
    assert dirty is False


# --------------------------------------------------------------------------
# CLI parser / help
# --------------------------------------------------------------------------


def test_build_parser_requires_exactly_one_mode():
    parser = epe.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--download-v2", "--extract"])


def test_build_parser_accepts_extract_mode_with_defaults():
    parser = epe.build_parser()
    args = parser.parse_args(["--extract"])
    assert args.extract is True
    assert args.download_v2 is False
    assert args.model_dir == epe.DEFAULT_MODEL_DIR
    assert args.split_dir == epe.DEFAULT_SPLIT_DIR
    assert args.output_dir == epe.DEFAULT_OUTPUT_DIR
    assert args.batch_size == epe.DEFAULT_BATCH_SIZE
    assert args.splits == ["train", "val", "test"]
    assert args.max_failures == epe.DEFAULT_MAX_FAILURES
    assert args.max_failures_per_split == epe.DEFAULT_MAX_FAILURES_PER_SPLIT
    assert args.weights_license == epe.DEFAULT_WEIGHTS_LICENSE
    assert args.weights_license_source_url == epe.DEFAULT_WEIGHTS_LICENSE_SOURCE_URL
    assert args.code_license == epe.DEFAULT_CODE_LICENSE
    assert args.code_license_source_url == epe.DEFAULT_CODE_LICENSE_SOURCE_URL
    assert args.limit is None
    assert args.resume is False


def test_build_parser_accepts_download_v2_mode():
    parser = epe.build_parser()
    args = parser.parse_args(["--download-v2"])
    assert args.download_v2 is True
    assert args.extract is False


def test_build_parser_splits_choices_reject_invalid():
    parser = epe.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--extract", "--splits", "bogus"])


def test_build_parser_help_does_not_crash(capsys):
    parser = epe.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    captured = capsys.readouterr()
    assert "--download-v2" in captured.out
    assert "--extract" in captured.out


def test_build_parser_limit_and_resume_flags():
    parser = epe.build_parser()
    args = parser.parse_args(["--extract", "--limit", "10", "--resume"])
    assert args.limit == 10
    assert args.resume is True


def test_main_does_not_import_tensorflow_module_symbol():
    # Importing the module (already done at collection time) must not have
    # pulled tensorflow, librosa, or kagglehub into sys.modules.
    assert "tensorflow" not in sys.modules
    assert "kagglehub" not in sys.modules


def test_output_directory_lock_writes_pid_and_releases(tmp_path):
    output_dir = tmp_path / "embeddings"
    with epe.OutputDirectoryLock(output_dir):
        lock_path = output_dir / ".embedding_extraction.lock"
        assert lock_path.is_file()
        assert f"pid={epe.os.getpid()}" in lock_path.read_text()
    # The persistent lock file is harmless; the advisory lock itself is released.
    with epe.OutputDirectoryLock(output_dir):
        pass
