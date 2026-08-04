"""Tests for train_perch_logreg.py using synthetic, small data only.

Covers: NPZ/class-list schema and hash validation, L2 normalization and zero-row
rejection, eligibility computation, fitted-column reindexing to the full
vocabulary, C-grid tie-break selection, the test-data access guard (val-only
selection), AP/prevalence metrics, thin-support sensitivity subsets, the stable
any-bird-score formula, per-species diagnostics (including non-fabrication for
non-trainable/non-finite species), atomic CSV/NPZ/joblib/manifest writes,
resume-hash logic, output-directory locking, and CLI parsing.

No GPU, TensorFlow, or full real training run is required. A single small
end-to-end test exercises ``run_training`` against tiny synthetic NPZ fixtures
to validate the full artifact set is produced and resume is a true no-op.
"""

import errno
import fcntl
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_perch_logreg as tpl

EMBEDDING_DIM = tpl.EMBEDDING_DIM


# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------


def make_class_entries(k):
    return [{"index": i, "code": f"CODE{i}", "species": f"Genus species{i}"} for i in range(k)]


def write_class_list(path: Path, entries):
    path.write_text(json.dumps(entries))


def make_split_arrays(
    n,
    k,
    rng,
    *,
    sound_id_offset=0,
    window_id_offset=0,
    label_states=None,
    target_vectors=None,
    embeddings=None,
    make_finite=True,
    make_nonzero=True,
):
    """Build a fully schema-compliant NPZ arrays dict for `n` rows / `k` classes."""
    if label_states is None:
        label_states = ["no_bird"] * n
    if target_vectors is None:
        target_vectors = np.zeros((n, k), dtype=np.uint8)
        for i, state in enumerate(label_states):
            if state == "species_window":
                target_vectors[i, i % k] = 1
    else:
        target_vectors = np.asarray(target_vectors, dtype=np.uint8)

    if embeddings is None:
        embeddings = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
        # Inject a per-class signal so a real (tiny) fit is learnable in the
        # end-to-end test: row's embedding shifted along dimension j when
        # target column j is 1.
        for i in range(n):
            positive_cols = np.flatnonzero(target_vectors[i])
            for j in positive_cols:
                embeddings[i, j] += 5.0
    if not make_finite:
        embeddings = embeddings.copy()
        embeddings[0, 0] = np.nan
    if not make_nonzero:
        embeddings = embeddings.copy()
        embeddings[0, :] = 0.0

    window_id = np.arange(window_id_offset, window_id_offset + n, dtype=np.int64)
    sound_id = np.arange(sound_id_offset, sound_id_offset + n, dtype=np.int64)
    start = np.zeros(n, dtype=np.int64)
    end = np.full(n, 240000, dtype=np.int64)
    sample_rate = np.full(n, 48000, dtype=np.int32)
    is_canonical = np.ones(n, dtype=np.uint8)
    target_codes = np.array(["" for _ in range(n)], dtype="<U27")
    label_state_arr = np.array(label_states, dtype="<U14")
    sound_filepath = np.array([f"/audio/f{sid}.wav" for sid in sound_id], dtype="<U64")
    sound_filename = np.array([f"f{sid}.wav" for sid in sound_id], dtype="<U27")
    dataset = np.array(["MAP1"] * n, dtype="<U4")
    project = np.array(["MAP1"] * n, dtype="<U4")

    return {
        "embeddings": embeddings.astype(np.float32),
        "target_vector": target_vectors,
        "target_codes": target_codes,
        "label_state": label_state_arr,
        "window_id": window_id,
        "sound_id": sound_id,
        "start": start,
        "end": end,
        "sample_rate": sample_rate,
        "sound_filepath": sound_filepath,
        "sound_filename": sound_filename,
        "dataset": dataset,
        "project": project,
        "is_canonical": is_canonical,
    }


def write_npz(path: Path, arrays):
    np.savez(path, **arrays)


def make_full_embedding_manifest(class_list_sha256, split_manifest_sha256=None, **overrides):
    manifest = {
        "class_list": {"n_classes": 4, "path": "class_list.json", "sha256": class_list_sha256},
        "identity_hash_inputs": {
            "class_list_sha256": class_list_sha256,
            "split_manifest_sha256": split_manifest_sha256 or "deadbeef",
        },
        "limited": False,
        "limit": None,
        "counts": {
            "train": {"requested": 1, "succeeded": 1, "excluded": 0},
            "val": {"requested": 1, "succeeded": 1, "excluded": 0},
            "test": {"requested": 1, "succeeded": 1, "excluded": 0},
        },
        "padding": {"train": 0, "val": 0, "test": 0},
    }
    manifest.update(overrides)
    return manifest


# ==========================================================================
# class_list.json loading / validation
# ==========================================================================


def test_load_class_list_valid(tmp_path):
    entries = make_class_entries(4)
    p = tmp_path / "class_list.json"
    write_class_list(p, entries)
    loaded = tpl.load_class_list(p)
    assert tpl.class_codes(loaded) == ["CODE0", "CODE1", "CODE2", "CODE3"]


def test_load_class_list_bad_index_order_raises(tmp_path):
    entries = make_class_entries(3)
    entries[1]["index"] = 5
    p = tmp_path / "class_list.json"
    write_class_list(p, entries)
    with pytest.raises(ValueError, match="index"):
        tpl.load_class_list(p)


def test_load_class_list_duplicate_code_raises(tmp_path):
    entries = make_class_entries(3)
    entries[2]["code"] = entries[0]["code"]
    p = tmp_path / "class_list.json"
    write_class_list(p, entries)
    with pytest.raises(ValueError, match="duplicate"):
        tpl.load_class_list(p)


def test_load_class_list_empty_raises(tmp_path):
    p = tmp_path / "class_list.json"
    p.write_text("[]")
    with pytest.raises(ValueError):
        tpl.load_class_list(p)


# ==========================================================================
# embedding_manifest.json validation
# ==========================================================================


def test_validate_embedding_manifest_passes_full_clean(tmp_path):
    class_list_sha = "abc123"
    manifest = make_full_embedding_manifest(class_list_sha)
    tpl.validate_embedding_manifest(manifest, class_list_sha256=class_list_sha)  # no raise


def test_validate_embedding_manifest_class_hash_mismatch_raises(tmp_path):
    manifest = make_full_embedding_manifest("abc123")
    with pytest.raises(ValueError, match="class_list"):
        tpl.validate_embedding_manifest(manifest, class_list_sha256="different")


def test_validate_embedding_manifest_limited_rejected(tmp_path):
    manifest = make_full_embedding_manifest("abc123", limited=True)
    with pytest.raises(ValueError, match="limited"):
        tpl.validate_embedding_manifest(manifest, class_list_sha256="abc123")


def test_validate_embedding_manifest_nonnull_limit_rejected(tmp_path):
    manifest = make_full_embedding_manifest("abc123", limit=5)
    with pytest.raises(ValueError, match="limit"):
        tpl.validate_embedding_manifest(manifest, class_list_sha256="abc123")


def test_validate_embedding_manifest_excluded_nonzero_rejected(tmp_path):
    manifest = make_full_embedding_manifest("abc123")
    manifest["counts"]["train"]["excluded"] = 2
    with pytest.raises(ValueError, match="excluded"):
        tpl.validate_embedding_manifest(manifest, class_list_sha256="abc123")


def test_validate_embedding_manifest_padding_nonzero_rejected(tmp_path):
    manifest = make_full_embedding_manifest("abc123")
    manifest["padding"]["val"] = 3
    with pytest.raises(ValueError, match="padding"):
        tpl.validate_embedding_manifest(manifest, class_list_sha256="abc123")


def test_validate_embedding_manifest_missing_split_counts_rejected(tmp_path):
    manifest = make_full_embedding_manifest("abc123")
    del manifest["counts"]["test"]
    with pytest.raises(ValueError, match="counts"):
        tpl.validate_embedding_manifest(manifest, class_list_sha256="abc123")


# ==========================================================================
# NPZ schema / identity / disjointness / consistency validation
# ==========================================================================


def test_validate_npz_schema_ok():
    rng = np.random.default_rng(0)
    arrays = make_split_arrays(5, 4, rng)
    tpl.validate_npz_schema(arrays, 4, "train")  # no raise


def test_validate_npz_schema_missing_key_raises():
    rng = np.random.default_rng(0)
    arrays = make_split_arrays(5, 4, rng)
    del arrays["sound_id"]
    with pytest.raises(ValueError, match="missing required keys"):
        tpl.validate_npz_schema(arrays, 4, "train")


def test_validate_npz_schema_wrong_embedding_dtype_raises():
    rng = np.random.default_rng(0)
    arrays = make_split_arrays(5, 4, rng)
    arrays["embeddings"] = arrays["embeddings"].astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        tpl.validate_npz_schema(arrays, 4, "train")


def test_validate_npz_schema_wrong_target_vector_dtype_raises():
    rng = np.random.default_rng(0)
    arrays = make_split_arrays(5, 4, rng)
    arrays["target_vector"] = arrays["target_vector"].astype(np.int32)
    with pytest.raises(ValueError, match="uint8"):
        tpl.validate_npz_schema(arrays, 4, "train")


def test_validate_npz_schema_wrong_embedding_shape_raises():
    rng = np.random.default_rng(0)
    arrays = make_split_arrays(5, 4, rng)
    arrays["embeddings"] = rng.standard_normal((5, 10)).astype(np.float32)
    with pytest.raises(ValueError, match="1536"):
        tpl.validate_npz_schema(arrays, 4, "train")


def test_validate_npz_schema_wrong_int64_field_dtype_raises():
    rng = np.random.default_rng(0)
    arrays = make_split_arrays(5, 4, rng)
    arrays["start"] = arrays["start"].astype(np.int32)
    with pytest.raises(ValueError, match="int64"):
        tpl.validate_npz_schema(arrays, 4, "train")


def test_validate_row_identity_duplicate_window_id_raises():
    rng = np.random.default_rng(0)
    arrays = make_split_arrays(5, 4, rng)
    arrays["window_id"][1] = arrays["window_id"][0]
    with pytest.raises(ValueError, match="unique"):
        tpl.validate_row_identity(arrays, "train")


def test_validate_splits_disjoint_sound_id_overlap_raises():
    rng = np.random.default_rng(0)
    train = make_split_arrays(5, 4, rng, sound_id_offset=0, window_id_offset=0)
    val = make_split_arrays(5, 4, rng, sound_id_offset=0, window_id_offset=100)  # overlapping sound_id
    test = make_split_arrays(5, 4, rng, sound_id_offset=200, window_id_offset=200)
    with pytest.raises(ValueError, match="sound_id leakage"):
        tpl.validate_splits_disjoint({"train": train, "val": val, "test": test})


def test_validate_splits_disjoint_window_id_overlap_raises():
    rng = np.random.default_rng(0)
    train = make_split_arrays(5, 4, rng, sound_id_offset=0, window_id_offset=0)
    val = make_split_arrays(5, 4, rng, sound_id_offset=50, window_id_offset=0)  # overlapping window_id
    test = make_split_arrays(5, 4, rng, sound_id_offset=200, window_id_offset=200)
    with pytest.raises(ValueError, match="window_id overlap"):
        tpl.validate_splits_disjoint({"train": train, "val": val, "test": test})


def test_validate_splits_disjoint_ok():
    rng = np.random.default_rng(0)
    train = make_split_arrays(5, 4, rng, sound_id_offset=0, window_id_offset=0)
    val = make_split_arrays(5, 4, rng, sound_id_offset=50, window_id_offset=50)
    test = make_split_arrays(5, 4, rng, sound_id_offset=200, window_id_offset=200)
    tpl.validate_splits_disjoint({"train": train, "val": val, "test": test})  # no raise


def test_validate_finite_nonzero_embeddings_rejects_nonfinite():
    rng = np.random.default_rng(0)
    arrays = make_split_arrays(5, 4, rng, make_finite=False)
    with pytest.raises(ValueError, match="non-finite"):
        tpl.validate_finite_nonzero_embeddings(arrays["embeddings"], "train")


def test_validate_finite_nonzero_embeddings_rejects_zero_row():
    rng = np.random.default_rng(0)
    arrays = make_split_arrays(5, 4, rng, make_nonzero=False)
    with pytest.raises(ValueError, match="all-zero"):
        tpl.validate_finite_nonzero_embeddings(arrays["embeddings"], "train")


def test_validate_binary_targets_rejects_non_binary():
    tv = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="non-binary"):
        tpl.validate_binary_targets(tv, "train")


def test_validate_label_state_rejects_unknown_state():
    label_state = np.array(["no_bird", "weird_state"], dtype="<U14")
    tv = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="label_state"):
        tpl.validate_label_state_target_consistency(label_state, tv, "train")


def test_validate_label_state_no_bird_with_nonzero_target_raises():
    label_state = np.array(["no_bird"], dtype="<U14")
    tv = np.array([[1, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="no_bird"):
        tpl.validate_label_state_target_consistency(label_state, tv, "train")


def test_validate_label_state_species_window_with_zero_target_raises():
    label_state = np.array(["species_window"], dtype="<U14")
    tv = np.array([[0, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="species_window"):
        tpl.validate_label_state_target_consistency(label_state, tv, "train")


def test_validate_label_state_consistent_ok():
    label_state = np.array(["no_bird", "species_window"], dtype="<U14")
    tv = np.array([[0, 0], [1, 0]], dtype=np.uint8)
    tpl.validate_label_state_target_consistency(label_state, tv, "train")  # no raise


# ==========================================================================
# L2 normalization
# ==========================================================================


def test_l2_normalize_rows_unit_norm():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, EMBEDDING_DIM)).astype(np.float32) * 3.0
    Xn = tpl.l2_normalize_rows(X)
    norms = np.linalg.norm(Xn, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_assert_unit_norm_raises_on_deviation():
    bad = np.ones((3, 4), dtype=np.float32) * 2.0  # norm = 4, not 1
    with pytest.raises(ValueError, match="unit norm"):
        tpl.assert_unit_norm(bad, "train")


def test_assert_unit_norm_passes_for_normalized():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((5, 8)).astype(np.float32)
    Xn = tpl.l2_normalize_rows(X)
    tpl.assert_unit_norm(Xn, "train")  # no raise


# ==========================================================================
# Eligibility
# ==========================================================================


def test_compute_eligibility_trainable_requires_both_classes():
    k = 3
    train_Y = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0]])  # col0: pos+neg, col1: all neg, col2: all neg
    val_Y = np.array([[1, 0, 0], [0, 0, 0]])
    test_Y = np.array([[1, 1, 0], [0, 0, 0]])
    elig = tpl.compute_eligibility(train_Y, val_Y, test_Y)
    assert elig.trainable.tolist() == [True, False, False]
    assert elig.val_evaluable.tolist() == [True, False, False]
    assert elig.test_evaluable.tolist() == [True, True, False]
    assert elig.trainable_idx.tolist() == [0]


def test_compute_eligibility_counts_correct():
    train_Y = np.array([[1, 0], [0, 0], [1, 1]])
    val_Y = np.array([[1, 0]])
    test_Y = np.array([[0, 1], [0, 1]])
    elig = tpl.compute_eligibility(train_Y, val_Y, test_Y)
    assert elig.train_pos.tolist() == [2, 1]
    assert elig.train_neg.tolist() == [1, 2]
    assert elig.test_pos.tolist() == [0, 2]
    assert elig.test_neg.tolist() == [2, 0]


# ==========================================================================
# Test-data access guard
# ==========================================================================


def test_test_data_guard_raises_before_unlock():
    guard = tpl.TestDataGuard({"embeddings": np.zeros((2, 2))})
    with pytest.raises(RuntimeError, match="before C\\* was selected"):
        guard["embeddings"]


def test_test_data_guard_allows_after_unlock():
    arr = np.ones((2, 2))
    guard = tpl.TestDataGuard({"embeddings": arr})
    guard.unlock()
    np.testing.assert_array_equal(guard["embeddings"], arr)


def test_build_test_data_guard_keeps_normalized_arrays_over_raw_arrays():
    raw = {
        "embeddings": np.full((2, 2), 9.0),
        "target_vector": np.zeros((2, 1), dtype=np.uint8),
        "window_id": np.array([1, 2]),
    }
    normalized = np.array([[1.0, 0.0], [0.0, 1.0]])
    targets = np.ones((2, 1), dtype=np.int64)
    guard = tpl.build_test_data_guard(raw, normalized, targets)
    guard.unlock()
    np.testing.assert_array_equal(guard["embeddings"], normalized)
    np.testing.assert_array_equal(guard["target_vector"], targets)


def test_compute_eligibility_can_defer_test_counts():
    train = np.array([[1, 0], [0, 0]])
    val = np.array([[1, 0], [0, 0]])
    eligibility = tpl.compute_eligibility(train, val)
    assert eligibility.trainable.tolist() == [True, False]
    assert eligibility.val_evaluable.tolist() == [True, False]
    assert eligibility.test_evaluable.tolist() == [False, False]
    assert eligibility.test_pos.tolist() == [-1, -1]


def test_sweep_c_grid_never_receives_test_data():
    """Structural guarantee: sweep_c_grid's signature has no 'test' parameter."""
    import inspect

    sig = inspect.signature(tpl.sweep_c_grid)
    for name in sig.parameters:
        assert "test" not in name.lower()


# ==========================================================================
# C-grid selection / tie-break
# ==========================================================================


def test_select_best_c_prefers_higher_val_macro_ap():
    rows = [
        {"C": 0.1, "val_macro_ap": 0.5},
        {"C": 1.0, "val_macro_ap": 0.8},
        {"C": 10.0, "val_macro_ap": 0.6},
    ]
    best = tpl.select_best_c(rows)
    assert best["C"] == 1.0


def test_select_best_c_tie_break_lowest_c():
    rows = [
        {"C": 10.0, "val_macro_ap": 0.9},
        {"C": 0.01, "val_macro_ap": 0.9},
        {"C": 1.0, "val_macro_ap": 0.9},
    ]
    best = tpl.select_best_c(rows)
    assert best["C"] == 0.01


def test_select_best_c_nan_never_selected_over_real_value():
    rows = [{"C": 0.01, "val_macro_ap": float("nan")}, {"C": 1.0, "val_macro_ap": 0.1}]
    best = tpl.select_best_c(rows)
    assert best["C"] == 1.0


def test_select_best_c_empty_raises():
    with pytest.raises(ValueError):
        tpl.select_best_c([])


def test_sweep_c_grid_fits_only_trainable_and_scores_val_only():
    rng = np.random.default_rng(3)
    k = 3
    n = 60
    # Column 2 constant-negative in train => not trainable.
    train_Y = np.zeros((n, k), dtype=np.int64)
    train_Y[: n // 2, 0] = 1
    train_Y[n // 4 : 3 * n // 4, 1] = 1
    train_X = rng.standard_normal((n, 8)).astype(np.float64)
    for i in range(n):
        for j in range(k):
            if train_Y[i, j]:
                train_X[i, j % 8] += 4.0

    val_Y = np.zeros((20, k), dtype=np.int64)
    val_Y[:5, 0] = 1
    val_Y[5:10, 1] = 1
    val_X = rng.standard_normal((20, 8)).astype(np.float64)
    for i in range(20):
        for j in range(k):
            if val_Y[i, j]:
                val_X[i, j % 8] += 4.0

    elig = tpl.compute_eligibility(train_Y, val_Y, val_Y)
    trainable_idx = elig.trainable_idx
    assert 2 not in trainable_idx.tolist()

    rows = tpl.sweep_c_grid(
        train_X,
        train_Y,
        val_X,
        val_Y,
        trainable_idx=trainable_idx,
        val_evaluable_idx=np.flatnonzero(elig.val_evaluable),
        c_grid=[0.1, 1.0],
        class_weight="balanced",
        solver="lbfgs",
        max_iter=200,
        n_jobs=1,
    )
    assert len(rows) == 2
    for row in rows:
        assert not np.isnan(row["val_macro_ap"])
        assert row["n_species_val_evaluable"] == 2  # only columns 0 and 1
        assert "val_pos_ge5_macro_ap" in row
        assert "val_pos_ge10_n_species" in row


# ==========================================================================
# Predict reindexing to full vocabulary
# ==========================================================================


class _FakeEstimator:
    def __init__(self, n_iter=5, coef=None, intercept=None):
        self.n_iter_ = np.array([n_iter])
        self.coef_ = coef if coef is not None else np.array([[1.0, 2.0]])
        self.intercept_ = intercept if intercept is not None else np.array([0.5])


class _FakeClassifier:
    def __init__(self, estimators, proba):
        self.estimators_ = estimators
        self._proba = proba

    def predict_proba(self, X):
        return self._proba


def test_predict_proba_reindexed_nan_for_untrained_columns():
    n, k = 4, 5
    trainable_idx = np.array([1, 3])
    proba_fitted = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    clf = _FakeClassifier([_FakeEstimator(), _FakeEstimator()], proba_fitted)
    full, duration = tpl.predict_proba_reindexed(clf, np.zeros((n, 2)), trainable_idx, k)
    assert full.shape == (n, k)
    assert np.isnan(full[:, 0]).all()
    assert np.isnan(full[:, 2]).all()
    assert np.isnan(full[:, 4]).all()
    np.testing.assert_array_equal(full[:, 1], proba_fitted[:, 0])
    np.testing.assert_array_equal(full[:, 3], proba_fitted[:, 1])
    assert duration >= 0


class _FakeFitClassifier:
    """Minimal stand-in for OneVsRestClassifier exposing only what fit needs."""

    def __init__(self, n_jobs):
        self.n_jobs = n_jobs
        self.fit_called_with = None

    def fit(self, X, Y):
        self.fit_called_with = (X, Y)
        return self


def test_fit_classifier_capture_warnings_uses_threadpool_limits_and_threading_backend(monkeypatch):
    calls = {"threadpool_limits": [], "parallel_backend": []}

    class _NullCtx:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    def fake_threadpool_limits(limits=None):
        calls["threadpool_limits"].append(limits)
        return _NullCtx()

    def fake_parallel_backend(backend, n_jobs=None):
        calls["parallel_backend"].append((backend, n_jobs))
        return _NullCtx()

    monkeypatch.setattr(tpl, "threadpool_limits", fake_threadpool_limits)
    monkeypatch.setattr(tpl.joblib, "parallel_backend", fake_parallel_backend)

    clf = _FakeFitClassifier(n_jobs=6)
    X = np.zeros((2, 2))
    Y = np.zeros((2, 2), dtype=np.int64)
    fitted, n_warnings, duration = tpl.fit_classifier_capture_warnings(clf, X, Y)

    assert fitted is clf
    assert clf.fit_called_with is not None
    assert n_warnings == 0
    assert duration >= 0
    assert calls["threadpool_limits"] == [1]
    assert calls["parallel_backend"] == [("threading", 6)]


def test_predict_proba_reindexed_uses_threadpool_limits_and_threading_backend(monkeypatch):
    calls = {"threadpool_limits": [], "parallel_backend": []}

    class _NullCtx:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    def fake_threadpool_limits(limits=None):
        calls["threadpool_limits"].append(limits)
        return _NullCtx()

    def fake_parallel_backend(backend, n_jobs=None):
        calls["parallel_backend"].append((backend, n_jobs))
        return _NullCtx()

    monkeypatch.setattr(tpl, "threadpool_limits", fake_threadpool_limits)
    monkeypatch.setattr(tpl.joblib, "parallel_backend", fake_parallel_backend)

    proba_fitted = np.array([[0.1], [0.2]])
    clf = _FakeClassifier([_FakeEstimator()], proba_fitted)
    clf.n_jobs = 4
    full, duration = tpl.predict_proba_reindexed(clf, np.zeros((2, 1)), np.array([1]), 3)

    assert full.shape == (2, 3)
    assert duration >= 0
    assert calls["threadpool_limits"] == [1]
    assert calls["parallel_backend"] == [("threading", 4)]


def test_count_non_converged_counts_only_maxed_out():
    clf = _FakeClassifier([_FakeEstimator(n_iter=5), _FakeEstimator(n_iter=100)], proba=None)
    assert tpl.count_non_converged(clf, max_iter=100) == 1
    assert tpl.count_non_converged(clf, max_iter=200) == 0


# ==========================================================================
# Diagnostics
# ==========================================================================


def test_build_species_diagnostics_non_trainable_has_nan_fields():
    entries = make_class_entries(3)
    trainable_idx = np.array([0])
    clf = _FakeClassifier([_FakeEstimator(n_iter=5)], proba=None)
    rows = tpl.build_species_diagnostics(entries, trainable_idx, clf, max_iter=100)
    assert rows[0]["trainable"] is True
    assert rows[0]["fit_attempted"] is True
    assert rows[0]["converged"] is True
    assert rows[0]["coef_finite"] is True
    for row in rows[1:]:
        assert row["trainable"] is False
        assert row["fit_attempted"] is False
        assert np.isnan(row["n_iter"])
        assert (row["coef_finite"] is False) or np.isnan(row["coef_finite"])


def test_build_species_diagnostics_nonfinite_coef_flagged():
    entries = make_class_entries(1)
    trainable_idx = np.array([0])
    bad_est = _FakeEstimator(n_iter=5, coef=np.array([[np.nan, 1.0]]))
    clf = _FakeClassifier([bad_est], proba=None)
    rows = tpl.build_species_diagnostics(entries, trainable_idx, clf, max_iter=100)
    assert rows[0]["coef_finite"] is False


def test_build_species_diagnostics_non_converged_flagged():
    entries = make_class_entries(1)
    trainable_idx = np.array([0])
    clf = _FakeClassifier([_FakeEstimator(n_iter=100)], proba=None)
    rows = tpl.build_species_diagnostics(entries, trainable_idx, clf, max_iter=100)
    assert rows[0]["converged"] is False
    assert rows[0]["convergence_warning"] is True


def test_build_species_diagnostics_maps_gapped_trainable_columns_correctly():
    entries = make_class_entries(3)
    trainable_idx = np.array([0, 2])
    clf = _FakeClassifier(
        [
            _FakeEstimator(n_iter=3, coef=np.array([[1.0, 0.0]])),
            _FakeEstimator(n_iter=7, coef=np.array([[0.0, 2.0]])),
        ],
        proba=None,
    )
    rows = tpl.build_species_diagnostics(entries, trainable_idx, clf, max_iter=100)
    assert rows[0]["n_iter"] == 3
    assert np.isnan(rows[1]["n_iter"])
    assert rows[2]["n_iter"] == 7
    assert rows[2]["coef_l2_norm"] == pytest.approx(2.0)


def test_build_species_diagnostics_includes_eligibility_and_supports_when_provided():
    # 3 species: col 0 trainable+val_evaluable+test_evaluable, col 1 trainable only
    # (val/test not evaluable), col 2 not trainable at all -- exercises the full
    # "supports, eligibility" columns required alongside fit-derived fields.
    entries = make_class_entries(3)
    trainable_idx = np.array([0, 1])
    clf = _FakeClassifier([_FakeEstimator(n_iter=5), _FakeEstimator(n_iter=7)], proba=None)
    train_Y = np.array([[1, 0, 0]] * 5 + [[0, 1, 0]] * 5)
    val_Y = np.array([[1, 0, 0]] * 5 + [[0, 0, 0]] * 5)
    test_Y = np.array([[1, 0, 0]] * 5 + [[0, 0, 0]] * 5)
    eligibility = tpl.compute_eligibility(train_Y, val_Y, test_Y)

    rows = tpl.build_species_diagnostics(entries, trainable_idx, clf, max_iter=100, eligibility=eligibility)

    assert rows[0]["trainable"] is True
    assert rows[0]["val_evaluable"] is True
    assert rows[0]["test_evaluable"] is True
    assert rows[0]["train_pos"] == 5
    assert rows[0]["train_neg"] == 5
    assert rows[0]["val_pos"] == 5
    assert rows[0]["test_pos"] == 5

    assert rows[1]["trainable"] is True
    assert rows[1]["val_evaluable"] is False
    assert rows[1]["test_evaluable"] is False

    assert rows[2]["trainable"] is False
    assert rows[2]["val_evaluable"] is False
    assert rows[2]["test_evaluable"] is False
    assert rows[2]["train_pos"] == 0


def test_build_species_diagnostics_eligibility_omitted_yields_nan_columns():
    entries = make_class_entries(2)
    trainable_idx = np.array([0])
    clf = _FakeClassifier([_FakeEstimator(n_iter=5)], proba=None)
    rows = tpl.build_species_diagnostics(entries, trainable_idx, clf, max_iter=100)
    for row in rows:
        assert np.isnan(row["val_evaluable"])
        assert np.isnan(row["test_evaluable"])
        assert np.isnan(row["train_pos"])


# ==========================================================================
# AP metrics / prevalence / thin support
# ==========================================================================


def test_macro_average_precision_correct():
    y_true = np.array([[0, 1], [1, 0], [0, 1], [1, 1]])
    y_score = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.6]])
    macro_ap, n = tpl.macro_average_precision(y_true, y_score, [0, 1])
    assert n == 2
    assert 0.0 <= macro_ap <= 1.0


def test_macro_average_precision_empty_columns_is_nan():
    y_true = np.zeros((3, 2))
    y_score = np.zeros((3, 2))
    macro_ap, n = tpl.macro_average_precision(y_true, y_score, [])
    assert n == 0
    assert np.isnan(macro_ap)


def test_prevalence_basic():
    assert tpl.prevalence(np.array([0, 1, 1, 0])) == 0.5
    assert tpl.prevalence(np.array([])) != tpl.prevalence(np.array([]))  # NaN != NaN


def test_prevalence_baseline_ap_equals_constant_score_average_precision():
    # The design's prevalence baseline (test_pos / test_n) must equal the AP a
    # constant-score (uninformative) classifier would achieve -- not merely a
    # convenient proxy. Verified against sklearn's actual AP computation for
    # several class-balance ratios (including highly imbalanced).
    rng = np.random.default_rng(7)
    for p in (0.02, 0.15, 0.5, 0.9):
        y = (rng.random(500) < p).astype(np.int64)
        const_score = np.full(500, 0.4321)  # any constant works -- ties only
        baseline = tpl.prevalence(y)
        constant_score_ap = average_precision_score(y, const_score)
        assert baseline == pytest.approx(constant_score_ap, abs=1e-12)
        assert baseline == pytest.approx(y.sum() / y.shape[0], abs=1e-12)  # test_pos / test_n


def test_macro_ap_summary_thin_support_subsets():
    entries = make_class_entries(3)
    elig = tpl.Eligibility(
        trainable=np.array([True, True, True]),
        val_evaluable=np.array([True, True, True]),
        test_evaluable=np.array([True, True, True]),
        train_pos=np.array([10, 10, 10]),
        train_neg=np.array([10, 10, 10]),
        val_pos=np.array([5, 5, 5]),
        val_neg=np.array([5, 5, 5]),
        test_pos=np.array([2, 6, 12]),
        test_neg=np.array([8, 4, 3]),
    )
    n = 10
    test_Y = np.zeros((n, 3), dtype=np.int64)
    test_Y[:2, 0] = 1
    test_Y[:6, 1] = 1
    test_Y[:, 2] = np.array([1] * 8 + [0] * 2)  # 8 positives (index mismatch vs elig.test_pos is fine; AP uses this)
    proba = np.column_stack([np.linspace(0, 1, n) for _ in range(3)])
    coef_finite = np.array([True, True, True])
    species_rows = tpl._species_ap_rows(entries, elig, test_Y, proba, coef_finite)
    summary = tpl._macro_ap_summary(species_rows, selected_c=1.0)
    assert summary["n_eligible_species"] == 3
    assert summary["thin_support_ge5_n_species"] == 2  # test_pos: 2,6,12 -> >=5: 6,12
    assert summary["thin_support_ge10_n_species"] == 1  # only 12
    assert "macro_prevalence_baseline" in summary
    assert "prevalence_baseline_ap" in species_rows[0]


# ==========================================================================
# Nonfinite exclusion from aggregates
# ==========================================================================


def test_species_ap_rows_nonfinite_forces_effective_false():
    entries = make_class_entries(2)
    elig = tpl.Eligibility(
        trainable=np.array([True, True]),
        val_evaluable=np.array([True, True]),
        test_evaluable=np.array([True, True]),
        train_pos=np.array([5, 5]),
        train_neg=np.array([5, 5]),
        val_pos=np.array([3, 3]),
        val_neg=np.array([3, 3]),
        test_pos=np.array([4, 4]),
        test_neg=np.array([4, 4]),
    )
    test_Y = np.zeros((8, 2), dtype=np.int64)
    test_Y[:4, 0] = 1
    test_Y[:4, 1] = 1
    proba = np.random.default_rng(0).uniform(size=(8, 2))
    coef_finite = np.array([True, False])  # species 1 nonfinite
    rows = tpl._species_ap_rows(entries, elig, test_Y, proba, coef_finite)
    assert rows[0]["test_evaluable_effective"] is True
    assert not np.isnan(rows[0]["ap"])
    assert rows[1]["test_evaluable_effective"] is False
    assert np.isnan(rows[1]["ap"])

    summary = tpl._macro_ap_summary(rows, selected_c=1.0)
    assert summary["n_eligible_species"] == 1
    assert summary["n_species_excluded_nonfinite"] == 1


def test_sweep_c_grid_excludes_nonfinite_fitted_column(monkeypatch):
    train_X = np.zeros((6, 2))
    val_X = np.zeros((4, 2))
    train_Y = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 1]])
    val_Y = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    clf = _FakeClassifier(
        [
            _FakeEstimator(coef=np.array([[1.0, 0.0]])),
            _FakeEstimator(coef=np.array([[np.nan, 0.0]])),
        ],
        proba=np.array([[0.1, np.nan], [0.8, np.nan], [0.2, np.nan], [0.9, np.nan]]),
    )

    monkeypatch.setattr(
        tpl,
        "build_classifier",
        lambda *args, **kwargs: clf,
    )
    monkeypatch.setattr(
        tpl,
        "fit_classifier_capture_warnings",
        lambda fitted, X, Y: (fitted, 0, 0.1),
    )
    rows = tpl.sweep_c_grid(
        train_X,
        train_Y,
        val_X,
        val_Y,
        trainable_idx=np.array([0, 1]),
        val_evaluable_idx=np.array([0, 1]),
        c_grid=[1.0],
        class_weight="balanced",
        solver="lbfgs",
        max_iter=100,
        n_jobs=1,
    )
    assert rows[0]["n_species_val_evaluable"] == 1
    assert np.isfinite(rows[0]["val_macro_ap"])


# ==========================================================================
# Stable any-bird score
# ==========================================================================


def test_any_bird_score_matches_naive_product():
    proba = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9]])
    stable = tpl.any_bird_score_stable(proba)
    naive = 1.0 - np.prod(1.0 - proba, axis=1)
    np.testing.assert_allclose(stable, naive, atol=1e-9)


def test_any_bird_score_monotonic_in_each_p():
    base = np.array([[0.2, 0.3, 0.4]])
    higher = np.array([[0.5, 0.3, 0.4]])
    assert tpl.any_bird_score_stable(higher)[0] > tpl.any_bird_score_stable(base)[0]


def test_any_bird_score_nan_treated_as_zero_contribution():
    proba_with_nan = np.array([[0.5, np.nan]])
    proba_without = np.array([[0.5, 0.0]])
    np.testing.assert_allclose(tpl.any_bird_score_stable(proba_with_nan), tpl.any_bird_score_stable(proba_without))


def test_any_bird_score_p_equals_one_saturates_to_one():
    proba = np.array([[1.0, 0.0]])
    result = tpl.any_bird_score_stable(proba)
    assert result[0] == pytest.approx(1.0)


def test_compute_no_bird_metrics_basic():
    any_bird = np.array([0.1, 0.9, 0.2, 0.8])
    y_any = np.array([0, 1, 0, 1])
    metrics = tpl.compute_no_bird_metrics(any_bird, y_any)
    assert metrics["n_positive"] == 2
    assert metrics["n_negative"] == 2
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["ap"] == pytest.approx(1.0)


def test_compute_no_bird_metrics_degenerate_all_one_class():
    any_bird = np.array([0.1, 0.9])
    y_any = np.array([1, 1])
    metrics = tpl.compute_no_bird_metrics(any_bird, y_any)
    assert np.isnan(metrics["auroc"])
    assert np.isnan(metrics["ap"])


# ==========================================================================
# Atomic writers
# ==========================================================================


def test_write_csv_atomic(tmp_path):
    path = tmp_path / "out.csv"
    tpl.write_csv_atomic(path, ["a", "b"], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert path.is_file()
    content = path.read_text()
    assert "a,b" in content
    assert "1,2" in content
    # No leftover temp files.
    assert list(tmp_path.glob(".*.tmp")) == []


def test_write_json_atomic(tmp_path):
    path = tmp_path / "out.json"
    tpl.write_json_atomic(path, {"x": 1})
    assert json.loads(path.read_text()) == {"x": 1}


def test_save_npz_atomic(tmp_path):
    path = tmp_path / "out.npz"
    tpl.save_npz_atomic(path, {"a": np.arange(5)})
    with np.load(path) as npz:
        np.testing.assert_array_equal(npz["a"], np.arange(5))


def test_save_joblib_atomic(tmp_path):
    import joblib

    path = tmp_path / "out.joblib"
    tpl.save_joblib_atomic(path, {"x": [1, 2, 3]})
    assert joblib.load(path) == {"x": [1, 2, 3]}


def test_atomic_write_cleans_up_temp_file_on_failure(tmp_path):
    path = tmp_path / "out.csv"

    def _boom(fh):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        tpl._atomic_write(path, "w", _boom)
    assert not path.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_output_directory_lock_is_exclusive(tmp_path):
    with tpl.OutputDirectoryLock(tmp_path):
        lock_path = tmp_path / ".train_perch_logreg.lock"
        assert lock_path.is_file()
        # A second, independent open+flock(NB) must fail while held.
        fh2 = open(lock_path, "a+")
        try:
            with pytest.raises(OSError):
                fcntl.flock(fh2.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            fh2.close()


def test_output_directory_lock_releases_after_context(tmp_path):
    with tpl.OutputDirectoryLock(tmp_path):
        pass
    lock_path = tmp_path / ".train_perch_logreg.lock"
    fh2 = open(lock_path, "a+")
    try:
        fcntl.flock(fh2.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # must succeed now
        fcntl.flock(fh2.fileno(), fcntl.LOCK_UN)
    finally:
        fh2.close()


# ==========================================================================
# Config hash / resume
# ==========================================================================


def _config_hash_kwargs(**overrides):
    kwargs = dict(
        class_list_sha256="abc",
        embedding_manifest_sha256="def",
        embedding_npz_sha256={"train": "t", "val": "v", "test": "x"},
        c_grid=[0.01, 0.1, 1.0, 10.0],
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        limit_train=None,
        script_source_sha256="ghi",
    )
    kwargs.update(overrides)
    return kwargs


def test_compute_config_hash_deterministic():
    h1 = tpl.compute_config_hash(**_config_hash_kwargs())
    h2 = tpl.compute_config_hash(**_config_hash_kwargs())
    assert h1 == h2


def test_compute_config_hash_changes_with_c_grid():
    h1 = tpl.compute_config_hash(**_config_hash_kwargs())
    h2 = tpl.compute_config_hash(**_config_hash_kwargs(c_grid=[0.01, 0.1]))
    assert h1 != h2


@pytest.mark.parametrize(
    "field,value",
    [
        ("class_weight", "none"),
        ("solver", "saga"),
        ("max_iter", 500),
        ("limit_train", 10),
        ("embedding_npz_sha256", {"train": "changed", "val": "v", "test": "x"}),
    ],
)
def test_compute_config_hash_changes_with_model_or_input_config(field, value):
    h1 = tpl.compute_config_hash(**_config_hash_kwargs())
    h2 = tpl.compute_config_hash(**_config_hash_kwargs(**{field: value}))
    assert h1 != h2


def test_compute_config_hash_ignores_n_jobs_by_construction():
    # n_jobs is not a parameter at all -- assert the function signature excludes it.
    import inspect

    assert "n_jobs" not in inspect.signature(tpl.compute_config_hash).parameters


def test_try_resume_no_manifest_returns_false(tmp_path):
    assert tpl.try_resume(tmp_path, "somehash") is False


def test_try_resume_matches_returns_true(tmp_path):
    config_hash = "matchhash"
    artifact_content = {}
    for filename in tpl.ARTIFACT_FILENAMES:
        p = tmp_path / filename
        p.write_bytes(b"content-" + filename.encode())
        artifact_content[filename] = tpl.sha256_file(p)
    manifest = {"config_hash": config_hash, "limited": False, "artifact_hashes": artifact_content}
    (tmp_path / "training_manifest.json").write_text(json.dumps(manifest))
    assert tpl.try_resume(tmp_path, config_hash) is True


def test_try_resume_config_hash_mismatch_returns_false(tmp_path):
    for filename in tpl.ARTIFACT_FILENAMES:
        (tmp_path / filename).write_bytes(b"x")
    manifest = {"config_hash": "old", "limited": False, "artifact_hashes": {}}
    (tmp_path / "training_manifest.json").write_text(json.dumps(manifest))
    assert tpl.try_resume(tmp_path, "new") is False


def test_try_resume_artifact_hash_mismatch_returns_false(tmp_path):
    config_hash = "h"
    artifact_hashes = {}
    for filename in tpl.ARTIFACT_FILENAMES:
        p = tmp_path / filename
        p.write_bytes(b"original")
        artifact_hashes[filename] = tpl.sha256_file(p)
    manifest = {"config_hash": config_hash, "limited": False, "artifact_hashes": artifact_hashes}
    (tmp_path / "training_manifest.json").write_text(json.dumps(manifest))
    # Mutate one artifact after the manifest was written.
    (tmp_path / tpl.ARTIFACT_FILENAMES[0]).write_bytes(b"tampered")
    assert tpl.try_resume(tmp_path, config_hash) is False


def test_try_resume_missing_artifact_file_returns_false(tmp_path):
    config_hash = "h"
    manifest = {"config_hash": config_hash, "limited": False, "artifact_hashes": {}}
    (tmp_path / "training_manifest.json").write_text(json.dumps(manifest))
    assert tpl.try_resume(tmp_path, config_hash) is False


def test_try_resume_limited_manifest_never_resumes(tmp_path):
    config_hash = "h"
    artifact_hashes = {}
    for filename in tpl.ARTIFACT_FILENAMES:
        p = tmp_path / filename
        p.write_bytes(b"x")
        artifact_hashes[filename] = tpl.sha256_file(p)
    manifest = {"config_hash": config_hash, "limited": True, "artifact_hashes": artifact_hashes}
    (tmp_path / "training_manifest.json").write_text(json.dumps(manifest))
    assert tpl.try_resume(tmp_path, config_hash) is False


# ==========================================================================
# CLI parsing
# ==========================================================================


def test_build_arg_parser_defaults():
    args = tpl.parse_args([])
    assert args.embeddings_dir == tpl.DEFAULT_EMBEDDINGS_DIR
    assert args.class_list == tpl.DEFAULT_CLASS_LIST
    assert args.out_dir == tpl.DEFAULT_OUTPUT_DIR
    assert args.c_grid == [0.01, 0.1, 1.0, 10.0]
    assert args.class_weight == "balanced"
    assert args.solver == "lbfgs"
    assert args.max_iter == 1000
    assert args.n_jobs == 16
    assert args.limit_train is None
    assert args.force is False


def test_parse_c_grid_valid():
    assert tpl._parse_c_grid("0.01,0.1,1,10") == [0.01, 0.1, 1.0, 10.0]
    assert tpl._parse_c_grid(" 0.5 , 2 ") == [0.5, 2.0]


def test_parse_c_grid_invalid_raises():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        tpl._parse_c_grid("not_a_number")


def test_parse_c_grid_empty_raises():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        tpl._parse_c_grid("")


def test_cli_overrides_apply():
    args = tpl.parse_args(
        [
            "--embeddings-dir",
            "some/dir",
            "--class-list",
            "some/class_list.json",
            "--out-dir",
            "some/out",
            "--c-grid",
            "0.5,5",
            "--n-jobs",
            "4",
            "--limit-train",
            "100",
            "--force",
        ]
    )
    assert args.embeddings_dir == "some/dir"
    assert args.c_grid == [0.5, 5.0]
    assert args.n_jobs == 4
    assert args.limit_train == 100
    assert args.force is True


# ==========================================================================
# Small end-to-end run (tiny synthetic data, real fit, no GPU/TF)
# ==========================================================================


def _write_full_fixture(tmp_path, k=4, n_per_split=40):
    rng = np.random.default_rng(42)
    entries = make_class_entries(k)
    class_list_path = tmp_path / "splits" / "class_list.json"
    class_list_path.parent.mkdir(parents=True)
    write_class_list(class_list_path, entries)

    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()

    def build(split_name, sound_offset):
        label_states = []
        for i in range(n_per_split):
            label_states.append("species_window" if i % 3 == 0 else "no_bird")
        arrays = make_split_arrays(
            n_per_split, k, rng, sound_id_offset=sound_offset, window_id_offset=sound_offset, label_states=label_states
        )
        write_npz(embeddings_dir / f"{split_name}_emb.npz", arrays)

    build("train", 0)
    build("val", 10_000)
    build("test", 20_000)

    class_list_sha = tpl.sha256_file(class_list_path)
    manifest = make_full_embedding_manifest(class_list_sha)
    (embeddings_dir / "embedding_manifest.json").write_text(json.dumps(manifest))
    return class_list_path, embeddings_dir


def test_run_training_end_to_end_produces_all_artifacts(tmp_path, capsys):
    class_list_path, embeddings_dir = _write_full_fixture(tmp_path)
    out_dir = tmp_path / "out"

    args = tpl.parse_args(
        [
            "--embeddings-dir",
            str(embeddings_dir),
            "--class-list",
            str(class_list_path),
            "--out-dir",
            str(out_dir),
            "--c-grid",
            "0.1,1.0",
            "--max-iter",
            "200",
        ]
    )
    rc = tpl.run_training(args)
    assert rc == 0

    for filename in (
        "c_selection.csv",
        "species_diagnostics.csv",
        "species_ap.csv",
        "macro_ap_summary.csv",
        "no_bird_detection.csv",
        "test_predictions.npz",
        "logreg_species.joblib",
        "training_manifest.json",
    ):
        assert (out_dir / filename).is_file(), f"missing {filename}"

    manifest = json.loads((out_dir / "training_manifest.json").read_text())
    assert manifest["limited"] is False
    assert manifest["selected_c"] in [0.1, 1.0]
    assert set(manifest["artifact_hashes"].keys()) == set(tpl.ARTIFACT_FILENAMES)

    with np.load(out_dir / "test_predictions.npz") as npz:
        assert npz["probabilities"].shape == (40, 4)
        assert npz["probabilities"].dtype == np.float64
        assert npz["any_bird_score"].dtype == np.float64
        assert npz["any_bird_score"].shape == (40,)
        assert npz["target_vector"].shape == (40, 4)

    # Re-run: must resume (no-op), not re-train.
    capsys.readouterr()
    rc2 = tpl.run_training(args)
    assert rc2 == 0
    out = capsys.readouterr().out
    assert "Resuming" in out


def test_run_training_final_test_predict_receives_normalized_not_raw_embeddings(tmp_path, monkeypatch):
    """The final test prediction receives L2-normalized, not raw, embeddings."""
    class_list_path, embeddings_dir = _write_full_fixture(tmp_path)
    out_dir = tmp_path / "out_norm_check"

    captured_calls = []
    real_predict_proba_reindexed = tpl.predict_proba_reindexed

    def spy_predict_proba_reindexed(clf, X, trainable_idx, k):
        captured_calls.append(np.array(X, copy=True))
        return real_predict_proba_reindexed(clf, X, trainable_idx, k)

    monkeypatch.setattr(tpl, "predict_proba_reindexed", spy_predict_proba_reindexed)

    args = tpl.parse_args(
        [
            "--embeddings-dir",
            str(embeddings_dir),
            "--class-list",
            str(class_list_path),
            "--out-dir",
            str(out_dir),
            "--c-grid",
            "0.1,1.0",
            "--max-iter",
            "200",
        ]
    )
    rc = tpl.run_training(args)
    assert rc == 0

    # 2 C values swept on val -> 2 calls, then exactly 1 final call on test.
    assert len(captured_calls) == 3
    final_test_X = captured_calls[-1]

    row_norms = np.linalg.norm(final_test_X, axis=1)
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-5)

    # Sanity: the raw on-disk test embeddings are emphatically NOT unit-norm
    # (rng.standard_normal over 1536 dims), so this is a meaningful assertion,
    # not a coincidence of the fixture already being normalized.
    with np.load(embeddings_dir / "test_emb.npz") as npz:
        raw_norms = np.linalg.norm(npz["embeddings"], axis=1)
    assert np.all(raw_norms > 10.0)
    assert not np.allclose(row_norms, raw_norms)


def test_run_training_does_not_open_test_npz_until_sweep_and_refit_finish(tmp_path, monkeypatch):
    class_list_path, embeddings_dir = _write_full_fixture(tmp_path)
    out_dir = tmp_path / "out_test_order"
    events = []

    real_load = tpl.load_split_npz
    real_fit = tpl.fit_classifier_capture_warnings

    def spy_load(path):
        events.append(f"load:{Path(path).stem}")
        return real_load(path)

    fit_count = {"n": 0}

    def spy_fit(clf, X, Y):
        fit_count["n"] += 1
        events.append(f"fit:{fit_count['n']}")
        return real_fit(clf, X, Y)

    monkeypatch.setattr(tpl, "load_split_npz", spy_load)
    monkeypatch.setattr(tpl, "fit_classifier_capture_warnings", spy_fit)

    args = tpl.parse_args(
        [
            "--embeddings-dir",
            str(embeddings_dir),
            "--class-list",
            str(class_list_path),
            "--out-dir",
            str(out_dir),
            "--c-grid",
            "0.1,1.0",
            "--max-iter",
            "200",
        ]
    )
    assert tpl.run_training(args) == 0
    assert events.index("load:test_emb") > events.index("fit:3")


def test_run_training_limit_train_marks_manifest_limited(tmp_path):
    class_list_path, embeddings_dir = _write_full_fixture(tmp_path)
    out_dir = tmp_path / "out_limited"
    args = tpl.parse_args(
        [
            "--embeddings-dir",
            str(embeddings_dir),
            "--class-list",
            str(class_list_path),
            "--out-dir",
            str(out_dir),
            "--c-grid",
            "1.0",
            "--max-iter",
            "50",
            "--limit-train",
            "10",
        ]
    )
    rc = tpl.run_training(args)
    assert rc == 0
    manifest = json.loads((out_dir / "training_manifest.json").read_text())
    assert manifest["limited"] is True
    assert manifest["limit_train"] == 10


def test_run_training_rejects_mismatched_class_list_hash(tmp_path):
    class_list_path, embeddings_dir = _write_full_fixture(tmp_path)
    # Tamper with the embedding manifest's recorded class list hash.
    manifest_path = embeddings_dir / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["class_list"]["sha256"] = "wronghash"
    manifest_path.write_text(json.dumps(manifest))

    out_dir = tmp_path / "out_bad"
    args = tpl.parse_args(
        [
            "--embeddings-dir",
            str(embeddings_dir),
            "--class-list",
            str(class_list_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    with pytest.raises(ValueError, match="class_list"):
        tpl.run_training(args)
