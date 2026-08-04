"""
Fit and evaluate a multilabel logistic-regression species probe on frozen Perch v2
embeddings (Phase 3 of ``docs/design/perch2_species_linear_probe_plan.md``).

Why
---
Phases 0-2 produce ``{train,val,test}_emb.npz`` (frozen 1536-dim Perch v2 embeddings)
and ``class_list.json`` (the fixed, ordered species vocabulary). This module is pure
array/metadata manipulation -- no audio I/O, no TensorFlow, no GPU -- so it can run
in the standard ``bioacoustics`` env and be re-run/resumed cheaply. Its correctness
depends entirely on (a) trusting the *exact* upstream artifacts (never re-deriving
labels or splits) and (b) never letting the test split influence any choice made
before it is scored exactly once, at the end, after ``C*`` is already fixed.

What
----
- Loads and exhaustively validates the three split NPZs and ``class_list.json``
  (schema, dtypes, hashes, row-identity uniqueness/disjointness, finite/nonzero
  embeddings, binary targets, ``label_state``/``target_vector`` consistency) --
  every gate is an unconditional, fail-loud check, never a silent repair.
- L2-normalizes each split's embeddings independently and statelessly
  (``sklearn.preprocessing.normalize``, no fitted scaler persisted).
- Computes train/validation eligibility before selection
  and fits ``OneVsRestClassifier(LogisticRegression(...))`` on ``trainable`` columns
  only, for every ``C`` in a pre-defined grid, scoring macro-AP on
  ``val_evaluable`` species only. The test NPZ is not opened until ``C*`` is selected
  and the train-only final refit has completed.
- Refits once at ``C*`` on train, evaluates test exactly once, and writes
  ``c_selection.csv``, ``species_diagnostics.csv``, ``species_ap.csv``,
  ``macro_ap_summary.csv``, ``no_bird_detection.csv``, ``test_predictions.npz``,
  ``logreg_species.joblib``, and ``training_manifest.json``.

How
---
Every output write is atomic (temp file in the same directory, then
``os.replace``), guarded by an exclusive ``OutputDirectoryLock`` for the whole run.
A second invocation with an unchanged configuration and matching artifact hashes
resumes (no-op) instead of re-training; any config or artifact drift triggers a
full, safe overwrite.

Convergence note: ``OneVsRestClassifier``'s default (``loky``, process-based) backend
does not propagate sub-estimator warnings to a ``warnings.catch_warnings()`` context
in the parent process (verified empirically). Convergence is therefore always derived
from each fitted estimator's ``n_iter_ >= max_iter`` (backend-independent, exactly
mirrors when lbfgs itself raises ``ConvergenceWarning``), never from a
warnings-context event count alone; the warnings context is retained only as a
best-effort diagnostic.

Parallelism note: fitting/predicting wraps ``OneVsRestClassifier`` in
``joblib.parallel_backend("threading")`` (shared-memory threads, not
``loky`` worker *processes*) so the ~700 MiB train embeddings matrix is never
pickled/duplicated per worker, and in a ``threadpoolctl.threadpool_limits(limits=1)``
context so each of the ``n_jobs`` per-species fits uses exactly one BLAS thread --
without this, ``n_jobs`` threads each running a multi-threaded BLAS call
oversubscribes the host's cores (measured: single-threaded BLAS, C=1 fits took
~1.0 s/9 iters for a rare species and ~2.64 s/29 iters for a common one on the full
train split). Default ``--n-jobs 16`` is the measured throughput optimum on this
80-core host; it is configurable and excluded from the resume config hash because it
affects wall-clock only, never results.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize as sk_normalize
from threadpoolctl import threadpool_limits

# --------------------------------------------------------------------------
# Constants / defaults
# --------------------------------------------------------------------------

EMBEDDING_DIM = 1536
SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")
THIN_SUPPORT_THRESHOLDS: Tuple[int, ...] = (5, 10)

NPZ_REQUIRED_KEYS: Tuple[str, ...] = (
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

DEFAULT_EMBEDDINGS_DIR = "data/embeddings/perch_v2/species_v1"
DEFAULT_CLASS_LIST = "data/splits_species_v1/class_list.json"
DEFAULT_OUTPUT_DIR = "checkpoints/perch/species_v1"
DEFAULT_C_GRID: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_SOLVER = "lbfgs"
DEFAULT_MAX_ITER = 1000
DEFAULT_N_JOBS = 16

VALID_LABEL_STATES = frozenset({"no_bird", "species_window"})

UNIT_NORM_ATOL = 1e-4


# --------------------------------------------------------------------------
# Hashing / git helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file, streamed to bound memory use."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_git_state(repo_dir: Path) -> Tuple[Optional[str], bool]:
    """Return ``(commit_sha, dirty)``; ``commit_sha`` is ``None`` when the tree is dirty.

    Mirrors ``extract_perch_embeddings.get_git_commit`` -- a commit sha alone does
    not describe uncommitted working-tree changes, so a dirty tree never reports a
    sha that would overstate reproducibility.
    """
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), stderr=subprocess.DEVNULL)
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


# --------------------------------------------------------------------------
# class_list.json loading / validation
# --------------------------------------------------------------------------


def load_class_list(path: Path) -> List[Dict[str, Any]]:
    """Load and validate ``class_list.json``: ordered, unique, index == position."""
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: class_list.json must be a non-empty JSON list")
    codes: List[str] = []
    for i, entry in enumerate(entries):
        for required_key in ("index", "code", "species"):
            if required_key not in entry:
                raise ValueError(f"{path}: entry {i} missing required key {required_key!r}")
        if entry["index"] != i:
            raise ValueError(f"{path}: entry {i} has index={entry['index']!r}, expected {i} (order must be dense)")
        codes.append(entry["code"])
    if len(set(codes)) != len(codes):
        dupes = sorted({c for c in codes if codes.count(c) > 1})
        raise ValueError(f"{path}: duplicate species codes in class_list.json: {dupes}")
    return entries


def class_codes(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    return [e["code"] for e in entries]


# --------------------------------------------------------------------------
# embedding_manifest.json validation
# --------------------------------------------------------------------------


def load_embedding_manifest(embeddings_dir: Path) -> Dict[str, Any]:
    path = embeddings_dir / "embedding_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"embedding_manifest.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_embedding_manifest(
    manifest: Mapping[str, Any],
    *,
    class_list_sha256: str,
    splits: Sequence[str] = SPLIT_NAMES,
) -> None:
    """Fail loud unless the embedding manifest describes a full, unlimited, clean run.

    Checks: manifest's recorded ``class_list`` hash matches the class list actually
    loaded for this run (guards against a stale/mismatched vocabulary silently
    reindexing targets); the run is not ``limited`` and has no ``limit``; every
    requested split has zero ``excluded``/``padding`` rows (Phase 2 must be complete,
    not a partial/smoke extraction).
    """
    recorded_class_hash = manifest.get("class_list", {}).get("sha256")
    if recorded_class_hash != class_list_sha256:
        raise ValueError(
            "embedding_manifest.json class_list.sha256 "
            f"({recorded_class_hash!r}) does not match the loaded class_list.json "
            f"({class_list_sha256!r}) -- refusing to train against a mismatched vocabulary."
        )
    identity_hash = manifest.get("identity_hash_inputs", {}).get("class_list_sha256")
    if identity_hash != class_list_sha256:
        raise ValueError(
            "embedding_manifest.json identity_hash_inputs.class_list_sha256 "
            f"({identity_hash!r}) does not match the loaded class_list.json ({class_list_sha256!r})."
        )
    if manifest.get("limited", False) is not False:
        raise ValueError("embedding_manifest.json is 'limited' -- Phase 3 requires a full Phase 2 extraction.")
    if manifest.get("limit") is not None:
        raise ValueError(f"embedding_manifest.json has a non-null 'limit' ({manifest.get('limit')!r}).")

    counts = manifest.get("counts", {})
    padding = manifest.get("padding", {})
    for split in splits:
        split_counts = counts.get(split)
        if split_counts is None:
            raise ValueError(f"embedding_manifest.json missing counts for split {split!r}.")
        if split_counts.get("excluded", None) != 0:
            raise ValueError(
                f"embedding_manifest.json counts[{split!r}].excluded = "
                f"{split_counts.get('excluded')!r}, expected 0."
            )
        if padding.get(split, None) != 0:
            raise ValueError(f"embedding_manifest.json padding[{split!r}] = {padding.get(split)!r}, expected 0.")


# --------------------------------------------------------------------------
# NPZ loading / validation
# --------------------------------------------------------------------------


def load_split_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load an NPZ file without pickle, materializing each array eagerly."""
    if not path.is_file():
        raise FileNotFoundError(f"Split embeddings file not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        return {name: npz[name] for name in npz.files}


def validate_npz_schema(arrays: Mapping[str, np.ndarray], k: int, split_name: str) -> None:
    """Validate an in-memory NPZ arrays mapping against the fixed Phase-2 schema."""
    missing = set(NPZ_REQUIRED_KEYS) - set(arrays.keys())
    if missing:
        raise ValueError(f"{split_name}: NPZ missing required keys: {sorted(missing)}")

    embeddings = arrays["embeddings"]
    n = embeddings.shape[0]
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"{split_name}: 'embeddings' must be shape (N, {EMBEDDING_DIM}), got {embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise ValueError(f"{split_name}: 'embeddings' must be float32, got {embeddings.dtype}")

    target_vector = arrays["target_vector"]
    if target_vector.shape != (n, k):
        raise ValueError(f"{split_name}: 'target_vector' must be shape (N, {k}), got {target_vector.shape}")
    if target_vector.dtype != np.uint8:
        raise ValueError(f"{split_name}: 'target_vector' must be uint8, got {target_vector.dtype}")

    int64_fields = ("window_id", "sound_id", "start", "end")
    for field_name in int64_fields:
        arr = arrays[field_name]
        if arr.shape != (n,):
            raise ValueError(f"{split_name}: {field_name!r} must be shape (N,), got {arr.shape}")
        if arr.dtype != np.int64:
            raise ValueError(f"{split_name}: {field_name!r} must be int64, got {arr.dtype}")

    if arrays["sample_rate"].shape != (n,):
        raise ValueError(f"{split_name}: 'sample_rate' must be shape (N,), got {arrays['sample_rate'].shape}")
    if arrays["sample_rate"].dtype != np.int32:
        raise ValueError(f"{split_name}: 'sample_rate' must be int32, got {arrays['sample_rate'].dtype}")
    if arrays["is_canonical"].shape != (n,):
        raise ValueError(f"{split_name}: 'is_canonical' must be shape (N,), got {arrays['is_canonical'].shape}")
    if arrays["is_canonical"].dtype != np.uint8:
        raise ValueError(f"{split_name}: 'is_canonical' must be uint8, got {arrays['is_canonical'].dtype}")

    string_fields = ("target_codes", "label_state", "sound_filepath", "sound_filename", "dataset", "project")
    for field_name in string_fields:
        arr = arrays[field_name]
        if arr.shape != (n,):
            raise ValueError(f"{split_name}: {field_name!r} must be shape (N,), got {arr.shape}")
        if arr.dtype.kind != "U":
            raise ValueError(f"{split_name}: {field_name!r} must be a fixed-width unicode array, got {arr.dtype}")


def validate_row_identity(arrays: Mapping[str, np.ndarray], split_name: str) -> None:
    """``window_id`` must be unique within a single split."""
    window_id = arrays["window_id"]
    n_unique = len(np.unique(window_id))
    if n_unique != len(window_id):
        raise ValueError(f"{split_name}: window_id is not unique within the split ({n_unique} unique of {len(window_id)}).")


def validate_splits_disjoint(all_arrays: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    """``sound_id`` and ``window_id`` must be pairwise disjoint across train/val/test.

    ``sound_id`` disjointness is the leakage-prevention contract itself (no audio
    file crosses a split boundary); ``window_id`` disjointness is a stronger,
    independent identity check (every window is its own row exactly once, globally).
    """
    names = list(all_arrays.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sound_overlap = set(all_arrays[a]["sound_id"].tolist()) & set(all_arrays[b]["sound_id"].tolist())
            if sound_overlap:
                raise ValueError(
                    f"sound_id leakage between {a!r} and {b!r} splits: "
                    f"{len(sound_overlap)} shared sound_id(s), e.g. {sorted(sound_overlap)[:5]}"
                )
            window_overlap = set(all_arrays[a]["window_id"].tolist()) & set(all_arrays[b]["window_id"].tolist())
            if window_overlap:
                raise ValueError(
                    f"window_id overlap between {a!r} and {b!r} splits: "
                    f"{len(window_overlap)} shared window_id(s), e.g. {sorted(window_overlap)[:5]}"
                )


def validate_finite_nonzero_embeddings(embeddings: np.ndarray, split_name: str) -> None:
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{split_name}: embeddings contain non-finite values (NaN/Inf).")
    row_norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    n_zero = int((row_norms == 0).sum())
    if n_zero:
        raise ValueError(f"{split_name}: {n_zero} embedding row(s) are all-zero -- cannot L2-normalize.")


def validate_binary_targets(target_vector: np.ndarray, split_name: str) -> None:
    unique_vals = np.unique(target_vector)
    if not set(unique_vals.tolist()).issubset({0, 1}):
        raise ValueError(f"{split_name}: target_vector contains non-binary values: {unique_vals.tolist()}")


def validate_label_state_target_consistency(
    label_state: np.ndarray, target_vector: np.ndarray, split_name: str
) -> None:
    invalid_states = set(np.unique(label_state).tolist()) - VALID_LABEL_STATES
    if invalid_states:
        raise ValueError(f"{split_name}: unexpected label_state value(s): {sorted(invalid_states)}")
    row_sums = target_vector.sum(axis=1)
    is_no_bird = label_state == "no_bird"
    is_species = label_state == "species_window"
    bad_no_bird = int(((row_sums > 0) & is_no_bird).sum())
    if bad_no_bird:
        raise ValueError(f"{split_name}: {bad_no_bird} 'no_bird' row(s) have a nonzero target_vector.")
    bad_species = int(((row_sums == 0) & is_species).sum())
    if bad_species:
        raise ValueError(f"{split_name}: {bad_species} 'species_window' row(s) have an all-zero target_vector.")


def validate_split_npz(arrays: Mapping[str, np.ndarray], k: int, split_name: str) -> None:
    """Run every per-split validation gate (schema through label-state consistency)."""
    validate_npz_schema(arrays, k, split_name)
    validate_row_identity(arrays, split_name)
    validate_finite_nonzero_embeddings(arrays["embeddings"], split_name)
    validate_binary_targets(arrays["target_vector"], split_name)
    validate_label_state_target_consistency(arrays["label_state"], arrays["target_vector"], split_name)


# --------------------------------------------------------------------------
# L2 normalization (stateless, independent per split)
# --------------------------------------------------------------------------


def l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """Stateless L2 row-normalization. No scaler is fitted or persisted."""
    return sk_normalize(embeddings, norm="l2")


def assert_unit_norm(normalized: np.ndarray, split_name: str, atol: float = UNIT_NORM_ATOL) -> None:
    norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
    max_dev = float(np.max(np.abs(norms - 1.0))) if norms.size else 0.0
    if max_dev > atol:
        raise ValueError(f"{split_name}: L2-normalized rows deviate from unit norm by up to {max_dev:.2e} (atol={atol}).")


# --------------------------------------------------------------------------
# Species eligibility
# --------------------------------------------------------------------------


@dataclass
class Eligibility:
    trainable: np.ndarray  # bool[K]
    val_evaluable: np.ndarray  # bool[K]
    test_evaluable: np.ndarray  # bool[K]
    train_pos: np.ndarray
    train_neg: np.ndarray
    val_pos: np.ndarray
    val_neg: np.ndarray
    test_pos: np.ndarray
    test_neg: np.ndarray

    @property
    def trainable_idx(self) -> np.ndarray:
        return np.flatnonzero(self.trainable)


def _pos_neg(target_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = target_vector.shape[0]
    pos = target_vector.astype(np.int64).sum(axis=0)
    neg = n - pos
    return pos, neg


def compute_eligibility(
    train_target: np.ndarray,
    val_target: np.ndarray,
    test_target: Optional[np.ndarray] = None,
) -> Eligibility:
    train_pos, train_neg = _pos_neg(train_target)
    val_pos, val_neg = _pos_neg(val_target)
    if test_target is None:
        k = train_target.shape[1]
        test_pos = np.full(k, -1, dtype=np.int64)
        test_neg = np.full(k, -1, dtype=np.int64)
        test_evaluable = np.zeros(k, dtype=bool)
    else:
        test_pos, test_neg = _pos_neg(test_target)
        test_evaluable = (test_pos >= 1) & (test_neg >= 1)
    return Eligibility(
        trainable=(train_pos >= 1) & (train_neg >= 1),
        val_evaluable=(val_pos >= 1) & (val_neg >= 1),
        test_evaluable=test_evaluable,
        train_pos=train_pos,
        train_neg=train_neg,
        val_pos=val_pos,
        val_neg=val_neg,
        test_pos=test_pos,
        test_neg=test_neg,
    )


# --------------------------------------------------------------------------
# Test-data access guard (never touch test before C* is fixed)
# --------------------------------------------------------------------------


class TestDataGuard:
    """Wraps the test split's arrays so they cannot be read before :meth:`unlock`.

    Structural guarantee #1 is that :func:`sweep_c_grid` simply never receives a
    reference to test data at all. This class is guarantee #2, in-depth: even code
    that *does* hold a reference to the wrapped mapping cannot read through it
    until ``C*`` has been selected and :meth:`unlock` explicitly called.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray]):
        self._arrays = arrays
        self._unlocked = False

    def unlock(self) -> None:
        self._unlocked = True

    def __getitem__(self, key: str) -> np.ndarray:
        if not self._unlocked:
            raise RuntimeError(
                "TestDataGuard accessed before C* was selected -- test data must never "
                "influence C selection or any other pre-refit choice."
            )
        return self._arrays[key]

    def keys(self):
        return self._arrays.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._arrays


def build_test_data_guard(
    test_arrays: Mapping[str, np.ndarray],
    normalized_embeddings: np.ndarray,
    target_vector: np.ndarray,
) -> TestDataGuard:
    """Build a locked test view whose model inputs cannot be overwritten by raw arrays."""
    return TestDataGuard(
        {
            **test_arrays,
            "embeddings": normalized_embeddings,
            "target_vector": target_vector,
        }
    )


# --------------------------------------------------------------------------
# Classifier construction / fitting
# --------------------------------------------------------------------------


def build_classifier(
    C: float, *, class_weight: str, solver: str, max_iter: int, n_jobs: Optional[int]
) -> OneVsRestClassifier:
    """``OneVsRestClassifier(LogisticRegression(...))``, no ``multi_class`` kwarg.

    ``multi_class`` was removed from ``LogisticRegression`` in scikit-learn >= 1.7
    and raises a constructor-time ``TypeError`` if passed -- it is never passed here.
    """
    base = LogisticRegression(solver=solver, max_iter=max_iter, class_weight=class_weight, C=C)
    return OneVsRestClassifier(base, n_jobs=n_jobs)


def fit_classifier_capture_warnings(
    clf: OneVsRestClassifier, X: np.ndarray, Y: np.ndarray
) -> Tuple[OneVsRestClassifier, int, float]:
    """Fit, returning ``(fitted_clf, n_raw_convergence_warnings_captured, duration_sec)``.

    The warning count is best-effort and only reliable in-process (``n_jobs`` <= 1
    or the ``threading`` backend) -- see module docstring. The authoritative,
    backend-independent convergence signal is :func:`count_non_converged`.
    """
    start = time.perf_counter()
    n_jobs = getattr(clf, "n_jobs", 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=ConvergenceWarning)
        with threadpool_limits(limits=1):
            with joblib.parallel_backend("threading", n_jobs=n_jobs):
                clf.fit(X, Y)
    duration = time.perf_counter() - start
    n_warnings = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return clf, n_warnings, duration


def count_non_converged(clf: OneVsRestClassifier, max_iter: int) -> int:
    """Count fitted sub-estimators whose ``n_iter_ >= max_iter`` (did not converge).

    Backend-independent (works regardless of ``n_jobs``/joblib backend) because it
    reads the fitted estimators' own state, not a warnings-context event log.
    Non-``LogisticRegression`` sub-estimators (e.g. sklearn's internal
    ``_ConstantPredictor`` for a degenerate constant column) have no ``n_iter_`` and
    are treated as converged (they cannot fail to converge because they never
    optimize anything) -- this should never occur for ``trainable`` columns, since
    the eligibility gate guarantees both classes are present.
    """
    n_non_converged = 0
    for est in clf.estimators_:
        n_iter = getattr(est, "n_iter_", None)
        if n_iter is None:
            continue
        if int(np.max(n_iter)) >= max_iter:
            n_non_converged += 1
    return n_non_converged


def predict_proba_reindexed(
    clf: OneVsRestClassifier, X: np.ndarray, trainable_idx: np.ndarray, k: int
) -> Tuple[np.ndarray, float]:
    """Predict probabilities for fitted columns, reindexed to the full ``[N, K]`` vocabulary.

    Non-trainable columns are ``NaN`` -- never a fabricated zero probability.
    """
    start = time.perf_counter()
    n_jobs = getattr(clf, "n_jobs", 1)
    with threadpool_limits(limits=1):
        with joblib.parallel_backend("threading", n_jobs=n_jobs):
            proba_fitted = clf.predict_proba(X)
    duration = time.perf_counter() - start
    n = X.shape[0]
    full = np.full((n, k), np.nan, dtype=np.float64)
    full[:, trainable_idx] = proba_fitted
    return full, duration


def fitted_columns_finite(
    clf: OneVsRestClassifier, trainable_idx: np.ndarray, k: int
) -> np.ndarray:
    """Full-vocabulary mask for estimators with finite coefficients/intercepts."""
    finite = np.zeros(k, dtype=bool)
    for pos, column_idx in enumerate(trainable_idx.tolist()):
        estimator = clf.estimators_[pos]
        coef = getattr(estimator, "coef_", None)
        intercept = getattr(estimator, "intercept_", None)
        finite[column_idx] = bool(
            coef is not None
            and intercept is not None
            and np.isfinite(coef).all()
            and np.isfinite(intercept).all()
        )
    return finite


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def macro_average_precision(y_true: np.ndarray, y_score: np.ndarray, columns: Sequence[int]) -> Tuple[float, int]:
    """Macro-averaged AP over ``columns`` only. Returns ``(nan, 0)`` for an empty set."""
    columns = list(columns)
    if not columns:
        return float("nan"), 0
    aps = [average_precision_score(y_true[:, j], y_score[:, j]) for j in columns]
    return float(np.mean(aps)), len(columns)


def prevalence(y_true_col: np.ndarray) -> float:
    if y_true_col.size == 0:
        return float("nan")
    return float(np.mean(y_true_col))


def any_bird_score_stable(proba_full: np.ndarray) -> np.ndarray:
    """Stable ``1 - prod_i(1 - p_i)`` via ``log1p``/``exp``.

    ``NaN`` columns (species that were never fit) contribute no evidence -- treated
    as ``p = 0`` (a factor of 1 in the product), not silently dropped from the row
    count. Uses ``log1p(-p)``/``exp`` (not a direct running product) because the
    plain product underflows silently for a large number of small probabilities;
    summing logs and exponentiating once is the numerically stable equivalent.
    """
    p = np.where(np.isnan(proba_full), 0.0, proba_full)
    p = np.clip(p, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        log_survival = np.log1p(-p)  # log(1 - p_i), -inf iff p_i == 1 exactly (mathematically valid)
    total_log_survival = np.sum(log_survival, axis=1)
    return -np.expm1(total_log_survival)


def compute_no_bird_metrics(any_bird_score: np.ndarray, y_any: np.ndarray) -> Dict[str, Any]:
    n_pos = int(np.sum(y_any))
    n_neg = int(y_any.size - n_pos)
    result: Dict[str, Any] = {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": prevalence(y_any),
    }
    if n_pos == 0 or n_neg == 0:
        result["auroc"] = float("nan")
        result["ap"] = float("nan")
    else:
        result["auroc"] = float(roc_auc_score(y_any, any_bird_score))
        result["ap"] = float(average_precision_score(y_any, any_bird_score))
    return result


# --------------------------------------------------------------------------
# C-grid sweep (train + val only -- test is never touched here)
# --------------------------------------------------------------------------


def sweep_c_grid(
    train_X: np.ndarray,
    train_Y_full: np.ndarray,
    val_X: np.ndarray,
    val_Y_full: np.ndarray,
    *,
    trainable_idx: np.ndarray,
    val_evaluable_idx: np.ndarray,
    c_grid: Sequence[float],
    class_weight: str,
    solver: str,
    max_iter: int,
    n_jobs: Optional[int],
) -> List[Dict[str, Any]]:
    """Fit on train, score macro-AP on val, for every ``C``. Never receives test data."""
    scored_idx = sorted(set(trainable_idx.tolist()) & set(val_evaluable_idx.tolist()))
    rows: List[Dict[str, Any]] = []
    train_Y_sub = train_Y_full[:, trainable_idx]
    for C in c_grid:
        clf = build_classifier(C, class_weight=class_weight, solver=solver, max_iter=max_iter, n_jobs=n_jobs)
        clf, n_raw_warnings, fit_duration = fit_classifier_capture_warnings(clf, train_X, train_Y_sub)
        n_non_converged = count_non_converged(clf, max_iter)
        finite_columns = fitted_columns_finite(clf, trainable_idx, train_Y_full.shape[1])
        val_proba_full, predict_duration = predict_proba_reindexed(clf, val_X, trainable_idx, train_Y_full.shape[1])
        val_proba_full[:, ~finite_columns] = np.nan
        finite_scored_idx = [j for j in scored_idx if finite_columns[j]]
        macro_ap, n_species = macro_average_precision(
            val_Y_full, val_proba_full, finite_scored_idx
        )
        row = {
            "C": float(C),
            "val_macro_ap": macro_ap,
            "n_species_val_evaluable": n_species,
            "fit_duration_sec": fit_duration,
            "predict_duration_sec": predict_duration,
            "n_convergence_warnings_raw": n_raw_warnings,
            "n_non_converged": n_non_converged,
        }
        val_pos = val_Y_full.sum(axis=0)
        for threshold in THIN_SUPPORT_THRESHOLDS:
            robust_idx = [j for j in finite_scored_idx if val_pos[j] >= threshold]
            robust_ap, robust_n = macro_average_precision(
                val_Y_full, val_proba_full, robust_idx
            )
            row[f"val_pos_ge{threshold}_macro_ap"] = robust_ap
            row[f"val_pos_ge{threshold}_n_species"] = robust_n
        rows.append(row)
    return rows


def select_best_c(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Argmax ``val_macro_ap``, tie-break lowest ``C``. NaN AP never wins over a real one."""
    if not rows:
        raise ValueError("c_selection rows are empty -- nothing to select from.")

    def sort_key(row: Mapping[str, Any]) -> Tuple[float, float]:
        ap = row["val_macro_ap"]
        # NaN sorts as -inf so it is never selected over any real score.
        neg_ap = -ap if not np.isnan(ap) else float("inf")
        return (neg_ap, row["C"])

    return dict(min(rows, key=sort_key))


# --------------------------------------------------------------------------
# Diagnostics (full vocabulary; NaN for non-trainable)
# --------------------------------------------------------------------------


def build_species_diagnostics(
    class_entries: Sequence[Mapping[str, Any]],
    trainable_idx: np.ndarray,
    fitted_clf: OneVsRestClassifier,
    max_iter: int,
    eligibility: Optional[Eligibility] = None,
) -> List[Dict[str, Any]]:
    """Per-species diagnostics reindexed to the full vocabulary.

    Includes explicit eligibility status (``trainable``/``val_evaluable``/
    ``test_evaluable``) and per-split supports (``train_pos``/``train_neg``/
    ``val_pos``/``val_neg``/``test_pos``/``test_neg``) alongside the fit-derived
    fields, per the design doc: "non-trainable species contain NaN for fit-derived
    fields plus their explicit eligibility status." ``eligibility`` is optional only
    so existing direct-fit-only callers/tests need not construct one; when omitted,
    eligibility/support columns are filled with ``NaN``.

    ``coef_finite`` is a hard correctness gate (checked by the caller against
    ``test_evaluable``); ``converged``/``coef_l2_norm`` are informational only --
    no arbitrary coefficient-norm ceiling excludes a species here. This also
    surfaces convergence behavior for species with extreme ``class_weight``
    ratios (rare positives under ``class_weight="balanced"``): such species are
    exactly where elevated ``n_iter``/non-convergence is most likely to appear.
    """
    trainable_set = set(trainable_idx.tolist())
    idx_to_estimator_pos = {int(idx): pos for pos, idx in enumerate(trainable_idx.tolist())}
    rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(class_entries):
        row: Dict[str, Any] = {
            "index": i,
            "code": entry["code"],
            "species": entry["species"],
            "trainable": i in trainable_set,
            "val_evaluable": bool(eligibility.val_evaluable[i]) if eligibility is not None else np.nan,
            "test_evaluable": bool(eligibility.test_evaluable[i]) if eligibility is not None else np.nan,
            "train_pos": int(eligibility.train_pos[i]) if eligibility is not None else np.nan,
            "train_neg": int(eligibility.train_neg[i]) if eligibility is not None else np.nan,
            "val_pos": int(eligibility.val_pos[i]) if eligibility is not None else np.nan,
            "val_neg": int(eligibility.val_neg[i]) if eligibility is not None else np.nan,
            "test_pos": int(eligibility.test_pos[i]) if eligibility is not None else np.nan,
            "test_neg": int(eligibility.test_neg[i]) if eligibility is not None else np.nan,
            "fit_attempted": i in trainable_set,
            "n_iter": np.nan,
            "max_iter": max_iter,
            "converged": np.nan,
            "convergence_warning": np.nan,
            "coef_finite": np.nan,
            "coef_l2_norm": np.nan,
        }
        if i in trainable_set:
            est = fitted_clf.estimators_[idx_to_estimator_pos[i]]
            n_iter = getattr(est, "n_iter_", None)
            coef = getattr(est, "coef_", None)
            intercept = getattr(est, "intercept_", None)
            if n_iter is not None:
                n_iter_scalar = int(np.max(n_iter))
                converged = n_iter_scalar < max_iter
                row["n_iter"] = n_iter_scalar
                row["converged"] = converged
                row["convergence_warning"] = not converged
            if coef is not None and intercept is not None:
                finite = bool(np.isfinite(coef).all() and np.isfinite(intercept).all())
                row["coef_finite"] = finite
                row["coef_l2_norm"] = float(np.linalg.norm(coef.ravel())) if finite else float("nan")
            else:
                row["coef_finite"] = False
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Atomic writers
# --------------------------------------------------------------------------


def _atomic_write(path: Path, mode: str, write_fn) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, mode) as fh:
            write_fn(fh)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    def _write(fh):
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    _atomic_write(path, "w", _write)


def write_json_atomic(path: Path, obj: Any) -> None:
    def _write(fh):
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")

    _atomic_write(path, "w", _write)


def save_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    def _write(fh):
        np.savez(fh, **arrays)

    _atomic_write(path, "wb", _write)


def save_joblib_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


class OutputDirectoryLock:
    """Exclusive process lock preventing concurrent training runs on the same out_dir."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.handle = None

    def __enter__(self) -> "OutputDirectoryLock":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_dir / ".train_perch_logreg.lock"
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
# Config hash / resume
# --------------------------------------------------------------------------

ARTIFACT_FILENAMES: Tuple[str, ...] = (
    "c_selection.csv",
    "species_diagnostics.csv",
    "species_ap.csv",
    "macro_ap_summary.csv",
    "no_bird_detection.csv",
    "test_predictions.npz",
    "logreg_species.joblib",
)


def compute_config_hash(
    *,
    class_list_sha256: str,
    embedding_manifest_sha256: str,
    embedding_npz_sha256: Mapping[str, str],
    c_grid: Sequence[float],
    class_weight: str,
    solver: str,
    max_iter: int,
    limit_train: Optional[int],
    script_source_sha256: str,
) -> str:
    """Hash every input that determines the *content* of the output artifacts.

    ``n_jobs`` is deliberately excluded (affects wall-clock only, never results).
    """
    payload = {
        "class_list_sha256": class_list_sha256,
        "embedding_manifest_sha256": embedding_manifest_sha256,
        "embedding_npz_sha256": dict(embedding_npz_sha256),
        "c_grid": [float(c) for c in c_grid],
        "class_weight": class_weight,
        "solver": solver,
        "max_iter": max_iter,
        "limit_train": limit_train,
        "script_source_sha256": script_source_sha256,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def try_resume(out_dir: Path, config_hash: str) -> bool:
    """Return True iff a prior run's manifest/config/artifact hashes are all current."""
    manifest_path = out_dir / "training_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("config_hash") != config_hash:
        return False
    if manifest.get("limited", False):
        return False
    recorded_hashes = manifest.get("artifact_hashes", {})
    for filename in ARTIFACT_FILENAMES:
        path = out_dir / filename
        if not path.is_file():
            return False
        if recorded_hashes.get(filename) != sha256_file(path):
            return False
    return True


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _species_ap_rows(
    class_entries: Sequence[Mapping[str, Any]],
    eligibility: Eligibility,
    test_Y_full: np.ndarray,
    test_proba_full: np.ndarray,
    coef_finite: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []
    for i, entry in enumerate(class_entries):
        effective = bool(eligibility.test_evaluable[i] and coef_finite[i])
        ap = float("nan")
        if effective:
            ap = float(average_precision_score(test_Y_full[:, i], test_proba_full[:, i]))
        rows.append(
            {
                "index": i,
                "code": entry["code"],
                "species": entry["species"],
                "trainable": bool(eligibility.trainable[i]),
                "val_evaluable": bool(eligibility.val_evaluable[i]),
                "test_evaluable": bool(eligibility.test_evaluable[i]),
                "test_evaluable_effective": effective,
                "train_pos": int(eligibility.train_pos[i]),
                "train_neg": int(eligibility.train_neg[i]),
                "val_pos": int(eligibility.val_pos[i]),
                "val_neg": int(eligibility.val_neg[i]),
                "test_pos": int(eligibility.test_pos[i]),
                "test_neg": int(eligibility.test_neg[i]),
                "test_prevalence": prevalence(test_Y_full[:, i]),
                "prevalence_baseline_ap": prevalence(test_Y_full[:, i]),
                "ap": ap,
            }
        )
    return rows


def _macro_ap_summary(
    species_ap_rows: Sequence[Mapping[str, Any]], selected_c: float
) -> Dict[str, Any]:
    eligible = [r for r in species_ap_rows if r["test_evaluable_effective"]]
    aps = [r["ap"] for r in eligible]
    baselines = [r["prevalence_baseline_ap"] for r in eligible]
    macro_ap = float(np.mean(aps)) if aps else float("nan")
    macro_baseline = float(np.mean(baselines)) if baselines else float("nan")
    summary: Dict[str, Any] = {
        "selected_c": float(selected_c),
        "macro_ap": macro_ap,
        "macro_prevalence_baseline": macro_baseline,
        "delta": macro_ap - macro_baseline if eligible else float("nan"),
        "n_eligible_species": len(eligible),
        "eligible_species_codes": ";".join(r["code"] for r in eligible),
        "n_species_excluded_nonfinite": sum(
            1 for r in species_ap_rows if r["test_evaluable"] and not r["test_evaluable_effective"]
        ),
    }
    for threshold in THIN_SUPPORT_THRESHOLDS:
        subset = [r for r in eligible if r["test_pos"] >= threshold]
        subset_aps = [r["ap"] for r in subset]
        summary[f"thin_support_ge{threshold}_macro_ap"] = float(np.mean(subset_aps)) if subset_aps else float("nan")
        summary[f"thin_support_ge{threshold}_n_species"] = len(subset)
    return summary


def run_training(args: argparse.Namespace) -> int:
    repo_dir = Path(__file__).resolve().parent
    embeddings_dir = Path(args.embeddings_dir)
    class_list_path = Path(args.class_list)
    out_dir = Path(args.out_dir)
    c_grid = tuple(args.c_grid)

    with OutputDirectoryLock(out_dir):
        class_entries = load_class_list(class_list_path)
        k = len(class_entries)
        codes = class_codes(class_entries)
        class_list_sha256 = sha256_file(class_list_path)

        embedding_manifest_path = embeddings_dir / "embedding_manifest.json"
        embedding_manifest = load_embedding_manifest(embeddings_dir)
        embedding_manifest_sha256 = sha256_file(embedding_manifest_path)
        validate_embedding_manifest(embedding_manifest, class_list_sha256=class_list_sha256, splits=SPLIT_NAMES)

        split_manifest_sha256_verified: Optional[bool] = None
        split_manifest_path = (
            Path(args.split_manifest) if args.split_manifest else class_list_path.parent / "split_manifest.json"
        )
        if split_manifest_path.is_file():
            actual_hash = sha256_file(split_manifest_path)
            recorded_hash = embedding_manifest.get("identity_hash_inputs", {}).get("split_manifest_sha256")
            split_manifest_sha256_verified = actual_hash == recorded_hash
            if not split_manifest_sha256_verified:
                raise ValueError(
                    f"split_manifest.json at {split_manifest_path} has sha256={actual_hash!r}, "
                    f"but embedding_manifest.json recorded {recorded_hash!r}."
                )
        elif args.split_manifest is not None:
            raise FileNotFoundError(f"--split-manifest given but not found: {split_manifest_path}")

        embedding_npz_sha256 = {
            split: sha256_file(embeddings_dir / f"{split}_emb.npz")
            for split in SPLIT_NAMES
        }
        script_source_sha256 = sha256_file(Path(__file__).resolve())
        config_hash = compute_config_hash(
            class_list_sha256=class_list_sha256,
            embedding_manifest_sha256=embedding_manifest_sha256,
            embedding_npz_sha256=embedding_npz_sha256,
            c_grid=c_grid,
            class_weight=args.class_weight,
            solver=args.solver,
            max_iter=args.max_iter,
            limit_train=args.limit_train,
            script_source_sha256=script_source_sha256,
        )

        if not args.force and try_resume(out_dir, config_hash):
            print(f"Resuming: {out_dir} already up-to-date for this configuration (config_hash={config_hash}).")
            return 0
        if args.limit_train is not None and out_dir.resolve() == Path(DEFAULT_OUTPUT_DIR).resolve():
            raise ValueError(
                "--limit-train is smoke/benchmark-only and may not write to the default "
                "Phase 3 output directory. Supply a separate --out-dir."
            )

        print(f"Loading NPZ splits from {embeddings_dir} ...")
        raw_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        for split in ("train", "val"):
            arrays = load_split_npz(embeddings_dir / f"{split}_emb.npz")
            validate_split_npz(arrays, k, split)
            raw_arrays[split] = arrays
        validate_splits_disjoint(raw_arrays)

        train_arrays, val_arrays = raw_arrays["train"], raw_arrays["val"]

        train_X = l2_normalize_rows(train_arrays["embeddings"])
        val_X = l2_normalize_rows(val_arrays["embeddings"])
        assert_unit_norm(train_X, "train")
        assert_unit_norm(val_X, "val")

        train_Y = train_arrays["target_vector"].astype(np.int64)
        val_Y = val_arrays["target_vector"].astype(np.int64)

        limited = args.limit_train is not None
        if limited:
            n_limit = min(args.limit_train, train_X.shape[0])
            print(f"--limit-train {args.limit_train}: restricting train to first {n_limit} rows (benchmark/smoke only).")
            train_X = train_X[:n_limit]
            train_Y = train_Y[:n_limit]

        pretest_eligibility = compute_eligibility(train_Y, val_Y)
        trainable_idx = pretest_eligibility.trainable_idx
        if trainable_idx.size == 0:
            raise RuntimeError("No species is 'trainable' (both classes present in train) -- cannot fit any model.")
        val_evaluable_idx = np.flatnonzero(pretest_eligibility.val_evaluable)
        print(
            f"Eligibility: {trainable_idx.size}/{k} trainable, {val_evaluable_idx.size}/{k} val_evaluable, "
            "test eligibility deferred until C* is fixed."
        )

        print(f"Sweeping C grid {c_grid} ...")
        c_selection_rows = sweep_c_grid(
            train_X,
            train_Y,
            val_X,
            val_Y,
            trainable_idx=trainable_idx,
            val_evaluable_idx=val_evaluable_idx,
            c_grid=c_grid,
            class_weight=args.class_weight,
            solver=args.solver,
            max_iter=args.max_iter,
            n_jobs=args.n_jobs,
        )
        best_row = select_best_c(c_selection_rows)
        selected_c = best_row["C"]
        for row in c_selection_rows:
            row["selected"] = row["C"] == selected_c
        print(f"Selected C* = {selected_c} (val_macro_ap={best_row['val_macro_ap']:.4f})")

        print(f"Refitting at C*={selected_c} on train ...")
        final_clf = build_classifier(
            selected_c, class_weight=args.class_weight, solver=args.solver, max_iter=args.max_iter, n_jobs=args.n_jobs
        )
        final_clf, n_final_raw_warnings, final_fit_duration = fit_classifier_capture_warnings(
            final_clf, train_X, train_Y[:, trainable_idx]
        )
        n_final_non_converged = count_non_converged(final_clf, args.max_iter)

        # Test is opened for the first time only now, after C* is fixed and
        # the train-only final refit has completed.
        test_arrays = load_split_npz(embeddings_dir / "test_emb.npz")
        validate_split_npz(test_arrays, k, "test")
        validate_splits_disjoint({**raw_arrays, "test": test_arrays})
        test_X_real = l2_normalize_rows(test_arrays["embeddings"])
        assert_unit_norm(test_X_real, "test")
        test_Y_real = test_arrays["target_vector"].astype(np.int64)
        eligibility = compute_eligibility(train_Y, val_Y, test_Y_real)

        diagnostics_rows = build_species_diagnostics(
            class_entries, trainable_idx, final_clf, args.max_iter, eligibility=eligibility
        )
        coef_finite = np.array(
            [bool(r["coef_finite"]) if not (isinstance(r["coef_finite"], float) and np.isnan(r["coef_finite"])) else False
             for r in diagnostics_rows]
        )

        test_proba_full, test_predict_duration = predict_proba_reindexed(final_clf, test_X_real, trainable_idx, k)
        test_proba_full[:, ~coef_finite] = np.nan

        species_ap_rows = _species_ap_rows(class_entries, eligibility, test_Y_real, test_proba_full, coef_finite)
        macro_summary = _macro_ap_summary(species_ap_rows, selected_c)

        any_bird = any_bird_score_stable(test_proba_full)
        y_any = (test_Y_real.sum(axis=1) > 0).astype(np.int64)
        no_bird_metrics = compute_no_bird_metrics(any_bird, y_any)
        no_bird_metrics["n_species_used"] = int(trainable_idx.size)

        # --- Write artifacts ---
        write_csv_atomic(
            out_dir / "c_selection.csv",
            [
                "C",
                "val_macro_ap",
                "n_species_val_evaluable",
                "fit_duration_sec",
                "predict_duration_sec",
                "n_convergence_warnings_raw",
                "n_non_converged",
                "val_pos_ge5_macro_ap",
                "val_pos_ge5_n_species",
                "val_pos_ge10_macro_ap",
                "val_pos_ge10_n_species",
                "selected",
            ],
            c_selection_rows,
        )
        write_csv_atomic(
            out_dir / "species_diagnostics.csv",
            [
                "index",
                "code",
                "species",
                "trainable",
                "val_evaluable",
                "test_evaluable",
                "train_pos",
                "train_neg",
                "val_pos",
                "val_neg",
                "test_pos",
                "test_neg",
                "fit_attempted",
                "n_iter",
                "max_iter",
                "converged",
                "convergence_warning",
                "coef_finite",
                "coef_l2_norm",
            ],
            diagnostics_rows,
        )
        write_csv_atomic(
            out_dir / "species_ap.csv",
            [
                "index",
                "code",
                "species",
                "trainable",
                "val_evaluable",
                "test_evaluable",
                "test_evaluable_effective",
                "train_pos",
                "train_neg",
                "val_pos",
                "val_neg",
                "test_pos",
                "test_neg",
                "test_prevalence",
                "prevalence_baseline_ap",
                "ap",
            ],
            species_ap_rows,
        )
        write_csv_atomic(out_dir / "macro_ap_summary.csv", list(macro_summary.keys()), [macro_summary])
        write_csv_atomic(out_dir / "no_bird_detection.csv", list(no_bird_metrics.keys()), [no_bird_metrics])

        identity_arrays = {
            name: test_arrays[name] for name in ("window_id", "sound_id", "start", "end", "sample_rate", "label_state")
        }
        save_npz_atomic(
            out_dir / "test_predictions.npz",
            {
                "probabilities": test_proba_full.astype(np.float64),
                "any_bird_score": any_bird.astype(np.float64),
                "target_vector": test_Y_real.astype(np.uint8),
                **identity_arrays,
            },
        )

        model_metadata = {
            "class_codes": codes,
            "trainable_column_indices": trainable_idx.tolist(),
            "preprocessing": {"normalization": "l2_row_independent_stateless"},
            "model_config": {
                "estimator": "OneVsRestClassifier(LogisticRegression)",
                "C": selected_c,
                "class_weight": args.class_weight,
                "solver": args.solver,
                "max_iter": args.max_iter,
            },
            "hashes": {
                "class_list_sha256": class_list_sha256,
                "embedding_manifest_sha256": embedding_manifest_sha256,
                "embedding_npz_sha256": embedding_npz_sha256,
            },
        }
        save_joblib_atomic(out_dir / "logreg_species.joblib", {"model": final_clf, "metadata": model_metadata})

        git_commit, git_dirty = get_git_state(repo_dir)
        artifact_hashes = {
            filename: sha256_file(out_dir / filename)
            for filename in ARTIFACT_FILENAMES
            if (out_dir / filename).is_file()
        }
        manifest = {
            "config_hash": config_hash,
            "embeddings_dir": str(embeddings_dir),
            "class_list_path": str(class_list_path),
            "class_list_sha256": class_list_sha256,
            "embedding_manifest_sha256": embedding_manifest_sha256,
            "embedding_npz_sha256": embedding_npz_sha256,
            "split_manifest_sha256_verified": split_manifest_sha256_verified,
            "n_classes": k,
            "c_grid": list(c_grid),
            "selected_c": selected_c,
            "class_weight": args.class_weight,
            "solver": args.solver,
            "max_iter": args.max_iter,
            "n_jobs": args.n_jobs,
            "limited": limited,
            "limit_train": args.limit_train,
            "eligibility_summary": {
                "n_trainable": int(trainable_idx.size),
                "n_val_evaluable": int(val_evaluable_idx.size),
                "n_test_evaluable": int(eligibility.test_evaluable.sum()),
                "n_test_evaluable_effective": int(macro_summary["n_eligible_species"]),
            },
            "timings_sec": {
                "final_fit_duration": final_fit_duration,
                "test_predict_duration": test_predict_duration,
                "c_grid_total_fit_duration": sum(r["fit_duration_sec"] for r in c_selection_rows),
            },
            "final_fit_n_convergence_warnings_raw": n_final_raw_warnings,
            "final_fit_n_non_converged": n_final_non_converged,
            "dependency_versions": {
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "joblib": joblib.__version__,
            },
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "script_source_sha256": script_source_sha256,
            "artifact_hashes": artifact_hashes,
        }
        write_json_atomic(out_dir / "training_manifest.json", manifest)

        if limited:
            print(
                "WARNING: this run used --limit-train and is a benchmark/smoke artifact only -- "
                "it does NOT satisfy Phase 3 completion."
            )
        print(f"Done. macro_ap={macro_summary['macro_ap']:.4f} (baseline={macro_summary['macro_prevalence_baseline']:.4f})")
        return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_c_grid(value: str) -> List[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("C grid must contain at least one value")
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid C grid value in {value!r}: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--embeddings-dir", type=str, default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--class-list", type=str, default=DEFAULT_CLASS_LIST)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Default: sibling split_manifest.json next to --class-list, verified if present.",
    )
    parser.add_argument("--c-grid", type=_parse_c_grid, default=list(DEFAULT_C_GRID))
    parser.add_argument("--class-weight", type=str, default=DEFAULT_CLASS_WEIGHT)
    parser.add_argument("--solver", type=str, default=DEFAULT_SOLVER)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Benchmark/smoke only: restrict train to the first N rows. Limited runs cannot complete Phase 3.",
    )
    parser.add_argument("--force", action="store_true", help="Skip resume check and always retrain.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main() -> None:
    args = parse_args()
    sys.exit(run_training(args))


if __name__ == "__main__":
    main()
