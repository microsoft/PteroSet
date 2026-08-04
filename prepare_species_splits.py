"""
Reproducible species-classification train/val/test split generator.

Why
---
The existing leave-one-project-out folds (``prepare_dataset.py`` / ``splits``
step) target the binary Birds/No-Birds task and keep every identification
annotation, including calls that were never resolved to a species, and calls
resolved only to a coarse family/order-level placeholder. A species
classification benchmark instead needs: (1) a closed, supported species
vocabulary built on a *canonicalized* taxonomy (duplicate codes for the same
scientific name merged, non-species placeholders removed), (2) windows whose
label is unambiguous (a true no-bird window, or a window where every
overlapping call is a real, resolved species -- rare/OOV calls co-occurring
with a retained species are allowed, but they never contribute a positive
target of their own), and (3) a single held-out train/val/test split (not
leave-one-project-out) that preserves per-species and no-bird proportions as
close to 70/15/15 as the hard grouping/coverage constraints allow, without
ever letting one recording (``sound_id``) cross splits.

What
----
For every window in ``windows_mapping_*.json`` this script determines a
``label_state``:

- ``no_bird``: overlaps no identification annotation -> all-zero multilabel
  target, included.
- ``resolved_clean``: overlaps >= 1 identification annotation, every
  overlapping call is resolved to a real (non-placeholder) species, and at
  least one of those species survives the vocabulary support gate -> the
  window is included, with its target restricted to the *retained* species
  present (an excluded, low-support species co-occurring in the same window
  is silently dropped from the target vector, but does not exclude the
  window).
- ``unresolved_or_mixed``: overlaps >= 1 call that is not resolved to a real
  species -- either it never matched a species-level annotation at all, or
  it matched one of the non-species placeholder codes (see
  :class:`TaxonomyCrosswalk`) -- excluded, even if another overlapping call
  was cleanly resolved.
- ``excluded_all_oov``: every overlapping call is resolved to a real species,
  but none of them survive the vocabulary support gate -> excluded (never
  coerced to an all-zero vector, which would silently misrepresent a purely
  out-of-vocabulary call as no-bird).

How
---
``sound_id`` is the indivisible grouping unit (strictly -- no site/event
grouping is used). Split *assignment and audit* are defined on canonical
(non-overlapping, 5 s tiling) windows only, so the 70/15/15 ratio is exactly
auditable in window counts. ``train_split.csv`` is then augmented with every
other (overlapping) modeled window belonging to a train-assigned sound_id;
``val_split.csv``/``test_split.csv`` and ``canonical_train_split.csv`` remain
canonical-only, so the audited base distribution is always reconstructable
from ``canonical_train_split.csv`` + ``val_split.csv`` + ``test_split.csv``.

Group -> split assignment is done in two phases: (1) a deterministic greedy
reservation pass that pins the minimum set of sound_ids needed to guarantee
every retained species has >= 1 positive sound_id in every split (a hard
constraint; infeasibility is a fatal error naming the affected species), then
(2) a deterministic repeated-greedy + local-search optimizer (seeded,
multi-restart) that assigns the remaining sound_ids to minimize a
chi-square-like distance between per-split and global proportions for split
size, no-bird prevalence and per-species positive-window prevalence. Project
distribution is reported but is not part of that objective; it is
tie-broken with a token weight only, so it can never trade away species/
no-bird balance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# Label-state vocabulary
# --------------------------------------------------------------------------

NO_BIRD = "no_bird"
RESOLVED_CLEAN = "species_window"
UNRESOLVED_OR_MIXED = "excluded_mixed_unresolved"
EXCLUDED_ALL_OOV = "excluded_rare_only"

MODELED_STATES = (NO_BIRD, RESOLVED_CLEAN)
SPLIT_NAMES = ("train", "val", "test")

# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IdAnnotation:
    """One identification-level (call) annotation, resolved to a real,
    canonicalized species code only if it matched a species-level annotation
    whose code is not a non-species placeholder."""

    t_min: float
    t_max: float
    resolved: bool
    species_code: Optional[str]


@dataclass
class ClassifiedWindow:
    """A window after label-state classification and vocabulary gating.
    ``species_codes`` holds the *final* target set: empty for ``no_bird`` /
    ``unresolved_or_mixed`` / ``excluded_all_oov``, and the retained-species
    subset (never the full raw overlap) for ``resolved_clean``."""

    window_id: int
    sound_id: int
    project: str
    sample_rate: int
    start: int
    end: int
    is_canonical: bool
    label_state: str
    species_codes: FrozenSet[str]


@dataclass
class SoundStats:
    """Per-``sound_id`` aggregate of canonical modeled-window statistics --
    ``sound_id`` is the indivisible split-assignment unit (no site/event
    grouping)."""

    project: str
    n_modeled: int = 0
    n_nobird: int = 0
    species_counts: Counter = field(default_factory=Counter)
    positive_species: FrozenSet[str] = frozenset()


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file, streamed to bound memory use."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_species_csv(path: Path) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# Taxonomy resolution: non-species placeholders + duplicate-code merging
# --------------------------------------------------------------------------


_SP_TOKEN_RE = re.compile(r"sp\.?", re.IGNORECASE)
_DASH_CHARS = {"-", "\u2013", "\u2014"}


def normalize_species_name(name: str) -> str:
    """Collapse internal whitespace and lowercase, for duplicate-name
    detection (``"Daptrius  chimachima"`` == ``"Daptrius chimachima"``)."""
    return re.sub(r"\s+", " ", name.strip()).lower()


def is_placeholder_species_name(name: str) -> bool:
    """A ``species.csv`` row is a non-species placeholder (family/order-level
    tag, never a real resolvable species) if its scientific-name field is
    either a bare dash (missing name) or contains a standalone ``sp``/``sp.``
    token (an explicit "unidentified species" marker), e.g. ``"Tyrannidae sp
    1"`` or ``"Psittacidae sp."``. This general rule -- applied to the
    current ``data/species.csv`` -- identifies the approved placeholders,
    including bare higher-taxon names such as ``Picidae``."""
    stripped = name.strip()
    if stripped in _DASH_CHARS:
        return True
    tokens = re.split(r"\s+", stripped)
    return len(tokens) < 2 or any(_SP_TOKEN_RE.fullmatch(t) for t in tokens)


@dataclass
class TaxonomyCrosswalk:
    """Resolved species.csv taxonomy: which raw codes are non-species
    placeholders, and which raw codes canonicalize to which representative
    code (duplicate scientific names merged to a single deterministic
    representative -- the alphabetically first code in the duplicate
    group)."""

    known_codes: Set[str]
    placeholder_codes: Set[str]
    canonical_of: Dict[str, str]  # raw non-placeholder code -> canonical code
    duplicate_groups: List[Tuple[str, List[str]]]  # (canonical, [raw codes]), only len > 1
    species_name_of: Dict[str, str]  # raw code -> species.csv scientific name (for reporting)

    def is_placeholder(self, raw_code: str) -> bool:
        return raw_code in self.placeholder_codes

    def canonicalize(self, raw_code: str) -> str:
        return self.canonical_of[raw_code]


def build_taxonomy_crosswalk(species_ref: Sequence[dict]) -> TaxonomyCrosswalk:
    known_codes = {row["code"] for row in species_ref}
    placeholder_codes: Set[str] = set()
    by_name: Dict[str, List[str]] = defaultdict(list)
    species_name_of: Dict[str, str] = {}

    for row in species_ref:
        code, name = row["code"], row["species"]
        species_name_of[code] = name
        if is_placeholder_species_name(name):
            placeholder_codes.add(code)
            continue
        by_name[normalize_species_name(name)].append(code)

    canonical_of: Dict[str, str] = {}
    duplicate_groups: List[Tuple[str, List[str]]] = []
    for _, codes in by_name.items():
        codes_sorted = sorted(codes)
        canonical = codes_sorted[0]
        for c in codes_sorted:
            canonical_of[c] = canonical
        if len(codes_sorted) > 1:
            duplicate_groups.append((canonical, codes_sorted))
    duplicate_groups.sort(key=lambda g: g[0])

    return TaxonomyCrosswalk(
        known_codes=known_codes,
        placeholder_codes=placeholder_codes,
        canonical_of=canonical_of,
        duplicate_groups=duplicate_groups,
        species_name_of=species_name_of,
    )


# --------------------------------------------------------------------------
# Annotation identity matching (sound_id + geometry key)
# --------------------------------------------------------------------------


def geometry_key(anno: dict, decimals: int) -> Tuple:
    """Stable identity key for an annotation: sound_id + rounded geometry.

    Rounding to ``decimals`` places acts as an explicit tolerance for
    matching identification annotations to their species-resolved
    counterpart, guarding against floating point noise while still being
    exact for this dataset (species annotations are byte-identical copies
    of their identification annotation on (t_min, t_max, f_min, f_max)).
    """
    return (
        anno["sound_id"],
        round(anno["t_min"], decimals),
        round(anno["t_max"], decimals),
        round(anno["f_min"], decimals),
        round(anno["f_max"], decimals),
    )


def build_species_lookup(species_annotations: Sequence[dict], decimals: int) -> Dict[Tuple, str]:
    """Map geometry key -> raw species.csv code for every species-level
    annotation (including placeholder codes -- taxonomy resolution to
    "unresolved" happens later, in :func:`build_identification_records`)."""
    lookup: Dict[Tuple, str] = {}
    for anno in species_annotations:
        key = geometry_key(anno, decimals)
        if key in lookup and lookup[key] != anno["category"]:
            raise ValueError(
                f"Ambiguous species annotation geometry collision at key {key}: "
                f"{lookup[key]} vs {anno['category']}. Increase --geometry-decimals."
            )
        lookup[key] = anno["category"]
    return lookup


def build_identification_records(
    identification_annotations: Sequence[dict],
    species_lookup: Dict[Tuple, str],
    taxonomy: TaxonomyCrosswalk,
    decimals: int,
) -> Dict[int, List[IdAnnotation]]:
    """Group identification annotations by sound_id, resolving species where
    possible. A call is only ``resolved`` if it matched a species-level
    annotation *and* that annotation's code is a real (non-placeholder)
    species -- a coarse/placeholder match marks the identification event
    unresolved, exactly like a call that never matched anything."""
    by_sound: Dict[int, List[IdAnnotation]] = defaultdict(list)
    for anno in identification_annotations:
        key = geometry_key(anno, decimals)
        raw_code = species_lookup.get(key)
        if raw_code is None or taxonomy.is_placeholder(raw_code):
            resolved, species_code = False, None
        else:
            resolved, species_code = True, taxonomy.canonicalize(raw_code)
        by_sound[anno["sound_id"]].append(
            IdAnnotation(t_min=anno["t_min"], t_max=anno["t_max"], resolved=resolved, species_code=species_code)
        )

    # Data-integrity check: every species annotation must resolve to some
    # identification annotation (species annotations are a strict subset),
    # regardless of whether its code turns out to be a placeholder.
    id_keys = {geometry_key(a, decimals) for a in identification_annotations}
    unmatched = [k for k in species_lookup if k not in id_keys]
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} species annotations do not match any "
            "identification annotation geometry; matching assumption violated."
        )
    return by_sound


# --------------------------------------------------------------------------
# Window classification
# --------------------------------------------------------------------------


def window_time_bounds(window: dict) -> Tuple[float, float]:
    """Window (start, end) sample indices converted to seconds."""
    sr = window["sample_rate"]
    return window["start"] / sr, window["end"] / sr


def classify_window(window: dict, id_by_sound: Dict[int, List[IdAnnotation]]) -> Tuple[str, FrozenSet[str]]:
    """Classify a single window into a label_state and its species target set
    (pre-vocabulary-gate; ``resolved_clean`` here means "all overlapping
    calls resolved to a real species", not yet restricted to the retained
    vocabulary -- see :func:`finalize_label_state`).

    Overlap uses strict interval overlap on time only (matching the
    windows-mapping generation convention: ``event_end > start and
    event_start < end``), so touching boundaries do not count as overlap.
    """
    t_start, t_end = window_time_bounds(window)
    annos = id_by_sound.get(window["sound_id"], [])
    overlapping = [a for a in annos if a.t_max > t_start and a.t_min < t_end]

    if not overlapping:
        return NO_BIRD, frozenset()
    if any(not a.resolved for a in overlapping):
        return UNRESOLVED_OR_MIXED, frozenset()
    return RESOLVED_CLEAN, frozenset(a.species_code for a in overlapping)


def infer_canonical_window_size(windows: Sequence[dict]) -> int:
    """Infer the fixed window duration (samples) from the data itself."""
    durations = {w["end"] - w["start"] for w in windows}
    if len(durations) != 1:
        raise ValueError(f"Windows have inconsistent durations: {sorted(durations)}")
    return durations.pop()


def is_canonical(window: dict, window_size_samples: int) -> bool:
    """A canonical window is one of the non-overlapping tiling of a sound."""
    return window["start"] % window_size_samples == 0


# --------------------------------------------------------------------------
# Vocabulary construction (>= 7 distinct sound_ids support gate) & OOV gate
# --------------------------------------------------------------------------


def compute_species_support(
    classified: Sequence[ClassifiedWindow],
    vocab_filter: Optional[Set[str]] = None,
) -> Dict[str, Set[int]]:
    """Tally, over *canonical* resolved_clean windows, the set of distinct
    ``sound_id``s each species appears in.

    If ``vocab_filter`` is given, a window that would end up
    ``excluded_all_oov`` under that candidate vocabulary (i.e. its full raw
    species set does not intersect ``vocab_filter`` at all) is dropped from
    the tally -- but a window that mixes a retained species with an excluded
    one still counts towards the retained species' support (only fully-OOV
    windows are dropped; mixed-but-partially-retained windows are kept, per
    the approved design).
    """
    support: Dict[str, Set[int]] = defaultdict(set)
    for cw in classified:
        if not cw.is_canonical or cw.label_state != RESOLVED_CLEAN:
            continue
        codes = cw.species_codes if vocab_filter is None else (cw.species_codes & vocab_filter)
        if vocab_filter is not None and not codes:
            continue  # fully out-of-vocabulary window: contributes to no one's support
        for code in codes:
            support[code].add(cw.sound_id)
    return support


def build_vocabulary(
    support: Dict[str, Set[int]],
    min_sound_ids: int,
    candidates: Optional[Set[str]] = None,
) -> Set[str]:
    """Species with >= ``min_sound_ids`` distinct supporting sound_ids,
    restricted to ``candidates`` if given (defaults to every species with
    any support)."""
    species_pool = candidates if candidates is not None else set(support)
    vocab = {s for s in species_pool if len(support.get(s, set())) >= min_sound_ids}
    if not vocab:
        raise ValueError(
            f"No species satisfy the support gate (min_sound_ids={min_sound_ids}); "
            "relax the threshold or check inputs."
        )
    return vocab


def build_vocabulary_fixed_point(
    classified: Sequence[ClassifiedWindow],
    min_sound_ids: int,
    max_iterations: int = 1000,
) -> Tuple[Set[str], List[dict]]:
    """Support-gated vocabulary, rechecked to a fixed point.

    A species' own support (distinct sound_ids where it appears in a
    canonical resolved_clean window) does not depend on whether some *other*
    co-occurring species is retained, because only fully-OOV windows are
    dropped (mixed windows survive as long as the species itself is
    retained). The fixed point is therefore reached immediately in practice;
    the loop is kept as a defensive, explicitly-verified mechanism (rather
    than assuming single-pass correctness) and to produce an auditable
    per-round history for the manifest.
    """
    history: List[dict] = []
    support = compute_species_support(classified, vocab_filter=None)
    vocab = build_vocabulary(support, min_sound_ids)
    history.append(
        {
            "round": 0,
            "support": {s: len(v) for s, v in support.items()},
            "vocab_size": len(vocab),
            "dropped": sorted(set(support) - vocab),
        }
    )
    for round_idx in range(1, max_iterations + 1):
        support = compute_species_support(classified, vocab_filter=vocab)
        new_vocab = build_vocabulary(support, min_sound_ids, candidates=vocab)
        history.append(
            {
                "round": round_idx,
                "support": {s: len(v) for s, v in support.items()},
                "vocab_size": len(new_vocab),
                "dropped": sorted(vocab - new_vocab),
            }
        )
        if new_vocab == vocab:
            return vocab, history
        vocab = new_vocab
    raise RuntimeError("Vocabulary fixed-point failed to converge within max_iterations.")


def finalize_label_state(
    label_state: str, species_codes: FrozenSet[str], vocabulary: Set[str]
) -> Tuple[str, FrozenSet[str]]:
    """Restrict a ``resolved_clean`` window's target to the retained
    vocabulary. If nothing survives, the window is fully excluded
    (``excluded_all_oov``) rather than silently coerced to an all-zero
    (no_bird-looking) vector. ``no_bird``/``unresolved_or_mixed`` pass
    through untouched (target stays empty)."""
    if label_state == RESOLVED_CLEAN:
        target = species_codes & vocabulary
        if target:
            return RESOLVED_CLEAN, target
        return EXCLUDED_ALL_OOV, frozenset()
    return label_state, frozenset()


# --------------------------------------------------------------------------
# End-to-end window classification pipeline
# --------------------------------------------------------------------------


def classify_all_windows(
    windows: Sequence[dict],
    id_by_sound: Dict[int, List[IdAnnotation]],
    min_sound_ids: int,
) -> Tuple[List[ClassifiedWindow], Set[str], List[dict]]:
    """Classify every window, build the vocabulary, and apply the OOV gate.
    Operates over *all* windows (canonical and overlapping) so overlapping
    windows are ready for train-split augmentation; the vocabulary itself is
    always computed from canonical windows only."""
    window_size_samples = infer_canonical_window_size(windows)

    pre_vocab: List[ClassifiedWindow] = []
    for w in windows:
        label_state, species_codes = classify_window(w, id_by_sound)
        pre_vocab.append(
            ClassifiedWindow(
                window_id=w["window_id"],
                sound_id=w["sound_id"],
                project=w["dataset"],
                sample_rate=w["sample_rate"],
                start=w["start"],
                end=w["end"],
                is_canonical=is_canonical(w, window_size_samples),
                label_state=label_state,
                species_codes=species_codes,
            )
        )

    vocabulary, vocab_history = build_vocabulary_fixed_point(pre_vocab, min_sound_ids)

    classified: List[ClassifiedWindow] = []
    for cw in pre_vocab:
        final_state, final_codes = finalize_label_state(cw.label_state, cw.species_codes, vocabulary)
        classified.append(
            ClassifiedWindow(
                window_id=cw.window_id,
                sound_id=cw.sound_id,
                project=cw.project,
                sample_rate=cw.sample_rate,
                start=cw.start,
                end=cw.end,
                is_canonical=cw.is_canonical,
                label_state=final_state,
                species_codes=final_codes,
            )
        )
    return classified, vocabulary, vocab_history


# --------------------------------------------------------------------------
# Per-sound_id statistics (canonical modeled windows only)
# --------------------------------------------------------------------------


def build_sound_stats(classified: Sequence[ClassifiedWindow]) -> Dict[int, SoundStats]:
    """Aggregate canonical modeled-window statistics per ``sound_id`` -- the
    indivisible split-assignment unit. Only sound_ids with >= 1 canonical
    modeled window enter the population (a sound_id whose canonical windows
    are entirely unresolved/mixed/OOV is outside the modeled population and
    is never assigned to a split)."""
    stats: Dict[int, SoundStats] = {}
    positive: Dict[int, Set[str]] = defaultdict(set)
    for cw in classified:
        if not cw.is_canonical or cw.label_state not in MODELED_STATES:
            continue
        if cw.sound_id not in stats:
            stats[cw.sound_id] = SoundStats(project=cw.project)
        ss = stats[cw.sound_id]
        if ss.project != cw.project:
            raise ValueError(f"sound_id {cw.sound_id} maps to multiple projects ({ss.project} vs {cw.project}).")
        ss.n_modeled += 1
        if cw.label_state == NO_BIRD:
            ss.n_nobird += 1
        else:
            ss.species_counts.update(cw.species_codes)
            positive[cw.sound_id].update(cw.species_codes)
    for sound_id, ss in stats.items():
        ss.positive_species = frozenset(positive.get(sound_id, frozenset()))
    return stats


# --------------------------------------------------------------------------
# Hard constraint: every retained species positive in every split
# --------------------------------------------------------------------------


def reserve_hard_constraints(
    sound_stats: Dict[int, SoundStats],
    vocab_sorted: Sequence[str],
) -> Tuple[Dict[int, str], List[Tuple[str, str]]]:
    """Deterministic greedy reservation: pin the minimum set of sound_ids
    needed so every species in ``vocab_sorted`` has >= 1 positive sound_id
    reserved to each split. Species are processed most-constrained-first
    (fewest candidate sound_ids, tie-broken alphabetically); within a
    species, an unreserved candidate is picked deterministically (fewest
    other retained species present, tie-broken by sound_id) to preserve
    flexibility for species processed later.

    Returns ``(reserved, infeasible)`` where ``infeasible`` lists
    ``(species, split)`` pairs that could not be satisfied -- callers must
    treat a non-empty ``infeasible`` as a fatal error.
    """
    candidates: Dict[str, List[int]] = {
        s: sorted(sid for sid, ss in sound_stats.items() if s in ss.positive_species) for s in vocab_sorted
    }
    order = sorted(vocab_sorted, key=lambda s: (len(candidates[s]), s))

    reserved: Dict[int, str] = {}
    infeasible: List[Tuple[str, str]] = []

    for s in order:
        cand = candidates[s]
        for split in SPLIT_NAMES:
            if any(reserved.get(sid) == split for sid in cand):
                continue  # already satisfied by an earlier species' reservation
            unreserved = [sid for sid in cand if sid not in reserved]
            if not unreserved:
                infeasible.append((s, split))
                continue
            unreserved.sort(key=lambda sid: (len(sound_stats[sid].positive_species), sid))
            reserved[unreserved[0]] = split

    return reserved, infeasible


# --------------------------------------------------------------------------
# Deterministic grouped stratified split (sound_id-only grouping)
# --------------------------------------------------------------------------


class _SplitOptimizerState:
    """Mutable per-split running counts used by the greedy/local-search optimizer."""

    def __init__(self, n_species: int):
        self.size = {k: 0 for k in SPLIT_NAMES}
        self.nobird = {k: 0 for k in SPLIT_NAMES}
        self.species = {k: [0.0] * n_species for k in SPLIT_NAMES}


def assign_groups(
    sound_stats: Dict[int, SoundStats],
    ratios: Dict[str, float],
    vocab_sorted: Sequence[str],
    seed: int = 42,
    num_restarts: int = 50,
    local_search_iters: int = 30,
) -> Tuple[Dict[int, str], dict]:
    """Deterministic reservation + repeated-greedy + local-search grouped
    stratified split. ``sound_id`` is the sole grouping unit (no site/event
    grouping). Phase 1 pins the minimum reservations required so every
    retained species has >= 1 positive sound_id in every split (hard
    constraint; raises if infeasible). Phase 2 assigns the remaining
    (unreserved) sound_ids to minimize a chi-square-like distance between
    per-split and global proportions for split size, no-bird count and
    per-species positive-window count. Project distribution is report-only
    and does not influence assignment.
    """
    if not sound_stats:
        raise ValueError("No sound_ids to assign; sound_stats is empty.")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    reserved, infeasible = reserve_hard_constraints(sound_stats, vocab_sorted)
    if infeasible:
        detail = ", ".join(f"{s} (missing {split})" for s, split in infeasible)
        raise ValueError(
            "Hard per-species split-presence constraint is infeasible for the following "
            f"species/split combinations: {detail}. Each retained species must have >= 1 "
            "positive sound_id available to reserve for every split; relax the vocabulary "
            "support gate or inspect these species' sound_id coverage."
        )

    sound_ids = sorted(sound_stats.keys())
    free_ids = [sid for sid in sound_ids if sid not in reserved]

    n_species = len(vocab_sorted)
    species_idx = {s: i for i, s in enumerate(vocab_sorted)}

    sound_species_vec: Dict[int, List[float]] = {}
    for sid, ss in sound_stats.items():
        vec = [0.0] * n_species
        for s, c in ss.species_counts.items():
            if s in species_idx:
                vec[species_idx[s]] = c
        sound_species_vec[sid] = vec

    total_modeled = sum(ss.n_modeled for ss in sound_stats.values())
    if total_modeled == 0:
        raise ValueError("Total modeled canonical window count is zero.")

    target_size = {k: ratios[k] * total_modeled for k in SPLIT_NAMES}

    global_nobird = sum(ss.n_nobird for ss in sound_stats.values())
    target_nobird = {k: (global_nobird / total_modeled) * target_size[k] for k in SPLIT_NAMES}

    global_species = [0.0] * n_species
    for vec in sound_species_vec.values():
        for i, v in enumerate(vec):
            global_species[i] += v
    target_species = {k: [g * ratios[k] for g in global_species] for k in SPLIT_NAMES}

    def split_cost(k: str, state: _SplitOptimizerState) -> float:
        size_term = ((state.size[k] - target_size[k]) / total_modeled) ** 2
        nobird_term = ((state.nobird[k] - target_nobird[k]) ** 2) / (target_nobird[k] + 1.0)
        species_term = 0.0
        if n_species:
            acc = 0.0
            for i in range(n_species):
                diff = state.species[k][i] - target_species[k][i]
                acc += (diff * diff) / (target_species[k][i] + 1.0)
            species_term = acc / n_species
        return size_term + nobird_term + species_term

    def total_cost(state: _SplitOptimizerState) -> float:
        return sum(split_cost(k, state) for k in SPLIT_NAMES)

    def add(state: _SplitOptimizerState, k: str, sid: int) -> None:
        ss = sound_stats[sid]
        state.size[k] += ss.n_modeled
        state.nobird[k] += ss.n_nobird
        for i, v in enumerate(sound_species_vec[sid]):
            state.species[k][i] += v

    def remove(state: _SplitOptimizerState, k: str, sid: int) -> None:
        ss = sound_stats[sid]
        state.size[k] -= ss.n_modeled
        state.nobird[k] -= ss.n_nobird
        for i, v in enumerate(sound_species_vec[sid]):
            state.species[k][i] -= v

    def build_initial_state() -> _SplitOptimizerState:
        state = _SplitOptimizerState(n_species)
        for sid, k in reserved.items():
            add(state, k, sid)
        return state

    best_assignment: Optional[Dict[int, str]] = None
    best_cost = math.inf
    best_seed: Optional[int] = None

    for restart in range(num_restarts):
        seed_r = seed + restart
        rng = random.Random(seed_r)
        order = free_ids.copy()
        rng.shuffle(order)

        state = build_initial_state()
        assignment: Dict[int, str] = dict(reserved)

        # Greedy construction: assign each free sound_id to the split
        # minimizing the marginal increase in that split's own cost term.
        for sid in order:
            best_k, best_delta = None, math.inf
            for k in SPLIT_NAMES:
                before = split_cost(k, state)
                add(state, k, sid)
                after = split_cost(k, state)
                remove(state, k, sid)
                delta = after - before
                if delta < best_delta - 1e-12:
                    best_delta, best_k = delta, k
            add(state, best_k, sid)
            assignment[sid] = best_k

        # Local search: single-sound_id moves (free ones only) that reduce
        # total cost, repeated in deterministic sound_id order.
        for _ in range(local_search_iters):
            improved = False
            for sid in free_ids:
                cur_k = assignment[sid]
                cost_cur_before = split_cost(cur_k, state)
                remove(state, cur_k, sid)
                cost_cur_after = split_cost(cur_k, state)
                best_k, best_net_change = cur_k, 0.0
                for k in SPLIT_NAMES:
                    if k == cur_k:
                        continue
                    cost_k_before = split_cost(k, state)
                    add(state, k, sid)
                    cost_k_after = split_cost(k, state)
                    remove(state, k, sid)
                    net_change = (cost_cur_after - cost_cur_before) + (cost_k_after - cost_k_before)
                    if net_change < best_net_change - 1e-9:
                        best_net_change, best_k = net_change, k
                add(state, best_k, sid)
                if best_k != cur_k:
                    assignment[sid] = best_k
                    improved = True
            if not improved:
                break

        final_cost = total_cost(state)
        if final_cost < best_cost - 1e-12:
            best_cost, best_assignment, best_seed = final_cost, dict(assignment), seed_r

    diagnostics = {
        "objective_score": best_cost,
        "selected_seed": best_seed,
        "num_restarts": num_restarts,
        "local_search_iters": local_search_iters,
        "n_reserved_sound_ids": len(reserved),
        "n_free_sound_ids": len(free_ids),
    }
    return best_assignment, diagnostics


# --------------------------------------------------------------------------
# Output row construction
# --------------------------------------------------------------------------


def target_codes_str(species_codes: FrozenSet[str]) -> str:
    return ";".join(sorted(species_codes))


def target_vector(species_codes: FrozenSet[str], vocab_sorted: Sequence[str]) -> List[int]:
    return [1 if s in species_codes else 0 for s in vocab_sorted]


def spectrogram_filename(sound_path: str, start_sample: int, end_sample: int) -> str:
    """Mirror prepare_dataset.py's spectrogram naming convention."""
    base = Path(sound_path).stem
    return f"{base}_{start_sample}_{end_sample}.npy"


FIELDNAMES = [
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
]


def build_rows(
    classified: Sequence[ClassifiedWindow],
    assignment: Dict[int, str],
    sounds_by_id: Dict[int, dict],
    vocab_sorted: Sequence[str],
) -> Dict[str, List[dict]]:
    """Build the per-output-file rows: ``train`` (canonical + augmented
    overlapping windows from train-assigned sound_ids), ``canonical_train``
    (canonical-only subset of ``train`` -- the audited base membership),
    ``val`` and ``test`` (canonical-only)."""
    rows: Dict[str, List[dict]] = {"train": [], "canonical_train": [], "val": [], "test": []}
    for cw in classified:
        if cw.label_state not in MODELED_STATES:
            continue
        split = assignment.get(cw.sound_id)
        if split is None:
            continue
        if split != "train" and not cw.is_canonical:
            continue  # val/test are canonical-only

        sound = sounds_by_id.get(cw.sound_id, {})
        sound_path = sound.get("file_name_path", str(cw.sound_id))
        row = {
            "window_id": cw.window_id,
            "dataset": cw.project,
            "sample_rate": cw.sample_rate,
            "sound_id": cw.sound_id,
            "start": cw.start,
            "end": cw.end,
            "project": cw.project,
            "is_canonical": int(cw.is_canonical),
            "label_state": cw.label_state,
            "spec_name": spectrogram_filename(sound_path, cw.start, cw.end),
            "sound_filename": Path(sound_path).name,
            "target_codes": target_codes_str(cw.species_codes),
            "target_vector": json.dumps(target_vector(cw.species_codes, vocab_sorted)),
        }
        if split == "train":
            rows["train"].append(row)
            if cw.is_canonical:
                rows["canonical_train"].append(row)
        else:
            rows[split].append(row)

    for k in rows:
        rows[k].sort(key=lambda r: r["window_id"])
    return rows


def write_split_csv(rows: List[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Validation gates (fail fast)
# --------------------------------------------------------------------------


def validate_no_leakage(assignment: Dict[int, str]) -> None:
    """Every sound_id maps to exactly one split (trivially true for a dict,
    but re-derive split->sound_id sets and check pairwise disjointness as a
    defensive, explicit gate)."""
    by_split: Dict[str, Set[int]] = defaultdict(set)
    for sound_id, split in assignment.items():
        by_split[split].add(sound_id)
    splits = list(by_split.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = by_split[splits[i]] & by_split[splits[j]]
            if overlap:
                raise AssertionError(f"sound_id leakage between {splits[i]} and {splits[j]}: {sorted(overlap)}")


def validate_rows(rows: Dict[str, List[dict]], vocab_sorted: Sequence[str]) -> None:
    """Fail if any excluded label_state leaked into output, if an all-zero
    included row is not a true no_bird window, or if val/test/canonical_train
    contain a non-canonical row."""
    vocab_set = set(vocab_sorted)
    for split, split_rows in rows.items():
        for row in split_rows:
            if row["label_state"] not in MODELED_STATES:
                raise AssertionError(f"Row with excluded label_state {row['label_state']!r} leaked into {split}")
            if split != "train" and not row["is_canonical"]:
                raise AssertionError(f"Non-canonical row leaked into canonical-only output {split}: {row}")
            vec = json.loads(row["target_vector"])
            is_zero = not any(vec)
            if is_zero and row["label_state"] != NO_BIRD:
                raise AssertionError(f"All-zero target row in {split} is not true no_bird: {row}")
            if row["label_state"] == NO_BIRD and not is_zero:
                raise AssertionError(f"no_bird row has non-zero target in {split}: {row}")
            codes = set(c for c in row["target_codes"].split(";") if c)
            if codes and not codes.issubset(vocab_set):
                raise AssertionError(f"Row in {split} contains out-of-vocabulary codes: {row}")


def validate_group_disjoint_from_rows(rows: Dict[str, List[dict]]) -> None:
    """Cross-check: no sound_id appears in more than one *split* (``train``,
    ``val``, ``test`` -- ``canonical_train`` is a subset of ``train`` by
    construction and is intentionally excluded from this pairwise check)."""
    split_keys = [k for k in rows if k != "canonical_train"]
    by_split: Dict[str, Set[int]] = {k: {row["sound_id"] for row in rows[k]} for k in split_keys}
    for i in range(len(split_keys)):
        for j in range(i + 1, len(split_keys)):
            overlap = by_split[split_keys[i]] & by_split[split_keys[j]]
            if overlap:
                raise AssertionError(
                    f"sound_id leakage in output rows between {split_keys[i]} and {split_keys[j]}: {sorted(overlap)}"
                )


def validate_train_augmentation_source(rows: Dict[str, List[dict]]) -> None:
    """Every augmented (non-canonical) train row's sound_id must also appear
    in canonical_train -- i.e. augmentation only ever adds overlapping
    windows from sound_ids already in the canonical train base, never from a
    val/test sound_id."""
    canonical_train_sounds = {r["sound_id"] for r in rows["canonical_train"]}
    for row in rows["train"]:
        if not row["is_canonical"] and row["sound_id"] not in canonical_train_sounds:
            raise AssertionError(f"Augmented train row from a sound_id absent from canonical_train: {row}")


def validate_species_present_in_all_splits(rows: Dict[str, List[dict]], vocab_sorted: Sequence[str]) -> None:
    """Hard gate: every retained species must have >= 1 positive canonical
    window in canonical_train, val and test."""
    missing: List[Tuple[str, str]] = []
    for split in ("canonical_train", "val", "test"):
        present = set()
        for r in rows[split]:
            present.update(c for c in r["target_codes"].split(";") if c)
        for s in vocab_sorted:
            if s not in present:
                missing.append((s, split))
    if missing:
        detail = ", ".join(f"{s} (missing in {split})" for s, split in missing)
        raise AssertionError(f"Species missing from a required split after assignment: {detail}")


# --------------------------------------------------------------------------
# Reporting: species distribution & manifest
# --------------------------------------------------------------------------


def species_distribution_table(rows: Dict[str, List[dict]], vocab_sorted: Sequence[str]) -> List[dict]:
    """Per-split (canonical_train/val/test), per-label (species + no_bird)
    counts, proportions and deviation from the pooled canonical-audited
    proportion. This is the audited 70/15/15 distribution -- it uses
    ``canonical_train``, never the augmented ``train``."""
    audited = {"train": rows["canonical_train"], "val": rows["val"], "test": rows["test"]}
    all_rows = [r for split_rows in audited.values() for r in split_rows]
    total_all = len(all_rows)

    def counts_for(split_rows: List[dict]) -> Dict[str, int]:
        c: Counter = Counter()
        for r in split_rows:
            if r["label_state"] == NO_BIRD:
                c[NO_BIRD] += 1
            else:
                for code in r["target_codes"].split(";"):
                    if code:
                        c[code] += 1
        return c

    global_counts = counts_for(all_rows)
    table = []
    labels = [NO_BIRD] + list(vocab_sorted)
    for label in labels:
        global_count = global_counts.get(label, 0)
        global_frac = global_count / total_all if total_all else 0.0
        entry = {"label": label, "global_count": global_count, "global_frac": global_frac}
        for split in SPLIT_NAMES:
            split_rows = audited[split]
            split_total = len(split_rows)
            split_count = counts_for(split_rows).get(label, 0)
            split_frac = split_count / split_total if split_total else 0.0
            entry[f"{split}_count"] = split_count
            entry[f"{split}_frac"] = split_frac
            entry[f"{split}_frac_deviation"] = split_frac - global_frac
        table.append(entry)
    return table


def write_species_distribution_csv(table: List[dict], path: Path) -> None:
    if not table:
        return
    fieldnames = list(table[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible species-classification train/val/test splits."
    )
    parser.add_argument(
        "--windows-mapping",
        default="data/windows_mapping_4.0overlap_segmented_v4.json",
        help="Path to the windows_mapping JSON (list of window dicts).",
    )
    parser.add_argument(
        "--annotations-identification",
        default="data/annotations_identification.json",
        help="Path to the identification-level COCO annotations JSON.",
    )
    parser.add_argument(
        "--annotations-species",
        default="data/annotations_species.json",
        help="Path to the species-level COCO annotations JSON.",
    )
    parser.add_argument(
        "--species-csv",
        default="data/species.csv",
        help="Path to the species reference CSV (code, species, ...).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits_species_v1",
        help="Directory to write train/val/test CSVs and reporting artifacts.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(    "--num-restarts", type=int, default=50, help="Seeded greedy restarts for the group split.")
    parser.add_argument(
        "--local-search-iters", type=int, default=30, help="Max local-improvement passes per restart."
    )
    parser.add_argument(
        "--min-sound-ids",
        type=int,
        default=7,
        help="Minimum distinct canonical-window sound_id support for a species to enter the vocabulary.",
    )
    parser.add_argument(
        "--geometry-decimals",
        type=int,
        default=6,
        help="Decimal places used to build the annotation geometry identity key.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    windows_path = Path(args.windows_mapping)
    id_path = Path(args.annotations_identification)
    sp_path = Path(args.annotations_species)
    species_csv_path = Path(args.species_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading windows mapping from {windows_path}")
    windows = load_json(windows_path)
    print(f"Loading identification annotations from {id_path}")
    id_data = load_json(id_path)
    print(f"Loading species annotations from {sp_path}")
    sp_data = load_json(sp_path)
    print(f"Loading species reference CSV from {species_csv_path}")
    species_ref = load_species_csv(species_csv_path)

    sounds_by_id = {s["id"]: s for s in id_data["sounds"]}

    taxonomy = build_taxonomy_crosswalk(species_ref)
    species_lookup = build_species_lookup(sp_data["annotations"], args.geometry_decimals)
    id_by_sound = build_identification_records(id_data["annotations"], species_lookup, taxonomy, args.geometry_decimals)

    classified, vocabulary, vocab_history = classify_all_windows(windows, id_by_sound, args.min_sound_ids)
    vocab_sorted = sorted(vocabulary)
    unknown_vocab = vocabulary - taxonomy.known_codes
    if unknown_vocab:
        raise ValueError(f"Vocabulary contains codes absent from species.csv: {sorted(unknown_vocab)}")

    state_counts_all = Counter(cw.label_state for cw in classified)
    state_counts_canonical = Counter(cw.label_state for cw in classified if cw.is_canonical)
    print(f"Label state counts (all windows): {dict(state_counts_all)}")
    print(f"Label state counts (canonical only): {dict(state_counts_canonical)}")
    print(f"Vocabulary size: {len(vocab_sorted)} species -> {vocab_sorted}")

    sound_stats = build_sound_stats(classified)

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    assignment, diagnostics = assign_groups(
        sound_stats,
        ratios,
        vocab_sorted,
        seed=args.seed,
        num_restarts=args.num_restarts,
        local_search_iters=args.local_search_iters,
    )
    validate_no_leakage(assignment)

    rows = build_rows(classified, assignment, sounds_by_id, vocab_sorted)
    validate_rows(rows, vocab_sorted)
    validate_group_disjoint_from_rows(rows)
    validate_train_augmentation_source(rows)
    validate_species_present_in_all_splits(rows, vocab_sorted)

    write_split_csv(rows["train"], output_dir / "train_split.csv")
    write_split_csv(rows["canonical_train"], output_dir / "canonical_train_split.csv")
    write_split_csv(rows["val"], output_dir / "val_split.csv")
    write_split_csv(rows["test"], output_dir / "test_split.csv")

    class_list = [
        {
            "index": i,
            "code": s,
            "species": taxonomy.species_name_of.get(s),
        }
        for i, s in enumerate(vocab_sorted)
    ]
    with open(output_dir / "class_list.json", "w") as f:
        json.dump(class_list, f, indent=2)

    dist_table = species_distribution_table(rows, vocab_sorted)
    write_species_distribution_csv(dist_table, output_dir / "species_distribution.csv")

    total_modeled_canonical = sum(ss.n_modeled for ss in sound_stats.values())
    global_nobird = sum(ss.n_nobird for ss in sound_stats.values())

    def project_counts(split_rows: List[dict]) -> Dict[str, int]:
        return dict(Counter(r["project"] for r in split_rows))

    manifest = {
        "config": {
            "windows_mapping": str(windows_path),
            "annotations_identification": str(id_path),
            "annotations_species": str(sp_path),
            "species_csv": str(species_csv_path),
            "ratios": ratios,
            "seed": args.seed,
            "num_restarts": args.num_restarts,
            "local_search_iters": args.local_search_iters,
            "min_sound_ids": args.min_sound_ids,
            "geometry_decimals": args.geometry_decimals,
            "grouping_unit": "sound_id",
        },
        "input_hashes": {
            "windows_mapping": sha256_file(windows_path),
            "annotations_identification": sha256_file(id_path),
            "annotations_species": sha256_file(sp_path),
            "species_csv": sha256_file(species_csv_path),
        },
        "taxonomy_crosswalk": {
            "placeholder_codes_excluded": sorted(taxonomy.placeholder_codes),
            "duplicate_code_merge_groups": [
                {"canonical": canon, "raw_codes": codes} for canon, codes in taxonomy.duplicate_groups
            ],
            "n_known_codes": len(taxonomy.known_codes),
        },
        "label_state_counts_all_windows": dict(state_counts_all),
        "label_state_counts_canonical_windows": dict(state_counts_canonical),
        "vocabulary_size": len(vocab_sorted),
        "vocabulary": vocab_sorted,
        "vocabulary_fixed_point_history": vocab_history,
        "vocabulary_gate_rationale": (
            f"Requested gate min_sound_ids={args.min_sound_ids} (distinct canonical-window sound_ids per "
            f"species, after taxonomy canonicalization) yielded {len(vocab_sorted)} species after fixed-point "
            "re-evaluation. Windows mixing a retained species with an excluded (rare or placeholder-derived) "
            "one are kept, modeled with the retained subset only; only windows where every overlapping species "
            "is excluded are dropped entirely (excluded_all_oov)."
        ),
        "canonical_window_size_samples": infer_canonical_window_size(windows),
        "total_modeled_canonical_windows": total_modeled_canonical,
        "no_bird_prevalence_canonical": global_nobird / total_modeled_canonical if total_modeled_canonical else None,
        "canonical_base_sizes": {
            "train": len(rows["canonical_train"]),
            "val": len(rows["val"]),
            "test": len(rows["test"]),
        },
        "augmented_train_size": len(rows["train"]),
        "augmented_train_extra_windows": len(rows["train"]) - len(rows["canonical_train"]),
        "split_no_bird_prevalence_canonical": {
            split: (
                sum(1 for r in rows[key] if r["label_state"] == NO_BIRD) / max(1, len(rows[key]))
            )
            for split, key in (("train", "canonical_train"), ("val", "val"), ("test", "test"))
        },
        "project_distribution_canonical": {
            "train": project_counts(rows["canonical_train"]),
            "val": project_counts(rows["val"]),
            "test": project_counts(rows["test"]),
        },
        "project_distribution_augmented_train": project_counts(rows["train"]),
        "n_sound_ids_per_split": {
            split: len({sid for sid, sp in assignment.items() if sp == split}) for split in SPLIT_NAMES
        },
        "n_sound_ids_total": len(assignment),
        "sound_id_disjointness_verified": True,
        "species_present_in_all_splits_verified": True,
        "species_distribution_deviation": {
            entry["label"]: {split: entry[f"{split}_frac_deviation"] for split in SPLIT_NAMES} for entry in dist_table
        },
        "objective_score": diagnostics["objective_score"],
        "selected_restart_seed": diagnostics["selected_seed"],
        "n_reserved_sound_ids": diagnostics["n_reserved_sound_ids"],
        "n_free_sound_ids": diagnostics["n_free_sound_ids"],
        "validation_gates_passed": [
            "no_leakage",
            "rows_label_state_and_canonicality",
            "group_disjoint_from_rows",
            "train_augmentation_source",
            "species_present_in_all_splits",
        ],
    }
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Wrote splits to {output_dir}")
    print(f"Canonical base sizes: {manifest['canonical_base_sizes']}")
    print(f"Augmented train size: {manifest['augmented_train_size']}")
    print(f"No-bird prevalence (canonical, global): {manifest['no_bird_prevalence_canonical']:.4f}")
    print(f"Objective score: {manifest['objective_score']:.6f} (seed {manifest['selected_restart_seed']})")


if __name__ == "__main__":
    main()
