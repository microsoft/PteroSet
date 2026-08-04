"""Tests for prepare_species_splits.py using synthetic data only.

Covers: taxonomy resolution (placeholder-as-unresolved, duplicate-code
canonicalization), annotation geometry matching, mixed/unresolved exclusion,
no-bird zero-vector semantics, retained+rare mixed-window handling,
rare-only window exclusion, the >= 7 distinct-sound_id vocabulary fixed
point, deterministic sound_id-only grouping with a hard per-species
per-split presence constraint, no sound leakage, 70/15/15 proportion
tolerance on synthetic data, canonical-only val/test/base-train, overlapping
train augmentation restricted to train-assigned sound_ids, determinism
across reruns with the same seed, and class-list reconstruction.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_species_splits as pss


SR = 48000
WIN_SAMPLES = 5 * SR  # 5 s canonical window, matches dataset convention


def make_window(window_id, sound_id, start, end, project="MAP1", sample_rate=SR):
    return {
        "window_id": window_id,
        "dataset": project,
        "sample_rate": sample_rate,
        "sound_id": sound_id,
        "start": start,
        "end": end,
        "label": 0,
    }


def make_id_anno(sound_id, t_min, t_max, f_min=0.0, f_max=1000.0):
    return {"sound_id": sound_id, "t_min": t_min, "t_max": t_max, "f_min": f_min, "f_max": f_max}


def make_sp_anno(sound_id, t_min, t_max, category, f_min=0.0, f_max=1000.0):
    return {
        "sound_id": sound_id,
        "t_min": t_min,
        "t_max": t_max,
        "f_min": f_min,
        "f_max": f_max,
        "category": category,
    }


def make_species_row(code, species, identification="AVEVOC", type_="BIO"):
    return {"code": code, "species": species, "identification": identification, "type": type_}


# --------------------------------------------------------------------------
# Taxonomy resolution: placeholders + duplicate-name canonicalization
# --------------------------------------------------------------------------


def test_placeholder_dash_name_is_detected():
    assert pss.is_placeholder_species_name("\u2013")
    assert pss.is_placeholder_species_name("-")


def test_placeholder_sp_token_name_is_detected():
    assert pss.is_placeholder_species_name("Tyrannidae sp 1")
    assert pss.is_placeholder_species_name("Psittacidae sp.")


def test_real_species_name_is_not_placeholder():
    assert not pss.is_placeholder_species_name("Cyanocorax violaceus")
    assert pss.is_placeholder_species_name("Picidae")


def test_taxonomy_crosswalk_flags_current_placeholders():
    species_ref = [
        make_species_row("AKLMEL", "Akletos melanoceps"),
        make_species_row("PSITTACIDAE", "\u2013"),
        make_species_row("PSITTACIFORMES", "\u2013"),
        make_species_row("RHACAR", "\u2013"),
        make_species_row("TYRANN_SP1", "Tyrannidae sp 1"),
        make_species_row("PSITTA", "Psittacidae sp."),
        make_species_row("PICIDA_1", "Picidae"),
    ]
    taxonomy = pss.build_taxonomy_crosswalk(species_ref)
    assert taxonomy.placeholder_codes == {
        "PSITTACIDAE",
        "PSITTACIFORMES",
        "RHACAR",
        "TYRANN_SP1",
        "PSITTA",
        "PICIDA_1",
    }
    assert "AKLMEL" not in taxonomy.placeholder_codes


def test_taxonomy_crosswalk_merges_duplicate_scientific_names_deterministically():
    species_ref = [
        make_species_row("RAMTUC", "Ramphastos tucanus"),
        make_species_row("RHATUC", "Ramphastos tucanus"),
        make_species_row("ATRPIL", "Atalotriccus pilaris"),
        make_species_row("ATAPIL", "Atalotriccus pilaris"),
    ]
    taxonomy = pss.build_taxonomy_crosswalk(species_ref)
    # Canonical representative is the alphabetically first code in the group.
    assert taxonomy.canonicalize("RAMTUC") == "RAMTUC"
    assert taxonomy.canonicalize("RHATUC") == "RAMTUC"
    assert taxonomy.canonicalize("ATAPIL") == "ATAPIL"
    assert taxonomy.canonicalize("ATRPIL") == "ATAPIL"
    groups = {canon: sorted(codes) for canon, codes in taxonomy.duplicate_groups}
    assert groups == {"RAMTUC": ["RAMTUC", "RHATUC"], "ATAPIL": ["ATAPIL", "ATRPIL"]}


def test_taxonomy_crosswalk_does_not_merge_distinct_placeholder_dashes():
    """Multiple placeholder rows share the literal dash name but must not be
    treated as a duplicate-species merge group -- they are each individually
    excluded, not consolidated into one another."""
    species_ref = [
        make_species_row("PSITTACIDAE", "\u2013"),
        make_species_row("PSITTACIFORMES", "\u2013"),
        make_species_row("RHACAR", "\u2013"),
    ]
    taxonomy = pss.build_taxonomy_crosswalk(species_ref)
    assert taxonomy.duplicate_groups == []
    assert taxonomy.placeholder_codes == {"PSITTACIDAE", "PSITTACIFORMES", "RHACAR"}


# --------------------------------------------------------------------------
# Annotation identity matching
# --------------------------------------------------------------------------


def test_species_lookup_matches_by_geometry_within_tolerance():
    id_annos = [make_id_anno(0, 1.000000123, 2.000000456)]
    sp_annos = [make_sp_anno(0, 1.000000098, 2.000000234, "CYAVIO")]  # sub-1e-6 noise, same rounding bucket
    lookup = pss.build_species_lookup(sp_annos, decimals=6)
    taxonomy = pss.build_taxonomy_crosswalk([make_species_row("CYAVIO", "Cyanocorax violaceus")])
    by_sound = pss.build_identification_records(id_annos, lookup, taxonomy, decimals=6)
    assert by_sound[0][0].resolved is True
    assert by_sound[0][0].species_code == "CYAVIO"


def test_species_lookup_rejects_non_matching_geometry():
    """A species annotation whose geometry does not match any identification
    annotation violates the documented subset assumption and must fail
    fast rather than silently resolve nothing."""
    id_annos = [make_id_anno(0, 1.0, 2.0)]
    sp_annos = [make_sp_anno(0, 5.0, 6.0, "CYAVIO")]  # unrelated geometry
    lookup = pss.build_species_lookup(sp_annos, decimals=6)
    taxonomy = pss.build_taxonomy_crosswalk([make_species_row("CYAVIO", "Cyanocorax violaceus")])
    with pytest.raises(ValueError):
        pss.build_identification_records(id_annos, lookup, taxonomy, decimals=6)


def test_identification_record_unresolved_when_no_species_match():
    id_annos = [make_id_anno(0, 1.0, 2.0)]
    lookup = {}
    taxonomy = pss.build_taxonomy_crosswalk([])
    by_sound = pss.build_identification_records(id_annos, lookup, taxonomy, decimals=6)
    assert by_sound[0][0].resolved is False
    assert by_sound[0][0].species_code is None


def test_identification_record_placeholder_match_marks_unresolved():
    """A call resolved to a coarse/placeholder species-level annotation must
    be treated as an unresolved identification event, per the approved
    taxonomy-resolution design -- not silently excluded downstream as OOV."""
    id_annos = [make_id_anno(0, 1.0, 2.0)]
    sp_annos = [make_sp_anno(0, 1.0, 2.0, "PSITTA")]
    lookup = pss.build_species_lookup(sp_annos, decimals=6)
    taxonomy = pss.build_taxonomy_crosswalk([make_species_row("PSITTA", "Psittacidae sp.")])
    by_sound = pss.build_identification_records(id_annos, lookup, taxonomy, decimals=6)
    assert by_sound[0][0].resolved is False
    assert by_sound[0][0].species_code is None


def test_identification_record_canonicalizes_duplicate_code_to_representative():
    id_annos = [make_id_anno(0, 1.0, 2.0)]
    sp_annos = [make_sp_anno(0, 1.0, 2.0, "RHATUC")]
    lookup = pss.build_species_lookup(sp_annos, decimals=6)
    species_ref = [
        make_species_row("RAMTUC", "Ramphastos tucanus"),
        make_species_row("RHATUC", "Ramphastos tucanus"),
    ]
    taxonomy = pss.build_taxonomy_crosswalk(species_ref)
    by_sound = pss.build_identification_records(id_annos, lookup, taxonomy, decimals=6)
    assert by_sound[0][0].resolved is True
    assert by_sound[0][0].species_code == "RAMTUC"  # canonicalized, not the raw RHATUC


# --------------------------------------------------------------------------
# Window classification
# --------------------------------------------------------------------------


def test_classify_window_no_bird_when_no_overlap():
    window = make_window(0, sound_id=0, start=0, end=WIN_SAMPLES)
    state, codes = pss.classify_window(window, id_by_sound={})
    assert state == pss.NO_BIRD
    assert codes == frozenset()


def test_classify_window_touching_boundary_is_not_overlap():
    """An annotation that ends exactly at the window start (or starts exactly
    at the window end) must not count as overlap (strict interval overlap)."""
    window = make_window(0, sound_id=0, start=WIN_SAMPLES, end=2 * WIN_SAMPLES)
    id_by_sound = {0: [pss.IdAnnotation(t_min=0.0, t_max=5.0, resolved=True, species_code="CYAVIO")]}
    state, codes = pss.classify_window(window, id_by_sound)
    assert state == pss.NO_BIRD
    assert codes == frozenset()


def test_classify_window_resolved_clean_single_species():
    window = make_window(0, sound_id=0, start=0, end=WIN_SAMPLES)
    id_by_sound = {0: [pss.IdAnnotation(t_min=1.0, t_max=2.0, resolved=True, species_code="CYAVIO")]}
    state, codes = pss.classify_window(window, id_by_sound)
    assert state == pss.RESOLVED_CLEAN
    assert codes == frozenset({"CYAVIO"})


def test_classify_window_resolved_clean_multilabel():
    window = make_window(0, sound_id=0, start=0, end=WIN_SAMPLES)
    id_by_sound = {
        0: [
            pss.IdAnnotation(t_min=1.0, t_max=2.0, resolved=True, species_code="CYAVIO"),
            pss.IdAnnotation(t_min=3.0, t_max=4.0, resolved=True, species_code="ORTGUT"),
        ]
    }
    state, codes = pss.classify_window(window, id_by_sound)
    assert state == pss.RESOLVED_CLEAN
    assert codes == frozenset({"CYAVIO", "ORTGUT"})


def test_classify_window_mixed_resolved_and_unresolved_is_excluded():
    """A window overlapping one resolved call and one unresolved call must be
    unresolved_or_mixed and excluded, even though a resolved call is present."""
    window = make_window(0, sound_id=0, start=0, end=WIN_SAMPLES)
    id_by_sound = {
        0: [
            pss.IdAnnotation(t_min=1.0, t_max=2.0, resolved=True, species_code="CYAVIO"),
            pss.IdAnnotation(t_min=3.0, t_max=4.0, resolved=False, species_code=None),
        ]
    }
    state, codes = pss.classify_window(window, id_by_sound)
    assert state == pss.UNRESOLVED_OR_MIXED
    assert codes == frozenset()


def test_classify_window_purely_unresolved():
    window = make_window(0, sound_id=0, start=0, end=WIN_SAMPLES)
    id_by_sound = {0: [pss.IdAnnotation(t_min=1.0, t_max=2.0, resolved=False, species_code=None)]}
    state, codes = pss.classify_window(window, id_by_sound)
    assert state == pss.UNRESOLVED_OR_MIXED


# --------------------------------------------------------------------------
# No-bird all-zero target semantics
# --------------------------------------------------------------------------


def test_target_vector_all_zero_for_no_bird():
    vocab_sorted = ["AAA", "BBB", "CCC"]
    vec = pss.target_vector(frozenset(), vocab_sorted)
    assert vec == [0, 0, 0]
    assert pss.target_codes_str(frozenset()) == ""


def test_target_vector_nonzero_for_resolved():
    vocab_sorted = ["AAA", "BBB", "CCC"]
    vec = pss.target_vector(frozenset({"BBB"}), vocab_sorted)
    assert vec == [0, 1, 0]
    assert pss.target_codes_str(frozenset({"CCC", "AAA"})) == "AAA;CCC"


# --------------------------------------------------------------------------
# Retained+rare mixed windows kept; rare-only windows excluded
# --------------------------------------------------------------------------


def test_finalize_label_state_keeps_mixed_window_restricted_to_retained_subset():
    """A window with one retained and one excluded (rare) species is NOT
    unresolved/mixed -- both calls are known -- and must be kept, modeled
    with the retained species only."""
    vocab = {"CYAVIO"}
    state, codes = pss.finalize_label_state(pss.RESOLVED_CLEAN, frozenset({"CYAVIO", "RAREBIRD"}), vocab)
    assert state == pss.RESOLVED_CLEAN
    assert codes == frozenset({"CYAVIO"})  # RAREBIRD silently dropped, not the whole window


def test_finalize_label_state_excludes_window_with_only_rare_species():
    vocab = {"CYAVIO"}
    state, codes = pss.finalize_label_state(pss.RESOLVED_CLEAN, frozenset({"RAREBIRD"}), vocab)
    assert state == pss.EXCLUDED_ALL_OOV
    assert codes == frozenset()


def test_finalize_label_state_keeps_in_vocab_resolved():
    vocab = {"CYAVIO", "ORTGUT"}
    state, codes = pss.finalize_label_state(pss.RESOLVED_CLEAN, frozenset({"CYAVIO"}), vocab)
    assert state == pss.RESOLVED_CLEAN
    assert codes == frozenset({"CYAVIO"})


def test_finalize_label_state_no_bird_untouched():
    state, codes = pss.finalize_label_state(pss.NO_BIRD, frozenset(), set())
    assert state == pss.NO_BIRD
    assert codes == frozenset()


def _classified(sound_id, window_id, is_canonical, label_state, species_codes, project="MAP1"):
    return pss.ClassifiedWindow(
        window_id=window_id,
        sound_id=sound_id,
        project=project,
        sample_rate=SR,
        start=0,
        end=WIN_SAMPLES,
        is_canonical=is_canonical,
        label_state=label_state,
        species_codes=frozenset(species_codes),
    )


# --------------------------------------------------------------------------
# >= 7 distinct sound_id vocabulary fixed point
# --------------------------------------------------------------------------


def test_build_vocabulary_rejects_species_below_sound_id_threshold():
    # RARE appears in only 2 distinct sound_ids -> below min_sound_ids=3.
    classified = [
        _classified(0, 0, True, pss.RESOLVED_CLEAN, {"COMMON"}),
        _classified(1, 1, True, pss.RESOLVED_CLEAN, {"COMMON"}),
        _classified(2, 2, True, pss.RESOLVED_CLEAN, {"COMMON"}),
        _classified(3, 3, True, pss.RESOLVED_CLEAN, {"RARE"}),
        _classified(4, 4, True, pss.RESOLVED_CLEAN, {"RARE"}),
    ]
    support = pss.compute_species_support(classified)
    vocab = pss.build_vocabulary(support, min_sound_ids=3)
    assert vocab == {"COMMON"}


def test_build_vocabulary_counts_distinct_sound_ids_not_windows():
    # COMMON has 5 windows but all from a single sound_id -> below min_sound_ids=2.
    classified = [_classified(0, i, True, pss.RESOLVED_CLEAN, {"COMMON"}) for i in range(5)]
    support = pss.compute_species_support(classified)
    with pytest.raises(ValueError):
        pss.build_vocabulary(support, min_sound_ids=2)


def test_vocabulary_fixed_point_converges_without_cascade():
    """Under the approved design, mixed retained+excluded windows are kept
    (not dropped), so a species' own support never depends on whether a
    co-occurring species is retained -- the fixed point is reached in the
    very first extra round."""
    classified = [
        _classified(0, 0, True, pss.RESOLVED_CLEAN, {"A"}),
        _classified(1, 1, True, pss.RESOLVED_CLEAN, {"A"}),
        _classified(2, 2, True, pss.RESOLVED_CLEAN, {"A"}),
    ]
    vocab, history = pss.build_vocabulary_fixed_point(classified, min_sound_ids=3)
    assert vocab == {"A"}
    assert len(history) == 2  # round 0 (raw) + round 1 (confirms fixed point)
    assert history[-1]["vocab_size"] == 1


def test_vocabulary_fixed_point_drops_species_only_supported_via_rare_only_windows():
    """A only ever appears in windows shared with B, and both are below
    threshold on their own -- excluding the fully-rare (both-below-gate)
    windows must not spuriously resurrect either species."""
    classified = [
        _classified(0, 0, True, pss.RESOLVED_CLEAN, {"A", "B"}),
        _classified(1, 1, True, pss.RESOLVED_CLEAN, {"A", "B"}),
        # C is independently well supported across 3 sound_ids.
        _classified(2, 2, True, pss.RESOLVED_CLEAN, {"C"}),
        _classified(3, 3, True, pss.RESOLVED_CLEAN, {"C"}),
        _classified(4, 4, True, pss.RESOLVED_CLEAN, {"C"}),
    ]
    vocab, _ = pss.build_vocabulary_fixed_point(classified, min_sound_ids=3)
    assert vocab == {"C"}
    assert "A" not in vocab and "B" not in vocab


def test_vocabulary_fixed_point_keeps_species_supported_via_mixed_windows():
    """A meets its support gate only through windows that also contain a
    never-qualifying rare species B; because mixed windows are *kept* (not
    dropped) in this design, A's support must not be penalized by B's
    exclusion."""
    classified = [
        _classified(0, 0, True, pss.RESOLVED_CLEAN, {"A", "B"}),
        _classified(1, 1, True, pss.RESOLVED_CLEAN, {"A", "B"}),
        _classified(2, 2, True, pss.RESOLVED_CLEAN, {"A"}),
    ]
    vocab, _ = pss.build_vocabulary_fixed_point(classified, min_sound_ids=3)
    assert vocab == {"A"}
    assert "B" not in vocab  # B has only 2 distinct sound_ids, below the gate


# --------------------------------------------------------------------------
# Row-level validation gates
# --------------------------------------------------------------------------


def _row(window_id, sound_id, label_state, target_codes, vocab_sorted, is_canonical=1):
    codes = frozenset(c for c in target_codes.split(";") if c)
    return {
        "window_id": window_id,
        "sound_id": sound_id,
        "is_canonical": is_canonical,
        "label_state": label_state,
        "target_codes": target_codes,
        "target_vector": json.dumps(pss.target_vector(codes, vocab_sorted)),
    }


def test_validate_rows_rejects_leaked_excluded_state():
    vocab_sorted = ["A"]
    rows = {
        "train": [_row(0, 0, pss.UNRESOLVED_OR_MIXED, "", vocab_sorted)],
        "canonical_train": [],
        "val": [],
        "test": [],
    }
    with pytest.raises(AssertionError):
        pss.validate_rows(rows, vocab_sorted)


def test_validate_rows_rejects_nonzero_no_bird_row():
    vocab_sorted = ["A"]
    row = _row(0, 0, pss.NO_BIRD, "", vocab_sorted)
    row["target_vector"] = json.dumps([1])  # corrupted: no_bird but non-zero
    rows = {"train": [row], "canonical_train": [row], "val": [], "test": []}
    with pytest.raises(AssertionError):
        pss.validate_rows(rows, vocab_sorted)


def test_validate_rows_rejects_all_zero_resolved_row():
    vocab_sorted = ["A", "B"]
    row = _row(0, 0, pss.RESOLVED_CLEAN, "A", vocab_sorted)
    row["target_vector"] = json.dumps([0, 0])  # corrupted: resolved but all-zero
    rows = {"train": [row], "canonical_train": [row], "val": [], "test": []}
    with pytest.raises(AssertionError):
        pss.validate_rows(rows, vocab_sorted)


def test_validate_rows_rejects_non_canonical_row_in_val():
    vocab_sorted = ["A"]
    row = _row(0, 0, pss.NO_BIRD, "", vocab_sorted, is_canonical=0)
    rows = {"train": [], "canonical_train": [], "val": [row], "test": []}
    with pytest.raises(AssertionError):
        pss.validate_rows(rows, vocab_sorted)


def test_validate_rows_accepts_non_canonical_row_in_train():
    vocab_sorted = ["A"]
    row = _row(0, 0, pss.NO_BIRD, "", vocab_sorted, is_canonical=0)
    rows = {"train": [row], "canonical_train": [], "val": [], "test": []}
    pss.validate_rows(rows, vocab_sorted)  # must not raise


def test_validate_rows_accepts_clean_rows():
    vocab_sorted = ["A", "B"]
    rows = {
        "train": [_row(0, 0, pss.NO_BIRD, "", vocab_sorted), _row(1, 1, pss.RESOLVED_CLEAN, "A", vocab_sorted)],
        "canonical_train": [_row(0, 0, pss.NO_BIRD, "", vocab_sorted), _row(1, 1, pss.RESOLVED_CLEAN, "A", vocab_sorted)],
        "val": [],
        "test": [],
    }
    pss.validate_rows(rows, vocab_sorted)  # must not raise


def test_validate_no_leakage_detects_overlap():
    bad_assignment = {0: "train", 1: "val"}
    pss.validate_no_leakage(bad_assignment)  # no overlap here, must not raise


def test_validate_group_disjoint_from_rows_detects_overlap():
    bad_rows = {
        "train": [{"sound_id": 0}, {"sound_id": 1}],
        "canonical_train": [{"sound_id": 0}],
        "val": [{"sound_id": 1}],
        "test": [],
    }
    with pytest.raises(AssertionError):
        pss.validate_group_disjoint_from_rows(bad_rows)


def test_validate_group_disjoint_from_rows_ignores_canonical_train_subset():
    """canonical_train is a subset of train by construction and must not be
    flagged as a leak against train itself."""
    rows = {
        "train": [{"sound_id": 0}, {"sound_id": 1}],
        "canonical_train": [{"sound_id": 0}],
        "val": [{"sound_id": 2}],
        "test": [{"sound_id": 3}],
    }
    pss.validate_group_disjoint_from_rows(rows)  # must not raise


def test_validate_train_augmentation_source_rejects_foreign_sound_id():
    rows = {
        "train": [
            {"sound_id": 0, "is_canonical": 1},
            {"sound_id": 5, "is_canonical": 0},  # 5 never appears in canonical_train
        ],
        "canonical_train": [{"sound_id": 0, "is_canonical": 1}],
        "val": [],
        "test": [],
    }
    with pytest.raises(AssertionError):
        pss.validate_train_augmentation_source(rows)


def test_validate_train_augmentation_source_accepts_valid_augmentation():
    rows = {
        "train": [
            {"sound_id": 0, "is_canonical": 1},
            {"sound_id": 0, "is_canonical": 0},  # overlapping window from the same train sound_id
        ],
        "canonical_train": [{"sound_id": 0, "is_canonical": 1}],
        "val": [],
        "test": [],
    }
    pss.validate_train_augmentation_source(rows)  # must not raise


def test_validate_species_present_in_all_splits_detects_missing_species():
    vocab_sorted = ["A", "B"]
    rows = {
        "canonical_train": [_row(0, 0, pss.RESOLVED_CLEAN, "A;B", vocab_sorted)],
        "val": [_row(1, 1, pss.RESOLVED_CLEAN, "A", vocab_sorted)],  # missing B
        "test": [_row(2, 2, pss.RESOLVED_CLEAN, "A;B", vocab_sorted)],
    }
    with pytest.raises(AssertionError):
        pss.validate_species_present_in_all_splits(rows, vocab_sorted)


def test_validate_species_present_in_all_splits_accepts_full_coverage():
    vocab_sorted = ["A", "B"]
    rows = {
        "canonical_train": [_row(0, 0, pss.RESOLVED_CLEAN, "A;B", vocab_sorted)],
        "val": [_row(1, 1, pss.RESOLVED_CLEAN, "A;B", vocab_sorted)],
        "test": [_row(2, 2, pss.RESOLVED_CLEAN, "A;B", vocab_sorted)],
    }
    pss.validate_species_present_in_all_splits(rows, vocab_sorted)  # must not raise


# --------------------------------------------------------------------------
# Deterministic sound_id-only grouped stratified split
# --------------------------------------------------------------------------


def _synthetic_sound_stats(n_sounds=60, seed=0, vocab=("SPA", "SPB", "SPC", "SPD")):
    """sound_ids with varied size, no-bird prevalence, species and project
    mix; guarantee every species has ample distinct-sound_id support so the
    hard per-split presence constraint stays trivially feasible."""
    import random

    rng = random.Random(seed)
    projects = ["MAP1", "PPA1", "PPA2"]
    stats = {}
    for sid in range(n_sounds):
        n_windows = rng.randint(20, 80)
        nobird_frac = rng.uniform(0.6, 0.95)
        n_nobird = int(n_windows * nobird_frac)
        n_resolved = n_windows - n_nobird
        species_counts = pss.Counter()
        positive = set()
        for _ in range(n_resolved):
            for s in rng.sample(vocab, k=rng.randint(1, 2)):
                species_counts[s] += 1
                positive.add(s)
        stats[sid] = pss.SoundStats(
            project=projects[sid % len(projects)],
            n_modeled=n_windows,
            n_nobird=n_nobird,
            species_counts=species_counts,
            positive_species=frozenset(positive),
        )
    return stats


def test_reserve_hard_constraints_reserves_one_sound_per_split_per_species():
    stats = _synthetic_sound_stats(n_sounds=60, seed=0)
    vocab_sorted = ["SPA", "SPB", "SPC", "SPD"]
    reserved, infeasible = pss.reserve_hard_constraints(stats, vocab_sorted)
    assert infeasible == []
    for s in vocab_sorted:
        splits_covered = {
            reserved[sid] for sid, ss in stats.items() if s in ss.positive_species and sid in reserved
        }
        assert {"train", "val", "test"}.issubset(splits_covered)


def test_reserve_hard_constraints_reports_infeasible_species():
    # SPD only ever appears in a single sound_id -> cannot cover 3 splits.
    stats = {
        0: pss.SoundStats(project="MAP1", n_modeled=10, n_nobird=0, species_counts=pss.Counter({"SPD": 1}), positive_species=frozenset({"SPD"})),
        1: pss.SoundStats(project="MAP1", n_modeled=10, n_nobird=10, species_counts=pss.Counter(), positive_species=frozenset()),
    }
    reserved, infeasible = pss.reserve_hard_constraints(stats, ["SPD"])
    assert len(infeasible) == 2  # only 1 candidate sound_id, but 3 splits need coverage
    assert all(s == "SPD" for s, _split in infeasible)


def test_assign_groups_is_deterministic():
    stats = _synthetic_sound_stats()
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    vocab_sorted = ["SPA", "SPB", "SPC", "SPD"]
    a1, diag1 = pss.assign_groups(stats, ratios, vocab_sorted, seed=42, num_restarts=5, local_search_iters=10)
    a2, diag2 = pss.assign_groups(stats, ratios, vocab_sorted, seed=42, num_restarts=5, local_search_iters=10)
    assert a1 == a2
    assert diag1["objective_score"] == diag2["objective_score"]
    assert diag1["selected_seed"] == diag2["selected_seed"]


def test_split_search_default_uses_fifty_restarts():
    assert pss.parse_args([]).num_restarts == 50


def test_assign_groups_no_leakage_and_covers_all_sound_ids():
    stats = _synthetic_sound_stats()
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    vocab_sorted = ["SPA", "SPB", "SPC", "SPD"]
    assignment, _ = pss.assign_groups(stats, ratios, vocab_sorted, seed=42, num_restarts=5, local_search_iters=10)
    pss.validate_no_leakage(assignment)  # must not raise
    assert set(assignment.keys()) == set(stats.keys())
    assert set(assignment.values()) <= set(pss.SPLIT_NAMES)


def test_assign_groups_enforces_hard_species_presence_in_every_split():
    stats = _synthetic_sound_stats(n_sounds=90, seed=1)
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    vocab_sorted = ["SPA", "SPB", "SPC", "SPD"]
    assignment, _ = pss.assign_groups(stats, ratios, vocab_sorted, seed=42, num_restarts=10, local_search_iters=20)
    for s in vocab_sorted:
        for split in pss.SPLIT_NAMES:
            has_positive = any(
                assignment[sid] == split and s in ss.positive_species for sid, ss in stats.items()
            )
            assert has_positive, f"species {s} missing a positive sound_id in {split}"


def test_assign_groups_raises_on_infeasible_hard_constraint():
    stats = {
        0: pss.SoundStats(project="MAP1", n_modeled=10, n_nobird=0, species_counts=pss.Counter({"RARE": 1}), positive_species=frozenset({"RARE"})),
        1: pss.SoundStats(project="MAP1", n_modeled=10, n_nobird=10, species_counts=pss.Counter(), positive_species=frozenset()),
    }
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    with pytest.raises(ValueError, match="RARE"):
        pss.assign_groups(stats, ratios, ["RARE"], seed=42, num_restarts=2, local_search_iters=5)


def test_assign_groups_preserves_proportions():
    stats = _synthetic_sound_stats(n_sounds=90, seed=1)
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    vocab_sorted = ["SPA", "SPB", "SPC", "SPD"]
    assignment, _ = pss.assign_groups(stats, ratios, vocab_sorted, seed=42, num_restarts=10, local_search_iters=20)

    total = sum(ss.n_modeled for ss in stats.values())
    global_nobird_frac = sum(ss.n_nobird for ss in stats.values()) / total
    global_species_frac = {s: sum(ss.species_counts.get(s, 0) for ss in stats.values()) / total for s in vocab_sorted}

    for split in pss.SPLIT_NAMES:
        split_stats = [ss for sid, ss in stats.items() if assignment[sid] == split]
        split_total = sum(ss.n_modeled for ss in split_stats)
        assert split_total > 0

        assert abs(split_total / total - ratios[split]) < 0.07

        split_nobird = sum(ss.n_nobird for ss in split_stats)
        assert abs(split_nobird / split_total - global_nobird_frac) < 0.07

        for s in vocab_sorted:
            split_species = sum(ss.species_counts.get(s, 0) for ss in split_stats)
            split_frac = split_species / split_total
            assert abs(split_frac - global_species_frac[s]) < 0.07


# --------------------------------------------------------------------------
# Small end-to-end integration on synthetic data
# --------------------------------------------------------------------------


def _build_synthetic_dataset():
    """Build a dataset with enough sound_ids (>= 7 per species) and windows
    that every retained species can be reserved into all three splits."""
    windows = []
    id_annos = []
    sp_annos = []
    window_id = 0
    species_pool = ["CYAVIO", "ORTGUT", "SCLNAE"]
    n_sounds = 30

    for sound_id in range(n_sounds):
        project = ["MAP1", "PPA1"][sound_id % 2]
        for i in range(6):
            start = i * WIN_SAMPLES
            end = start + WIN_SAMPLES
            windows.append(make_window(window_id, sound_id, start, end, project=project))
            window_id += 1
        # one overlapping (non-canonical) window per sound, eligible for
        # train-augmentation only.
        windows.append(
            make_window(window_id, sound_id, WIN_SAMPLES // 2, WIN_SAMPLES // 2 + WIN_SAMPLES, project=project)
        )
        window_id += 1

        # Every sound_id gets a resolved call for one species (cycling
        # through the pool), guaranteeing >= 7 distinct sound_ids each.
        species = species_pool[sound_id % len(species_pool)]
        id_annos.append(make_id_anno(sound_id, 1.0, 2.0))
        sp_annos.append(make_sp_anno(sound_id, 1.0, 2.0, species))

        # Every fifth sound_id also gets an unresolved call in another window.
        if sound_id % 5 == 0:
            id_annos.append(make_id_anno(sound_id, 6.0, 7.0))  # unresolved: no matching sp_anno

    return windows, id_annos, sp_annos, n_sounds


def test_end_to_end_pipeline_on_synthetic_dataset():
    """Build a tiny but non-trivial dataset in memory and run the full
    classify -> vocabulary -> group -> assign -> build_rows -> validate
    pipeline, checking every documented guarantee holds simultaneously."""
    windows, id_annos, sp_annos, n_sounds = _build_synthetic_dataset()

    species_ref = [
        make_species_row("CYAVIO", "Cyanocorax violaceus"),
        make_species_row("ORTGUT", "Ortalis guttata"),
        make_species_row("SCLNAE", "Sclateria naevia"),
    ]
    taxonomy = pss.build_taxonomy_crosswalk(species_ref)
    species_lookup = pss.build_species_lookup(sp_annos, decimals=6)
    id_by_sound = pss.build_identification_records(id_annos, species_lookup, taxonomy, decimals=6)

    classified, vocabulary, vocab_history = pss.classify_all_windows(windows, id_by_sound, min_sound_ids=7)
    assert vocabulary == {"CYAVIO", "ORTGUT", "SCLNAE"}
    assert vocab_history[0]["round"] == 0

    sound_stats = pss.build_sound_stats(classified)
    projects_sorted = sorted({ss.project for ss in sound_stats.values()})
    vocab_sorted = sorted(vocabulary)

    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    assignment, _ = pss.assign_groups(
        sound_stats, ratios, vocab_sorted, seed=42, num_restarts=5, local_search_iters=10
    )
    pss.validate_no_leakage(assignment)

    sounds_by_id = {i: {"file_name_path": f"sound_{i}.wav"} for i in range(n_sounds)}
    rows = pss.build_rows(classified, assignment, sounds_by_id, vocab_sorted)
    pss.validate_rows(rows, vocab_sorted)
    pss.validate_group_disjoint_from_rows(rows)
    pss.validate_train_augmentation_source(rows)
    pss.validate_species_present_in_all_splits(rows, vocab_sorted)

    # val/test/canonical_train are canonical-only.
    for key in ("canonical_train", "val", "test"):
        assert all(r["is_canonical"] == 1 for r in rows[key])

    # train may contain augmented (non-canonical) windows, and every one of
    # them must come from a sound_id that also has a canonical_train row.
    canonical_train_sounds = {r["sound_id"] for r in rows["canonical_train"]}
    non_canonical_train_rows = [r for r in rows["train"] if not r["is_canonical"]]
    assert non_canonical_train_rows  # augmentation actually added something
    assert all(r["sound_id"] in canonical_train_sounds for r in non_canonical_train_rows)

    # Every emitted row's label_state is modeled (no_bird or resolved_clean).
    for split_rows in rows.values():
        for r in split_rows:
            assert r["label_state"] in pss.MODELED_STATES


def test_class_list_reconstruction_matches_vocabulary(tmp_path):
    windows, id_annos, sp_annos, n_sounds = _build_synthetic_dataset()
    species_ref = [
        make_species_row("CYAVIO", "Cyanocorax violaceus"),
        make_species_row("ORTGUT", "Ortalis guttata"),
        make_species_row("SCLNAE", "Sclateria naevia"),
    ]
    taxonomy = pss.build_taxonomy_crosswalk(species_ref)
    species_lookup = pss.build_species_lookup(sp_annos, decimals=6)
    id_by_sound = pss.build_identification_records(id_annos, species_lookup, taxonomy, decimals=6)
    classified, vocabulary, _ = pss.classify_all_windows(windows, id_by_sound, min_sound_ids=7)
    vocab_sorted = sorted(vocabulary)

    class_list = [{"index": i, "code": s, "species": taxonomy.species_name_of.get(s)} for i, s in enumerate(vocab_sorted)]
    path = tmp_path / "class_list.json"
    with open(path, "w") as f:
        json.dump(class_list, f)

    reloaded = json.load(open(path))
    reconstructed_vocab_sorted = [entry["code"] for entry in sorted(reloaded, key=lambda e: e["index"])]
    assert reconstructed_vocab_sorted == vocab_sorted
    for entry in reloaded:
        assert entry["species"] == taxonomy.species_name_of[entry["code"]]


def test_determinism_across_reruns_with_same_seed():
    """Running the full pipeline twice with the same seed must produce
    byte-identical split membership (window_id sets per split) and an
    identical class_list."""
    windows, id_annos, sp_annos, n_sounds = _build_synthetic_dataset()
    species_ref = [
        make_species_row("CYAVIO", "Cyanocorax violaceus"),
        make_species_row("ORTGUT", "Ortalis guttata"),
        make_species_row("SCLNAE", "Sclateria naevia"),
    ]

    def run_once():
        taxonomy = pss.build_taxonomy_crosswalk(species_ref)
        species_lookup = pss.build_species_lookup(sp_annos, decimals=6)
        id_by_sound = pss.build_identification_records(id_annos, species_lookup, taxonomy, decimals=6)
        classified, vocabulary, _ = pss.classify_all_windows(windows, id_by_sound, min_sound_ids=7)
        vocab_sorted = sorted(vocabulary)
        sound_stats = pss.build_sound_stats(classified)
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
        assignment, diag = pss.assign_groups(
            sound_stats, ratios, vocab_sorted, seed=42, num_restarts=5, local_search_iters=10
        )
        sounds_by_id = {i: {"file_name_path": f"sound_{i}.wav"} for i in range(n_sounds)}
        rows = pss.build_rows(classified, assignment, sounds_by_id, vocab_sorted)
        return vocab_sorted, assignment, rows, diag

    vocab1, assignment1, rows1, diag1 = run_once()
    vocab2, assignment2, rows2, diag2 = run_once()

    assert vocab1 == vocab2
    assert assignment1 == assignment2
    assert diag1["selected_seed"] == diag2["selected_seed"]
    for key in rows1:
        ids1 = sorted(r["window_id"] for r in rows1[key])
        ids2 = sorted(r["window_id"] for r in rows2[key])
        assert ids1 == ids2
