# Data Validation Report: PteroSet Species Classification Split (segmented_v4)

**Scope**: quantitative audit of the label-state construction and split-feasibility assumptions
behind `docs/design/perch2_species_linear_probe_plan.md`, using the real artifacts
(`data/windows_mapping_4.0overlap_segmented_v4.json`, `data/annotations_identification.json`,
`data/annotations_species.json`, `data/species.csv`, `data/metadata.csv`). All numbers below were
computed directly from these files (read-only), not from the design doc's prose. `n=160,244` windows
throughout (the current, shipped `segmented_v4` window set).

---

## 1. Label-state construction (`annotations_identification.json` vs `annotations_species.json`)

Both JSONs are COCO exports of the **same 15,372 raw RAVEN annotation rows** (`Tipo=BIO`,
`ID=AVEVOC` for all rows -- this dataset's raw labels contain no other identification type).
`annotations_species.json` is the strict subset of those rows whose `Determination` column resolved
to a valid `species.csv` code:

- 15,372 raw annotations -> 8,458 have `Determination == NaN` (unresolved) -> of the remaining 6,914,
  211 resolve to `INDETE` (explicit "indeterminate" placeholder, not a real species.csv code) and 1
  more matches no code at all -> **6,702** annotations survive into `annotations_species.json`
  (verified: `15,372 - 8,458 - 212 = 6,702`, exact match).
- **Invariant checked and holds (0 violations)**: every one of the 6,702 species annotations has an
  exact `(sound_id, t_min, t_max)` match in `annotations_identification.json`'s 15,372 rows, i.e.
  species.json is a pure filter of identification.json, never an independent source. Also checked and
  holds (0 violations): no `label==0` ("no bird") window in the windows JSON overlaps a resolved
  species annotation -- the binary and species annotation layers are mutually consistent.
- Both JSONs' `sounds` arrays are identical (same 563 sound_ids, same order) -- the two annotation
  levels were exported from the same sound-id assignment, so joining windows to either JSON by
  `sound_id` is safe.

Mapping this onto the existing binary `label` field in `windows_mapping_4.0overlap_segmented_v4.json`
(160,244 windows, `label` = "any identification-level annotation overlaps this window"):

| State | Windows | % of 160,244 | Task disposition |
|---|---|---|---|
| `label==0` (no bird, identification-level) | 125,171 | 78.1% | **Keep, all-zero multilabel target** |
| `label==1` **and** >=1 overlapping species-level annotation | 17,039 | 10.6% | **Keep, multilabel target** |
| `label==1` **and** 0 overlapping species-level annotation (unresolved) | 18,034 | 11.3% | **Exclude** |

**Candidate set for the species task = 142,210 windows (88.7% of 160,244)**, split as 125,171
no-bird / 17,039 species-positive. The 18,034 unresolved windows are a real, non-trivial 11.3% of the
whole dataset and must be tracked as an explicit exclusion list (window_id + reason), not silently
dropped by a `dropna()`-style filter -- they are evidence of "a bird called but nobody could name it,"
not absence of a bird, and conflating them with `label==0` would inject false negatives into the
multilabel target.

---

## 2. No-bird prevalence

Within the 142,210-window candidate set: **88.0% no-bird / 12.0% species-positive**. This is a
~7:1 imbalance the training/eval design must acknowledge explicitly (already true of the binary task
today: 125,171/160,244 = 78.1% no-bird overall, 88.0% once unresolved windows are dropped, since
dropping unresolved windows removes only positives).

No-bird prevalence is **not uniform across projects** and neither is unresolved-rate -- both must be
reported per split, not just once globally:

| Project | Total | No-bird | Resolved (species) | Unresolved | Distinct species |
|---|---|---|---|---|---|
| MAP1 | 13,018 | 65.5% | 34.4% | 0.1% | 72 |
| PPA1 | 31,104 | 77.8% | 21.3% | 0.9% | 105 |
| PPA2 | 38,947 | 82.7% | 1.9% | 15.3% | 43 |
| PPA3 | 42,895 | 81.3% | 2.5% | 16.1% | 7 |
| PPA4 | 34,280 | 73.9% | 12.0% | 14.1% | 57 |

**This is a data-quality trap, not noise**: unresolved rate ranges from 0.1% (MAP1) to 16.1% (PPA3),
i.e. two orders of magnitude apart, and resolved-species rate ranges from 34.4% down to 1.9%. PPA3 in
particular resolves species for only 7 of the 168 catalogued species despite having the *largest*
window count of any project (42,895). Excluding unresolved windows therefore does not shrink the
candidate population uniformly -- it disproportionately erases species evidence from PPA2/PPA3/PPA4,
which must be reported per-project, not aggregated into one global "88.0% no-bird" number that hides
it.

---

## 3. Distinct species and rare-species/group support

`species.csv` / `annotations_species.json` categories: **168 distinct species codes**, and (checked)
all 168 have at least one overlapping window in the candidate set -- no dead category, no duplicate
`code` values in `species.csv`.

**Window-count-based rarity** (post-duplication, i.e. counting every overlapping sliding window,
see Section 4 for why this overstates true evidence):

| Threshold | Species below it | % of 168 |
|---|---|---|
| < 5 windows | 12 | 7.1% |
| < 10 windows | 34 | 20.2% |
| < 20 windows | 63 | 37.5% |
| < 30 windows | 75 | 44.6% |

**Independent-event-based rarity** (counting distinct raw `annotations_species.json` rows, i.e. the
true number of separate vocalization events, ignoring sliding-window duplication):

- **24 / 168 species (14.3%) have exactly 1 raw annotation in the entire dataset**, and every one of
  those 24 comes from a **single sound_id** -- one recording, one moment.
- **47 / 168 species (28.0%) have <=3 raw annotations.**

**Group (sound_id) concentration**, computed on candidate (post-exclusion) windows:

| `distinct_sound_ids` for the species | Species count | % of 168 |
|---|---|---|
| <= 1 | 34 | 20.2% |
| <= 2 | 54 | 32.1% |
| <= 3 | 66 | 39.3% |
| <= 5 | 84 | 50.0% |
| max observed | 253 (one species reaches 253 of 563 sound_ids) | -- |

**98 / 168 species (58.3%) occur in exactly one project**: MAP1 (47), PPA1 (36), PPA2 (15); none are
PPA3- or PPA4-exclusive (consistent with PPA3/PPA4's own species being a subset shared with other
projects, see Section 2). A single-project species is, by construction, unobservable as a positive in
any LOPO test fold except the one where its home project is held out -- and in that one fold it has
zero training examples (`structurally_unseen`). **No LOPO fold-design choice can fix this: it is a
population fact, not a splitting-algorithm limitation.**

---

## 4. Overlapping-window duplication

Sliding windows are 5 s with a 1 s stride within a segment (240,000-sample window, 48,000-sample
stride at 48 kHz -- confirmed directly from consecutive `start` deltas: 133,233 within-segment steps
of exactly 48,000 samples, plus 21,372 cross-boundary jumps of 240,000 and 5,076 jumps of 192,000 for
PPA1's 9 s-stride segments). That is an 80% frame overlap between consecutive same-segment windows.

Species annotation durations are short relative to the window: mean 1.93 s, median 0.88 s (max
26.97 s, min 0.04 s). Consequently a single vocalization event is captured by many overlapping
windows: **6,700 / 6,702 raw species annotations have >=1 overlapping window; the mean is 4.16
overlapping windows per annotation (median 4, max 20)**, for 27,843 total window-annotation "hits"
against only 6,702 underlying independent events -- **a ~4x duplication factor** between "window
support" and "independent acoustic evidence."

**Consequence for eligibility/support gates**: any `min_support` rule defined purely in window counts
(as in the reviewed plan: `train_pos >= 10`) can be satisfied by as few as 2-3 independent events for a
species whose calls happen to span more windows, while a species with the same 2-3 independent events
but shorter calls may fail the same threshold. The 10-window threshold is not a stable proxy for "10
independent observations" -- it conflates correlated, non-independent samples of the same acoustic
event with genuinely new evidence. This inflation is worst exactly where it matters most: the 24
single-event species above have all their "windows" (2-20 of them) coming from that one moment, so a
window-count-based support figure of, say, 8 for one of them is **zero independent corroborating
events**, not "close to trainable."

The existing pipeline already partially defends against this: `run_splits()`'s train/val split uses
`GroupShuffleSplit` grouped by `sound_id` (correct -- keeps a whole recording, hence all its
overlapping windows, on one side of the train/val boundary), and the LOPO test split is filtered to
`start % window_size_samples == 0` (stride-aligned, non-overlapping) specifically to avoid inflating
test metrics with near-duplicate windows. **Any new split design must keep both of these properties.**

---

## 5. Feasibility of group-aware (`sound_id`) stratification preserving species proportions

Group **size** is not the obstacle: candidate-window counts per `sound_id` are tightly clustered
(1-288, median 257, mean 253 across all 563 sound_ids; every sound_id contributes >=1 candidate
window). A group-based split can hit an overall train/val/test volume target easily.

Group **composition** is the obstacle, and it is a hard mathematical limit, not an algorithm
weakness: because an entire `sound_id` must go to exactly one split, any species confined to a small
number of sound_ids cannot have its per-split proportions preserved -- it can only be **all-in-one-
split** for that number of splits, and absent from the rest, no matter which stratification method is
used (single random split, iterative multilabel stratification, or a hand-tuned heuristic). Given
Section 3's numbers:

- **34/168 species (20.2%) are structurally excluded from "preserve proportions" by construction**
  (they exist in exactly 1 sound_id -- there is only one group to assign, so there is nothing to
  distribute).
- **84/168 species (50.0%) have <= 5 sound_ids** -- distributing 5 groups across 3 splits at
  15/70/15-style ratios cannot approximate a smooth proportion; at best a species like this lands with
  a lumpy 0/1/4 or similar split, and standard iterative-stratification tooling (e.g.
  scikit-multilearn's `IterativeStratification`) does not natively support group constraints, so a
  bespoke group-first-then-label-balance approach is required (assign whole sound_ids to splits
  greedily by minimizing per-species proportion deviation across all species simultaneously, similar
  in spirit to iterative stratification but operating on group-level aggregated label vectors, not
  window-level rows).

**Recommendation**: do not promise "preserve species proportions" as a universal property of the
split. Instead: (a) pre-register a minimum breadth (e.g. `>= 4` distinct sound_ids, matching
roughly the 25th percentile away from the singleton cliff) below which a species is declared
out-of-scope for proportion-preservation and is instead explicitly tracked as
`structurally_unseen`/`test_absent`/`val_absent`, exactly mirroring the plan's existing five-state
eligibility schema but computed per split rather than per LOPO fold; (b) for species at/above that
breadth, use a group-level iterative/greedy stratified assignment and report the *achieved* deviation
from global proportion per split, per species, as a QA artifact -- not an assumption.

---

## 6. Simulated consequence for the LOPO design in `perch2_species_linear_probe_plan.md`

The plan's own per-species-per-fold eligibility rules (`min_support=10` window-count threshold;
`trainable` = `train_pos>=10 and train_neg>=1`; `evaluable` = `trainable and test_pos>0 and
test_neg>0`) were run against the real 142,210-candidate population, one LOPO fold per held-out
project:

| Held-out project | trainable species | evaluable species (that fold) |
|---|---|---|
| MAP1 | 91 | 20 |
| PPA1 | 104 | 58 |
| PPA2 | 127 | 28 |
| PPA3 | 134 | **7** |
| PPA4 | 132 | 55 |

**`macro_ap_core` (species evaluable in ALL 5 folds simultaneously) = 2 species: `PITSUL`, `VANCHI`.**
Both are broadly distributed (present with >0 windows in all 5 projects), but `PITSUL`'s and
`VANCHI`'s PPA3-held-out test-positive counts are 4 and 3 windows respectively -- and per Section 4's
~4x duplication factor, a handful of windows this small is plausibly **one single vocalization event**
duplicated by the sliding window, not several independent observations; the PPA3-fold AP estimate for
either species would carry enormous variance.

The plan pre-registers `min_core_species` (recommended default: 5) as the gate for producing any
headline `macro_ap_core` scalar. **With real data, `n_species_core = 2 < 5`: the plan's own
acceptance gate would not be met, and it would correctly produce no headline scalar, only per-fold
tables** -- this is not a flaw in the plan's logic (which explicitly anticipates and handles this
case), but it is a highly likely, quantifiable outcome that should be stated as an expected result
up front rather than discovered after a training run. It is also a strong, independent argument for
why a group-aware, non-LOPO train/val/test split (as this task requests) is the more informative
design for a species-classification headline metric: LOPO's fold-intersection requirement is the
direct cause of the 2-species core, whereas a single stratified split only needs species to be
`evaluable` once.

---

## Data Validation Report (structured summary)

### Pipeline Correctness
- Alignment: verified -- `annotations_species.json` is an exact subset of `annotations_identification.json`
  (0 mismatched `(sound_id, t_min, t_max)` rows); `sounds` arrays identical and same-order in both
  JSONs; 0 contradictions between binary `label` and species-overlap.
- Type safety: `start`/`end` in `windows_mapping_*.json` are integer sample counts; `t_min`/`t_max` in
  annotation JSONs are floats in seconds -- any join between them (as done here) must divide by
  `sample_rate` consistently; no dtype coercion issues found in the JSONs themselves.
- Value ranges: no NaN/Inf found in annotation timings; no orphan/negative durations found.

### Leakage Check
- Split integrity: the *existing* train/val split (`GroupShuffleSplit` on `sound_id`) is leakage-safe
  for overlapping windows. A *new* species-classification split must preserve this same group
  constraint; see Section 5 for why full proportion preservation is infeasible for ~20-50% of species
  regardless of method.
- Statistics leakage: N/A at this stage (no normalization/vocabulary fitting reviewed here); flag for
  follow-up once `build_species_labels.py`/embedding-normalization code exists.
- Augmentation leakage: N/A -- out of scope for this label/split review.

### Schema Validation
- Expected fields: present in both JSONs and `windows_mapping_*.json`; `species.csv` has no duplicate
  `code` values.
- Types: consistent; `window_id` is **positionally reassigned** on every `segment_windows` re-run
  (`prepare_dataset.py`), so it is only a stable identity within one frozen windows-JSON version --
  any downstream join must also carry `(sound_id, start, end, sample_rate)` for self-verification
  (the reviewed plan already does this).
- Completeness: all 168 species categories have >=1 evidencing window; all 563 sound_ids contribute
  >=1 candidate window; 0 sound_ids are fully excluded.

### Edge Cases
- Unresolved bird-call windows (18,034, 11.3%): confirmed to be excludable-but-must-be-logged, not
  droppable-silently.
- Single-event species (24 species, 1 raw annotation each): confirmed present; any split places 100%
  of that evidence in exactly one split.
- Single-project species (98/168, 58.3%): confirmed present; unobservable cross-project by
  construction.
- Filename-substring sound_id lookup in `data/data_reader.py` (`sound_filename in
  s["file_name_path"]`): checked for collisions across all 563 filenames -- **0 collisions today**,
  but this is a substring-containment match, not an exact match, and is a latent fragility if a future
  filename is a substring of another (e.g. a re-recorded/renamed file); add an explicit exact-match or
  uniqueness assertion rather than relying on today's filenames happening not to collide.

### Recommendations
- [ ] CRITICAL: Treat the 18,034 unresolved bird-call windows as a first-class, logged exclusion set
  (window_id + reason), never merged into either the no-bird class or silently dropped by a generic
  `dropna`-style filter -- both would corrupt the multilabel target (false negative vs. missing
  evidence are different things).
- [ ] CRITICAL: Do not promise per-species proportion preservation as a blanket property of any new
  split. Pre-register a minimum `sound_id` breadth (recommend `>= 4`) below which a species is
  declared out of scope for proportion preservation and instead tracked via explicit
  eligibility states (`structurally_unseen` / `val_absent` / `test_absent` / `test_single_class` /
  `evaluable`), computed for the actual train/val/test split, not only for LOPO folds.
- [ ] CRITICAL: Report support gates using **both** window-count and independent-annotation-event
  count (and distinct-`sound_id` count) per species per split. A `min_support` rule defined purely on
  window counts is inflated ~4x by sliding-window overlap and will mis-classify single-event species as
  having "enough" training support.
- [ ] CRITICAL: If a LOPO-style cross-project evaluation is retained alongside/instead of the new
  split, pre-register and publish the expectation that the cross-fold-evaluable core will likely be
  very small (measured here: 2 species) *before* running training, so a small `macro_ap_core` is
  understood as a confirmed data-population fact, not a late-discovered failure.
- [ ] WARNING: Report no-bird prevalence and unresolved-rate **per project**, not only globally --
  they vary by up to two orders of magnitude (unresolved: 0.1%-16.1%) and materially change which
  species are even resolvable per project (PPA3 resolves only 7/168 species despite being the
  largest project by window count).
- [ ] WARNING: Preserve the existing test-set convention (stride-aligned, non-overlapping windows) for
  any new split's val/test portions to avoid near-duplicate windows inflating apparent metric
  stability; keep train's overlapping windows only in train, and only after grouping by `sound_id`.
- [ ] WARNING: `data/config.yaml`'s `splits.test_size` and `splits.n_splits` are currently dead --
  `prepare_dataset.py::run_splits()` only reads `val_size`/`random_state` (LOPO fixes test
  membership to the held-out project and fixes `n_splits` to the project count). Either wire a new
  species split into these config fields for real, or remove them, so the config file cannot silently
  diverge from what code actually executes.
- [ ] SUGGESTION: Add a regression check (runnable whenever `annotations_identification.json`/
  `annotations_species.json` are regenerated via `data_reader.py`) asserting the three invariants
  verified manually here: (1) every species annotation has an exact identification-annotation
  counterpart; (2) no `label==0` window overlaps a resolved species annotation; (3) no filename is a
  substring of another filename in the sound-id lookup.
- [ ] SUGGESTION: Publish the per-species `(window_count, event_count, distinct_sound_id_count,
  distinct_project_count)` table produced for this report as a versioned artifact (e.g.
  `data/species_labels/segmented_v4/species_support_v1.csv`) so every future split/eligibility
  decision is auditable against the same numbers rather than recomputed ad hoc and potentially
  drifting.

STATUS: DONE
