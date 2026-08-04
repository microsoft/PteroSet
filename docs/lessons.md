# Lessons

## A reference vocabulary table can contain placeholder and duplicate-referent rows; audit before defining "resolved"

- **Mistake**: The species-classification plan's label-state definitions (`resolved_clean` /
  `resolved_out_of_vocab` / `unresolved_or_mixed`) initially treated "has a row in `species.csv`" as
  equivalent to "resolves to a real species," without first auditing `species.csv` itself for rows
  that aren't actually species-level. In fact six codes there carry a dash placeholder or a bare
  higher-taxon name (`PSITTACIDAE`, `PSITTACIFORMES`, `RHACAR`, `PICIDA_1`, `TYRANN_SP1`, `PSITTA`) rather than a
  real binomial, and two pairs of codes (`ATAPIL`/`ATRPIL`, `RAMTUC`/`RHATUC`) name the exact same
  species under two different codes. Left unaddressed, the placeholder codes would have silently
  entered the class vocabulary as if they were real species, and the duplicate pairs would have split
  one species' support across two vocabulary entries, each potentially failing the support gate
  individually despite the species overall having enough evidence.
- **Correct approach**: before defining what "resolved"/"in-vocabulary" means for a categorical label
  derived from a reference table, read the reference table itself and check for (a) placeholder or
  sentinel values in the field that is supposed to carry the real category name, and (b) multiple rows
  that refer to the same real-world entity under different keys. Encode the exclusion/canonicalization
  as a **rule** evaluated against the table (e.g. "the name field must be a two-part binomial, not a
  dash or a family name plus `sp.`"; "group rows by exact-match name text and canonicalize to one key
  per group"), not a hand-maintained list -- so the behavior stays correct automatically if the
  reference table is edited later.
- **General rule**: "present in the lookup table" is not the same as "a valid, unique category." Any
  time a pipeline treats presence-in-a-reference-table as its resolution criterion, explicitly check
  the table for placeholder/sentinel rows and duplicate-referent rows first, and make the resolution
  rule a function of the table's actual content, not an assumption that every row is a distinct, valid
  entry.

## A leakage-prevention grouping key must match the data's actual correlation structure, not just its most granular ID column

- **Mistake**: The species-split design in `docs/design/perch2_species_linear_probe_plan.md` grouped
  windows by `sound_id` (one group per audio file) to prevent a recording from crossing a
  train/val/test boundary, without first checking `data/metadata.csv` for a coarser correlation
  structure. In fact, PteroSet's `event_indicator` field (a recorder/deployment code) repeats across
  multiple recording dates at the same physical site *within* a project, and short `event_indicator`
  codes are not guaranteed unique *across* projects either. Grouping by `sound_id` alone would have
  let a single physical site contribute audio files to more than one split -- a subtler, easy-to-miss
  leakage channel than the one `sound_id`-grouping itself was designed to close (the same recording
  split across train/test), because it operates at the site level, not the file level, and doesn't
  show up unless the metadata join is actually checked.
- **Correct approach (as first written)**: before fixing a grouping key for leakage prevention, check
  whether the dataset's own metadata (recorder/site IDs, deployment codes, timestamps, sensor IDs,
  etc.) implies a coarser correlation structure than the most granular ID column being considered, and
  use the coarsest key that is actually justified by the data (here, `(project, event_indicator)`,
  joined from `data/metadata.csv`) -- not the most convenient or most obviously-unique-looking column.
- **General rule (as first written)**: "no X crosses a split boundary" is only as strong as the
  correlation structure X was chosen to represent. When designing a leakage-safe split, explicitly ask
  "what real-world unit produces correlated samples here (device, site, session, subject, recording
  day), and does my candidate grouping key actually match that unit, or just its most obvious ID
  column?" -- and verify the answer against the dataset's own metadata, not assumption.
- **Correction (2026-07-28, same day)**: this analysis was correct as a piece of due-diligence
  reasoning, but the site-level `(project, event_indicator)` grouping it led to was **never presented
  to, or approved by, the user** -- it was substituted in unilaterally as a "safer" elaboration of a
  split design the user had only approved at a coarser level of detail. When the user's actually
  approved design was confirmed, it specified `sound_id` grouping, not site grouping. The plan has
  been reverted to `sound_id` grouping accordingly (see
  `docs/design/CHANGELOG.md`, "Approved-design correction"). The abstract due-diligence point above
  (check metadata for a coarser correlation structure before fixing a grouping key) remains valid
  *input* to a design conversation, but is not itself authorization to change the design -- see
  "Confirm split-design decisions before implementation" below. A more rigorous-looking alternative is
  not automatically the more correct one if the user was never asked.
- **General rule (updated)**: identifying a theoretically stronger safeguard is not the same as being
  authorized to implement it. Surface the finding ("here's a coarser correlation structure I found in
  the metadata; do you want to group by it instead?") and wait for a decision, rather than silently
  adopting the more conservative-looking option as if rigor alone were sufficient justification.

## Don't inherit a detector's LOPO/fold split for a differently-shaped task

- **Mistake**: The Perch v2 species-classification plan
  (`docs/design/perch2_species_linear_probe_plan.md`) reused PteroSet's existing 5-fold
  leave-one-project-out (LOPO) evaluation structure -- built for, and validated against, the binary
  bird/no-bird detector -- for a new multilabel species-classification task, without first checking
  whether that task's own label availability and class balance actually support a per-project split.
  Species labels are only resolvable for a subset of bird-positive windows, and windows with any
  unresolved species annotation must be excluded from the modeled dataset entirely (including mixed
  resolved+unresolved windows, to avoid injecting false negatives into every other species' negative
  class). Slicing that already-reduced, unevenly-distributed-across-projects label space into 5
  per-project folds produced per-fold species pools too small or too skewed for the resulting
  eligibility/comparability machinery to stay simple, and reintroduced a harder question
  ("does a species classifier generalize to an unseen project?") than the one actually being asked
  ("do frozen embeddings separate species at all, given a representative sample of the data?").
- **Correct approach**: before reusing an existing split/fold structure for a new task, explicitly
  define that task's own valid label states first (e.g. what counts as a usable positive, a true
  negative, and an excluded/ambiguous example), measure whether those states are evenly available
  under the existing split's grouping, and only then decide whether the existing split still fits or
  a task-specific split must be derived. Here, the fix was one dataset-wide, group-aware,
  multilabel-stratified 70/15/15 split that preserves species/no-bird proportions, grouped by
  `sound_id` (the approved population and grouping unit -- see
  `docs/design/perch2_species_linear_probe_plan.md` and `docs/design/CHANGELOG.md`), built
  specifically for the species task's own label space, instead of the detector's LOPO folds. (An
  intermediate draft changed the grouping unit from `sound_id` to a coarser recording-site key without
  user approval; that change was reverted -- see the lesson above this one and "Confirm split-design
  decisions before implementation" below.)
- **General rule**: a split or cross-validation structure is a property of the task it was designed
  to evaluate (what it holds out, and why), not a generic artifact to be copied onto the next task
  that touches the same underlying windows. When adding a new modeling task on existing data, derive
  its split from that task's own label-availability and class-balance profile before assuming an
  existing fold/split structure still applies.

## Respect user-owned Git actions

- **Mistake**: Attempted to create a commit after the user intended to commit the staged changes.
- **Correct approach**: Stop immediately when the user says they will commit, preserve the staging
  state, and only report what is staged versus unstaged.
- **General rule**: Never perform a Git commit after the user claims ownership of that action.

## Confirm split-design decisions before implementation

- **Mistake**: Expanded a request for species-balanced train/validation/test splits into unapproved
  choices about site-level grouping, canonical-only windows, taxonomy gates, and optimizer details.
- **Correct approach**: Separate the user's required behavior from optional methodological choices,
  present each consequential choice explicitly, and wait for approval before writing split code or
  generating artifacts.
- **General rule**: Never implement a new experimental split until the population, negative-class
  handling, grouping unit, ratios, balance objective, and class-inclusion rules are confirmed.
- **Recurrence (2026-07-28, same day)**: the same failure mode repeated one level deeper: after the
  user approved a dataset-wide, task-specific 70/15/15 split in principle, a follow-on revision
  unilaterally substituted site-level `(project, event_indicator)` grouping and a three-metric
  (`n_windows`/`n_events`/`n_sites`) vocabulary gate for the simpler, `sound_id`-grouped, single-metric
  design the user actually approved -- reasoning that the substitution was "more rigorous," not that
  it had been requested. The user corrected this back to the approved design (`sound_id` grouping,
  `n_sound_ids >= 7` plus presence in all three splits, retained+rare windows kept encoding only
  retained species, canonical windows for stratification with overlapping windows added to train only,
  `C` selected by validation macro-AP). **Every individual choice in an approved design -- including
  ones an agent later revises "for rigor" while implementing or documenting it -- is itself subject to
  the same confirm-before-locking-in rule as the original design.** Producing a more careful-sounding
  variant of an approved choice is not the same as being asked to change it; present the variant as a
  proposal and wait, exactly as the original rule already requires for a brand-new design.

## Stream overlapping-window extraction by recording, not by full-dataset waveform cache

- **What worked**: Decode and resample one `sound_id` at a time, slice its windows, and feed a global
  fixed-size inference buffer before releasing the recording. This retained full GPU batches while
  avoiding repeated audio decoding.
- **Evidence**: Caching every decoded window before inference used about 65 GB and stalled the run.
  The streaming design reduced the process to roughly 7 GB and sustained about 400 windows/second on
  the full training extraction.
- **General rule**: For heavily overlapping audio windows, reuse decoded audio at the recording
  boundary, but keep model batches global and bounded. Do not cache the full dataset's waveforms.

## Persist prediction scores at the precision used for reported metrics

- **Mistake**: Metrics were computed from float64 probabilities, but persisted predictions were cast
  to float32. Near-saturated scores collapsed into ties, so AP recomputed from the NPZ differed from
  the published CSV.
- **Correct approach**: Store probabilities and derived scores as float64 when they are the
  reproducibility source for ranking metrics; verify every published metric directly from the saved
  prediction artifact.
- **General rule**: A result is not reproducible if the persisted prediction precision changes its
  ranking metrics, even when the model itself is deterministic.
