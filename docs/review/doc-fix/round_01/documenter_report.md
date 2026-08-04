# Documenter Report — doc-fix Round 1

**Target files** (not yet edited): `docs/design/perch2_species_linear_probe_plan.md`,
`docs/design/CHANGELOG.md`.

**Task**: revise the Perch v2 species linear-probe plan to use only fold 4
(`data/folds_segmented_v4/fold_4_PPA4_segmented/`, held-out test project **PPA4**), reframe the
experiment as a technical validation / feasibility check (not general cross-project robustness),
and simplify to the smallest scientifically defensible baseline, per explicit user invitation.

This report is a change plan only. No edits have been made to the plan or the changelog yet.

---

## 1. Facts verified before proposing any edit

### 1.1 Fold 4 = PPA4, confirmed at two independent layers

- `CLAUDE.md` fold-mapping table: `PAREX → MAP1 (fold_0)`, `GEOPARK_PUTUMAYO_2024 → PPA1 (fold_1)`,
  `GeoPark_II_T2_2024 → PPA2 (fold_2)`, `GeoPark_II_T3_2025 → PPA3 (fold_3)`,
  `GeoPark_II_T4_2025 → PPA4 (fold_4)`.
- `data/folds_segmented_v4/` on disk contains exactly `fold_0_MAP1_segmented` ...
  `fold_4_PPA4_segmented` — directory name itself encodes the held-out project.
- `prepare_dataset.py::run_splits` (lines 338-490): for each fold, **test** = non-overlapping windows
  from the held-out project only (`start % window_size_samples == 0`); **train/val** = the other four
  projects, partitioned by `GroupShuffleSplit(groups=sound_id, test_size=val_size, random_state=42)`
  so no audio file straddles train/val. I read `fold_4_PPA4_segmented/{train,val,test}_split.csv`
  directly: `test_split.csv` rows are all `project=PPA4`; `train_split.csv`/`val_split.csv` rows are
  all `project ∈ {MAP1, PPA1, PPA2, PPA3}`. This matches the CLAUDE.md description exactly.
- Row counts (verified by direct line count, header excluded): **train = 106,862**,
  **val = 19,102**, **test = 11,474**, union = **137,438** windows — vs. 160,244 in the full 5-project
  pool the current plan builds. All three splits are windowid-disjoint by construction (test is
  PPA4-only; train/val are non-PPA4-only), so no dedup/union logic is needed when materializing them.
- Split CSV columns are already `window_id, dataset, sample_rate, sound_id, start, end, label,
  spec_name, sound_filename, project`, and `start`/`end` are already **integer samples**
  (e.g. `start=0, end=240000` for a 5 s window at 48 kHz = 240,000 samples), not seconds. This is a
  minor naming mismatch worth fixing while editing: the current plan's identity-discipline section
  invents `start_sample`/`end_sample` column names for a new `pool.csv` it no longer needs to build;
  the real, already-integer column names in the existing split CSVs are `start`/`end`.

### 1.2 `prepare_dataset.py` split semantics (used to write the narrowed claim)

Leave-one-project-out is a **repo-wide, existing, unchanged mechanism** (`run_splits`), not something
this plan invented. Reusing fold 4 alone means: for the "why fold 4" framing, this plan is not
inventing a new held-out set — it is using one specific, already-materialized member of an existing
5-member family and explicitly declining to run the other four. This is a legitimate and cheap thing
to say plainly in the "Why" section.

### 1.3 Species-annotation provenance and the unresolved-label gap (confirmed against `data/data_reader.py`)

- `data/annotations_identification.json` has **one** category, `AVEVOC` — i.e. "a bird vocalization
  was heard here," independent of species resolution. This is the source of the existing binary
  bird/no-bird label used by `train.py`.
- `data/annotations_species.json` is built by `HumboldtAves.add_annotations` (annotation_level=
  "species"): a RAVEN row is included **only if** its `Determination` field matches a `species.csv`
  code (`category_match` on `determination`); rows whose determination doesn't resolve to a species
  code are silently dropped from this file (`if not category_match: continue`), but such rows *do*
  still exist in `annotations_identification.json` if their generic `ID` field was `AVEVOC`.
- **Consequence**: a window can have `has_bird == 1` (an `AVEVOC` annotation from the identification
  file overlaps it) while having **zero** matching rows in `annotations_species.json` (its bird call(s)
  never resolved to a species code). The current final plan's "In scope" line ("a species multi-hot
  target derived from `annotations_species.json`") never states what happens to such a window, nor to
  a genuine no-bird window — both would silently collapse to an all-zero multi-hot row, i.e. an
  implicit negative for every species, if nothing else is specified. That is exactly the ambiguity the
  user's requirement is asking to close.
- This exact gap was already correctly identified and solved once before, in this repo's own design
  history: `docs/design/round_01/architect_minimalist_proposal.md` §4.2 defines a verified six-state
  schema per window — `species_CODE` (one-hot per class), `any_species_known`, `has_bird`,
  `unresolved_bird` (`has_bird==1` and no overlapping annotation resolved to a species — "not a
  negative for any class"), `usable_strict` (`unresolved_bird==0`). The *current* final plan
  (`perch2_species_linear_probe_plan.md`) references "the six-state label schema" twice (lines 290,
  311) as something `build_species_labels.py` implements, but never actually enumerates or defines it
  in the plan document itself, and never states the population rule that uses it. This is a real,
  pre-existing documentation gap, not something introduced by this round's fold change — worth fixing
  in the same edit pass since the user's requirement (defining the evaluation population explicitly)
  needs exactly this schema stated in the plan, not left implicit/off-document.

### 1.4 Orcas reference pipeline: checked the "collapse to 2-3 flat scripts" recommendation against actual precedent, not assumed

- `eval_perch_ecotype.py::run_fulldata` (lines 651-727): trains **one** `LogisticRegression` on one
  fixed train split, evaluates on one fixed test split — a single-split baseline, no cross-validation,
  no fold indexing, no global-pool/remap step at all. This is the closest structural analog to "one
  fold, three splits."
  `extract_perch_3class.py`: extracts Perch embeddings **directly per split CSV**
  (`train_split.csv`/`val_split.csv`/`test_split.csv`) into three separate `.npz` files — no
  intermediate global pool, no separate fold-materializer/remap script. It is used precisely when
  there is exactly one fixed partition to serve, which is now this plan's actual situation.
- `remap_perch_embeddings.py` exists in the sibling repo **only** to avoid re-extracting when the
  *same* underlying window pool gets **re-partitioned** into a *different* set of splits later
  (`splits_ecotype` → `splits_ecotype_cascade_balanced`), i.e. it amortizes GPU cost **across multiple
  competing partitions of one pool**. With a single fold and three disjoint, final splits, there is no
  second partition to amortize against — the entire justification for a pool + remap step is absent
  here. This directly supports (not just asserts) eliminating the pool/remap machinery.
- Conclusion: the "2-3 flat scripts modeled on `orcas_dclde2026`" recommendation in the task prompt is
  **verified correct** by the actual precedent in the sibling repo (`extract_perch_3class.py` +
  `eval_perch_ecotype.py::run_fulldata`), not merely assumed.

---

## 2. Proposed concrete edits to `perch2_species_linear_probe_plan.md`

### 2.1 Scope/claim narrowing (Why / What sections)

- **"Why" section** (currently lines ~14-25): add an explicit paragraph stating this experiment is a
  **feasibility / technical validation** of whether frozen Perch v2 embeddings carry usable species
  signal on PteroSet audio, evaluated on **one held-out project (PPA4, fold 4 of the existing 5-project
  LOPO family)**, and is **not** a claim of general cross-project robustness across all 5 projects. A
  full 5-fold LOPO species study remains a legitimate, separate future extension if this baseline
  succeeds — explicitly deferred, not silently dropped, mirroring the plan's own existing
  out-of-scope idiom.
- Replace "(2) evaluation uses 5 fixed leave-one-project-out (LOPO) folds, not one seed-selected
  stratified split" (line 22) with: "(2) evaluation uses the one existing fold whose held-out test
  project is PPA4 (`fold_4_PPA4_segmented`), reusing the repo's already-existing leave-one-project-out
  split mechanism (`prepare_dataset.py::run_splits`) as-is, rather than running it across all 5
  projects or defining a new stratified split."
- **"What (scope)" section**: add to **Out of scope**: "claims of species-classification performance
  generalizing to projects other than PPA4; a full 5-project LOPO species study (future work, gated on
  this baseline's result)." Remove any implied promise of 5-fold coverage from the in-scope line.

### 2.2 Replace the 5-fold architecture with the single-fold, no-pool architecture

Eliminate (with rationale re-stated in the "Future migration path" section, not silently dropped):

| Current mechanism | Disposition | Why |
|---|---|---|
| Global `pool.csv` / `pool_emb_v2.npz` / `pool_manifest.json` (all 160,244 windows, all 5 projects) | **Removed** | Exists only to amortize one extraction pass across 5 folds × 3 splits. With 1 fold, only the 137,438-window union of fold 4's own three splits is ever needed; extracting only those windows (not the full 160,244) is strictly less compute and less code, and is exactly what `extract_perch_3class.py` already does for a single fixed partition in the sibling repo. |
| `build_fold_embeddings.py` (gather-by-`window_id` fold materializer, ×5 folds ×3 splits) | **Removed** | No second partition of the same pool ever exists in this plan, so there is nothing to "materialize a view of." Extraction goes directly per split, mirroring `extract_perch_3class.py`. |
| `remap_perch_embeddings.py`-style remapping | **Removed** | Its entire purpose in the sibling repo is re-partitioning one pool into a *different* split layout without re-running Perch. Fold 4's three splits are final and disjoint; there is no second partition to remap into. |
| Dual hash lineage (`pool_embedding_hash` vs. `label_recipe_hash`, cross-fold cache invalidation) | **Removed, replaced with one manifest per split** | The dual-hash split existed to avoid re-running GPU work for 5 folds when only labels changed. With one fold and direct per-split extraction, a single extraction manifest per split (recording windows-JSON hash, script source hash, model-dir hash, extraction params, counts, exclusions) gives the same reproducibility guarantee with far less bookkeeping. Re-running label derivation is already nearly free (pure Python over ~137k rows) and does not need its own hash-independence proof once there is no GPU-cost amortization decision riding on it. |
| `species_eligibility_fold{i}.csv`, `species_diagnostics_fold{i}.csv`, `fulldata_results_species_fold{i}.csv`, `fold_manifest.json` (×15) | **Kept, unsuffixed** (drop the `_fold{i}`/×15 framing; one manifest/eligibility/diagnostics/results file each, since there is exactly one fold now) | The underlying gates (identity match, trainable-column selection, numerical diagnostics) are still load-bearing safeguards, independent of fold count — see §2.4. Only the fold-indexing and the "×5"/"×15" multiplicities are removed. |
| `macro_ap_core` / `macro_ap_core_summary.csv` / `min_core_species` (fixed cross-fold-comparable species intersection, averaged with std across 5 folds) | **Removed** | This entire mechanism exists solely to make a metric comparable *across* folds. With one fold there is nothing to intersect or average across — it degenerates to a no-op that would just restate `macro_ap_fold_own_eligible` under a different name. |
| `macro_ap_per_fold_own_eligible.csv` (never-averaged, one row per fold) | **Kept, renamed to `macro_ap_summary.csv`, one row** | This becomes the single headline metric file: fold 4's own evaluable species set, `n_species_evaluable`, and the exact species list — the "never average across folds" rule is now vacuous (there is nothing else to average with) but the underlying content (a single, own-eligible-set macro-AP with its species list disclosed) is exactly right and should be kept, just without the fold-comparison framing. |

Net script count: **3** (down from 4 required + 1 optional):
1. `build_species_targets.py` — six-state population schema (defined explicitly in-document this
   time, not merely referenced) restricted to fold 4's train/val/test `window_id`s; multi-hot species
   target; `class_list.json`.
2. `extract_perch_embeddings_fold4.py` — modeled directly on `extract_perch_3class.py`: one call per
   split (`train`, `val`, `test`), Perch v2 embeddings only for that split's **species-eligible**
   windows (see §2.3 — non-eligible windows are out of scope for this experiment and should not cost
   GPU time), writing `{train,val,test}_emb.npz` + one extraction manifest per split (fail-loud,
   global **and** per-project failure ceilings, since train/val still span 4 different projects).
3. `train_perch_logreg_fold4.py` — L2-normalize, trainable-column selection via
   `species_eligibility.csv`, fit `OneVsRestClassifier(LogisticRegression(solver="lbfgs",
   max_iter=1000, C=1.0))` (no `multi_class` kwarg) on train; report diagnostics + a val pass
   (diagnostic-only, never used to select `C` or any other hyperparameter); headline metrics on test.

### 2.3 Explicit evaluation-population definition (new subsection, does not exist today)

Add a new section, e.g. "## Evaluation population: what counts as a species-classification window,"
stating the six-state schema explicitly (borrowing the already-vetted terms from
`docs/design/round_01/architect_minimalist_proposal.md` §4.2, which the current final plan alludes to
but never defines):

- `has_bird` — existing v4 binary label: at least one `AVEVOC` (identification-level) annotation
  overlaps the window.
- `any_species_known` — at least one `annotations_species.json` annotation overlaps the window
  (i.e., at least one bird call in the window was resolved to a species code).
- `unresolved_bird` — `has_bird == 1 and any_species_known == 0`: a bird call is present but no
  overlapping call resolved to a species. **Not a negative for any class.**
- **Modeled population (closed-set species classification)** = windows with `any_species_known == 1`
  only. This is narrower than `usable_strict` in the round-1 schema (which also keeps pure
  no-bird windows): it deliberately **excludes** both `unresolved_bird` windows and true no-bird
  windows, so the experiment measures species discrimination conditioned on a resolved species call
  being present, and does not conflate itself with the separate bird/no-bird detection task
  `train.py` already owns.
- For every modeled window, its own resolved species class(es) are positive; every *other* class in
  `class_list.json` is an implicit negative for that window in the One-vs-Rest formulation — i.e.
  other resolved-species windows supply each class's negatives, exactly as required.
- **Two limitations to state verbatim, attached to every results table**:
  1. *Annotation incompleteness*: absence of a species-level annotation for class X in a modeled
     window means "not annotated as X," not "verified absent." All negatives in this design are soft
     negatives, inherited from finite human-annotation effort, not resolved absence.
  2. *Unseen/absent PPA4 species*: the classifier is trained only on modeled (i.e.
     `any_species_known == 1`) windows from MAP1/PPA1/PPA2/PPA3 (fold 4's train split). A species with
     zero training-eligible positives across those four projects is `structurally_unseen` and cannot
     be evaluated in PPA4 regardless of how common it is there; conversely a species well-represented
     in training but absent from PPA4's test split is `test_absent` and contributes no AP. Both are
     expected outcomes of PteroSet's regional species turnover, not defects, and must be visible in
     `species_eligibility.csv`, never silently dropped or scored as zero.

This directly answers the user's "define the evaluation population explicitly" and "explain annotation
incompleteness and unseen PPA4 species limitations" requirements, and closes the pre-existing
six-state-schema documentation gap noted in §1.3.

### 2.4 Mandatory safeguards — preserved, with where each one now lives

| Safeguard | Preserved via |
|---|---|
| Multilabel formulation | `OneVsRestClassifier(LogisticRegression)` over the modeled (species-eligible) population, unchanged. |
| Unresolved-label exclusion | New explicit population definition, §2.3 (`unresolved_bird` windows excluded, not treated as negatives). |
| No test tuning | `C` fixed at 1.0 (no grid, matching current scope); val used only for diagnostic sanity checks, never for any selection; test touched exactly once for headline metrics. |
| Integer sample identity | Existing split-CSV columns `window_id, sound_id, start, end, sample_rate` (already integers, confirmed in §1.1) — plan text corrected to reference these actual column names instead of inventing `start_sample`/`end_sample` for a `pool.csv` that no longer exists. |
| Fail-loud audio loading | Extraction manifest per split records global **and** per-project failure ceilings (default 0 for both, override must be explicit/reviewed); `SystemExit` on breach, unchanged in spirit from the current plan, just scoped to 3 manifests instead of 1 pool manifest. |
| L2 normalization | `sklearn.preprocessing.normalize(X, norm="l2")` per split, stateless, unchanged. |
| Trainable-column selection | `species_eligibility.csv` (renamed, unsuffixed) remains the pre-fit gate feeding `OneVsRestClassifier.fit()`; five-category schema (`trainable`, `structurally_unseen`, `test_absent`, `test_single_class`, `evaluable`) kept verbatim — it is about species support, not fold count. |
| Convergence/non-finite diagnostics | `species_diagnostics.csv` keeps `n_iter_`, `converged`, `coef_finite`, `coef_l2_norm` — reported for every trainable species. See §2.5 for what changes (the exclusion *rule*, not the diagnostic fields). |
| scikit-learn >= 1.7 compatibility | No `multi_class` kwarg anywhere; unchanged. |
| License/environment smoke test | Phase 0 smoke test + Perch v2 license/version verification, unchanged — this is independent of fold count and stays exactly as currently specified. |

### 2.5 Removals evaluated and recommended (non-baseline components)

- **Optional single-label comparator** (`train_perch_logreg.py --comparator single_label`,
  `comparator_retention.csv`, `comparator_eligibility_fold{i}.csv`,
  `comparator_diagnostics_fold{i}.csv`): **remove entirely.** It was already optional; it adds a
  second classifier, a second eligibility table, a second diagnostics table, and a retention-disclosure
  mechanism, none of which serve the core feasibility question ("does the frozen embedding carry
  species signal at all, on one held-out project"). Consistent with the user's explicit invitation to
  simplify to a baseline. If a future round wants a single-label reference number, it can be reproduced
  from the same eligibility/diagnostics machinery already specified for the primary classifier.
- **Coefficient-norm exclusion threshold** (`coef_norm_ceiling`, the rule that
  `numerically_trusted` gates headline-metric inclusion on an arbitrary norm ceiling): **remove the
  gating rule, keep the number.** `coef_l2_norm` stays as a reported diagnostic field (protects the
  mandatory "non-finite diagnostics" safeguard), but it no longer excludes a species from
  `macro_ap_summary.csv` on its own. Only unambiguous failure signals — `converged == False` (didn't
  converge within `max_iter`) or `coef_finite == False` (NaN/Inf) — exclude a species from the
  headline metric and are flagged in an appendix. A large-but-finite, converged coefficient norm is a
  known, expected symptom of a small-sample near-separable fit, not by itself evidence the fit is
  wrong; picking an ceiling value with no fold-specific data to calibrate it against (this plan's own
  original text flagged `50.0` as `[RECOMMENDATION, not a measured fact]`) is exactly the kind of
  unnecessary, hard-to-justify machinery the user's baseline invitation is asking to drop.
- **Calibration / `C`-grid**: already out of scope in the current final plan (fixed `C=1.0`, no
  sweep). **No change needed** — confirm this stays out of scope; do not reintroduce it.
- **Segment/file pooling**: already out of scope in the current final plan (window-level evaluation
  only, explicit rationale about PteroSet's duty-cycled non-continuous file structure).
  **No change needed** — confirm this stays out of scope.
- **`macro_ap_core` / cross-fold-comparable species intersection**: covered in §2.2 — removed, since
  it is meaningless with one fold.

### 2.6 Metrics recommendation

- **Primary**: one macro-AP number over fold 4's own `evaluable` species set (formerly
  `macro_ap_fold_own_eligible`, now the only macro-AP number), reported with `n_species_evaluable` and
  the exact species list in `macro_ap_summary.csv`. Go/no-go rule kept, reworded to drop the
  "across the 5 folds" framing: macro-AP beats a per-species prevalence-only baseline by a
  pre-registered margin (default +5 points, configurable); a negative result remains a valid, useful
  outcome, not a pipeline defect.
- **Secondary**: per-species AP + ROC-AUC table (`fulldata_results_species.csv`), eligibility-masked,
  with train/test support counts alongside every row (kept from the current plan, unsuffixed).
- **Reported, not decision-driving**: val-split diagnostics (same metrics, computed only to sanity
  check the fit, never used to pick `C` or any other setting, and never blended with the test-set
  headline number).

### 2.7 Minimal artifacts / commands / tests

**Artifacts** (flat, no fold suffix): `species_labels_fold4.json`, `class_list.json`,
`{train,val,test}_emb.npz` + one extraction manifest per split, `species_eligibility.csv`,
`checkpoints/perch/logreg_species_fold4.joblib`, `species_diagnostics.csv`,
`fulldata_results_species.csv`, `macro_ap_summary.csv`.

**Commands**: same 4-phase shape, flattened —
Phase 0 (smoke test + license check, unchanged) →
Phase 1 `build_species_targets.py` (fold 4 only) →
Phase 2 `extract_perch_embeddings_fold4.py --split {train,val,test}` (no separate materializer phase) →
Phase 3 `train_perch_logreg_fold4.py` → Phase 4 `results.md`, explicitly scoped in its own title/abstract
to "PPA4 held-out technical validation," not general robustness.

**Tests** — trimmed from 11 to 7, keeping only what maps to a preserved safeguard:

| Kept (renamed/simplified) | Maps to |
|---|---|
| `test_species_label_derivation.py` | Six-state schema / unresolved-label exclusion correctness. |
| `test_perch_io.py` | Audio segment loader + integer-sample identity discipline. |
| `test_eligibility_categories.py` | Trainable-column-selection boundary conditions (unchanged by fold count). |
| `test_l2_normalization.py` | L2 normalization safeguard. |
| `test_trainable_column_selection.py` | Trainable-column-selection / constant-column avoidance. |
| `test_extraction_identity_gates.py` (renamed from `test_fold_cache_gates.py`) | Fail-loud on a corrupted/missing `window_id` during extraction — no "cache" framing since there is no pool to invalidate. |
| `test_numerical_diagnostics.py` (collapses `test_numerical_trust_flagging.py` + `test_diagnostics_exclusion_reasons.py`) | Convergence/non-finite diagnostics, without the removed ceiling-exclusion logic. |
| Smoke test (end-to-end tiny synthetic run) | Kept, updated to produce `macro_ap_summary.csv` only. |

**Dropped**: `test_cache_hash_independence.py` (dual-hash lineage removed — nothing left to prove
independent), `test_comparator_gates.py` (comparator removed).

---

## 3. Proposed `docs/design/CHANGELOG.md` entry

Add a new dated section after the existing Round 8 history (do not edit Rounds 1-8's content — they
remain an accurate historical record of how the *5-fold* design was reached):

```markdown
## Round 9 — single-fold (PPA4) scope reduction (2026-07-28)

**Trigger**: explicit user requirement to run this experiment as a technical validation of PteroSet
species classification feasibility on exactly one held-out project (PPA4 / fold 4 of the existing
5-project leave-one-project-out family), with explicit invitation to simplify to a baseline.

**What changed**: the plan's scope was narrowed from "5 fixed LOPO folds" to "the one existing fold
whose held-out test project is PPA4." This removed every mechanism whose sole purpose was serving
multiple folds from one shared pool: the global embedding pool, the fold materializer/remapper, the
dual pool/label hash lineage, `macro_ap_core` and its cross-fold species intersection, and per-fold
(`_fold{i}`, ×5/×15) artifact suffixes/multiplicities. It also removed the optional single-label
comparator and the coefficient-norm exclusion threshold (kept the diagnostic field, dropped the
headline-metric-gating rule), consistent with the same simplification directive. It added an explicit
evaluation-population definition (closed-set species classification conditioned on windows with at
least one resolved species annotation) that the Round 1-8 plan referenced ("the six-state label
schema") but never actually defined in-document — this was a pre-existing documentation gap, not
introduced by the fold-count change, closed in the same pass.

**What did not change**: multilabel formulation, unresolved-label exclusion, no-test-tuning discipline,
integer-sample identity, fail-loud audio loading, L2 normalization, trainable-column selection,
convergence/non-finite diagnostics reporting, scikit-learn >= 1.7 compatibility, and the license/
environment smoke test are all preserved, just re-scoped to one fold instead of five.

**Status**: plan revised and re-finalized under this narrower scope; superseding the "5-fold" framing
of Rounds 1-8 without invalidating their reasoning about label derivation, identity discipline, or
numerical trust, which all remain load-bearing at any fold count.
```

---

## 4. Open items needing a decision before editing (flagged, not resolved unilaterally)

- **Extraction scope**: recommend extracting embeddings only for species-eligible
  (`any_species_known == 1`) windows in each split, not all ~137k windows in the union — this saves
  GPU time and is consistent with the narrowed population. This is a *new* choice (the current plan
  extracts the full 160,244-window pool regardless of eligibility) and should be called out explicitly
  as a change, not folded in silently.
- **Val-split embeddings**: recommend still extracting/reporting val-split diagnostics for
  no-test-tuning discipline and basic overfit sanity-checking, even though `C` is fixed and nothing is
  tuned on val. Confirm this is worth the (small) extra extraction cost, or drop val extraction entirely
  and rely on train/test only — either is defensible; the report's recommendation is to keep it, since
  it costs little and is exactly the kind of guard that catches an integration bug before it reaches
  the headline number.
- **Species eligibility artifact naming**: renaming `species_eligibility_fold{i}.csv` →
  `species_eligibility.csv` (and similarly for diagnostics/results files) is a pure simplification; flag
  in the edit that any future re-introduction of multi-fold LOPO species work would need to re-add the
  suffix, not invent a new name.

---

STATUS: PLAN READY
