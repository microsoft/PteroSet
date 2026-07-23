# Round 2 — Mirroring `orcas_dclde2026`'s Perch v2 + LogReg Protocol for PteroSet Species Classification

**Author**: architect-experiment (Round 2 revision)
**Inputs required by the user and inspected in this session**: `orcas_dclde2026/eval_perch_ecotype.py` (read in full), `orcas_dclde2026/reports/ecotype_classifier.md` §1–8 (read in full), `orcas_dclde2026/docs/experiments/log.md` entries S1.4/S2.3, `orcas_dclde2026/pip-requirements.txt`, and the three PteroSet Round‑1 drafts (`architect_experiment_proposal.md` [mine], `architect_minimalist_proposal.md`, `architect_pipeline_proposal.md`).

## 0. How to read this document

- `[VERIFIED: <source>]` — read directly from a file or fetched page in this session; quoted or paraphrased faithfully.
- `[RECOMMENDATION]` — my design judgment; falsifiable, revisable.
- `[GATE]` — a decision that MUST be made by running a specific, named computation before proceeding; the number is not yet known.
- `[UNRESOLVED FROM ROUND 1]` — a Round‑1 finding that still blocks a downstream step.
- This document **extends** `docs/design/round_01/architect_experiment_proposal.md`. It does not repeat Round 1's dataset‑quality audit (duplicate species codes, rank‑mixing, the 14,205/6,703-vs-15,372/6,702 discrepancy), Perch 2 dimensionality/licensing facts, or the general phased‑milestone skeleton — those stand as written unless explicitly revised below. Section 12 states exactly what carries over unchanged, what is revised, and what is new.

---

## 1. Why

The user's explicit instruction is: *make PteroSet's species baseline as comparable as possible to the exact, already-validated `eval_perch_ecotype.py` workflow* (5 s / 32 kHz / peak‑0.25, cached NPZ, L2‑normalized embeddings, LBFGS `LogisticRegression`, window‑level and recording‑level reporting) — but only where that workflow's assumptions actually hold for PteroSet. Reusing a working recipe verbatim (rather than re‑deriving one from scratch) has real scientific value: it lets both `birds_bioacoustics` and `orcas_dclde2026` reports cite the same methodology section, reduces the number of untested design choices, and reuses a dependency footprint (`tensorflow==2.21.0`, `kagglehub==1.0.0`, `librosa==0.11.0`, `scikit-learn==1.7.2`) already proven to work end‑to‑end on this cluster across two projects `[VERIFIED: orcas_dclde2026/pip-requirements.txt]`.

But `eval_perch_ecotype.py`'s protocol was built for a task with three properties PteroSet's species task does **not** have:

1. **Mutual exclusivity.** Ecotype is (almost) always one-per-recording; PteroSet species co-occur (dawn chorus).
2. **Downstream-of-a-detector, fully-labeled-when-present.** Every KW annotation that has an ecotype is unambiguous; the "missing ecotype" annotations (10.4%) are simply dropped from the ecotype dataset, not treated as an unresolved label needing special handling in the target vector.
3. **A single, class-stratifiable split** (`StratifiedGroupKFold` by `sound_filepath`, stratified on `ecotype_label`), not a leave-one-project-out design where entire sites/projects are withheld from training and species are unevenly distributed across sites.

Ignoring these differences and copying the multinomial-LogReg recipe verbatim would let scientifically-wrong numbers leave the building (e.g., a macro‑F1 computed by forcing one winner-take-all class per window, silently discarding true co-occurring positives; or per-species AUC computed on a fold where that species never occurs in training, giving `nan`/degenerate scores masqueraded as evaluable). This round's job is to specify **exactly which parts of the Orcas recipe transfer unchanged, which parts need a PteroSet-specific substitute, and how the choice is decided by measurement rather than by architect intuition.**

---

## 2. What (scope of Round 2)

**In scope:**
- A byte-for-byte-as-possible adaptation of `eval_perch_ecotype.py`'s preprocessing (window loading, peak-norm, embedding extraction, NPZ caching, L2-normalization, LBFGS LogReg) to PteroSet's `folds_segmented_v4`.
- A Phase 0 **audit protocol** with concrete numeric decision gates that determine (a) whether the primary baseline is single-label, multilabel one-vs-rest, or both; (b) which species are evaluable in which LOPO fold; (c) what the correct PteroSet analogue of "recording-level" reporting is.
- Exact metrics, exact leakage rules, exact file/CLI contracts for this adaptation.
- A convergence verdict against the three Round‑1 drafts.

**Out of scope (unchanged from Round 1):** fine-tuning Perch's backbone, temporal/sequence models over consecutive windows, active-learning/re-annotation workflows, and the binary bird-detector task itself (`train.py`/`checkpoints_v4`) — this proposal only adds a species head downstream of frozen Perch embeddings, using PteroSet's existing folds.

---

## 3. Exact Orcas recipe — what is being mirrored

`[VERIFIED: orcas_dclde2026/eval_perch_ecotype.py, orcas_dclde2026/reports/ecotype_classifier.md §8]`

| Element | Orcas exact value | Carries to PteroSet as-is? |
|---|---|---|
| Sample rate | 32,000 Hz | Yes — Perch 2 requires this; PteroSet resamples from 48 kHz. |
| Window duration | 5.0 s (`PERCH_WINDOW_SAMPLES = 160_000`) | Yes — PteroSet windows are already 5.0 s (`data/config.yaml:23`), so no resizing is needed (§5 below). |
| Window centering | `center_sec = (window_start+window_end)/2.0`; `librosa.load(filepath, sr=32000, offset=center-2.5, duration=5.0)` | Degenerates to "load the window's own `[start, end)` boundaries" for PteroSet because window length already equals Perch's native window length (5.0 s window centered on its own midpoint reproduces the same interval, modulo file-boundary clipping at the very first/last window of a file). **Adopt the same `librosa.load(sr=32000, offset=..., duration=5.0)` call pattern verbatim** rather than manually slicing at 48 kHz then resampling — this avoids re-implementing resampling and gives numerically identical semantics to Orcas' loader. `[RECOMMENDATION]` |
| Peak normalization | Manual re-implementation of perch-hoplite's `target_peak=0.25` (not delegated to the model wrapper) | Adopt verbatim — same function, same constant. |
| Model loading | `tf.saved_model.load(model_dir)` on a **local SavedModel directory** downloaded once via `kagglehub.model_download("google/bird-vocalization-classifier/tensorFlow2/perch_v2")`, called through `model.signatures["serving_default"]` | **Adopt verbatim in place of Round 1's `perch_hoplite.zoo.model_configs.load_model_by_name('perch_v2')` recommendation.** See §10 decision. |
| Embedding key selection | Heuristic: pick the non-logits 2‑D output whose last dim is 1280 or 1536 (`inspect_model_outputs`) | Adopt verbatim; add an assertion that the picked dim equals 1536 (fail loudly if Kaggle ships a different Perch v2 artifact revision). |
| Batch size | 64 | Adopt as default; expose as CLI flag (already true in Orcas script). |
| Embedding cache | `checkpoints/perch/ecotype_emb_{version}_{split}.npz` with keys `embeddings, ecotype_label, sound_filepath, dataset, window_id` | Adopt the same NPZ *shape of contract* (embeddings + label + filepath + partition-id + window_id), renamed for PteroSet (§9). |
| Pre-classifier normalization | `sklearn.preprocessing.normalize(embeddings, norm="l2")` applied to **window-level** embeddings before fitting/predicting, and again to **mean-pooled recording-level** embeddings after pooling | Adopt verbatim for window-level. For the recording-level analogue, see §7.3 — the pooling *unit* changes, the L2-norm-after-pooling step does not. |
| Classifier | `sklearn.linear_model.LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, multi_class="multinomial")`, no explicit `random_state` in the fitting calls themselves (only in few-shot subsampling) | Adopt `solver="lbfgs", max_iter=1000, C=1.0` verbatim as the shared numerical core. `multi_class="multinomial"` is retained **only** in the single-label branch (§7); the multilabel branch uses independent per-class `LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)` instances (`OneVsRestClassifier` semantics), each of which is itself deterministic given `lbfgs` convergence — matching Orcas' documented determinism property (`log.md` S2.3: "deterministic given fixed embeddings"). |
| Metrics | Macro F1, macro ROC-AUC (`multi_class="ovr", average="macro"`), per-class F1, confusion matrix, per-dataset breakdown (window-level only) | Adopted for the single-label branch verbatim (with an eligible-species mask, §6.2). The multilabel branch uses AP/ROC-AUC per species instead of forced-argmax F1 (§8). |
| Reporting granularity | Window-level (no pooling) **and** recording-level (mean-pooled over all windows sharing `sound_filepath`, majority-vote label) | Window-level: adopted verbatim. Recording-level: **rejected as-is** and replaced by segment-level pooling — see §7.3, this is the single largest structural adaptation in this round. |
| Few-shot protocol | k ∈ {4,8,16,32} recordings/class, 5 seeds, sample k per class, fit LogReg, report ROC-AUC/F1 mean±std | Retained as an optional secondary ablation (§9 ablation matrix), not primary, and only run on the single-label branch's per-fold eligible-species subset. |

---

## 4. Structural incompatibilities — the parts of the Orcas protocol that do not transfer

Each item below is a concrete, falsifiable claim about *why* a piece of the Orcas recipe would give a scientifically indefensible number if copied verbatim onto PteroSet, with the evidence available today and the exact measurement Phase 0 must run to confirm/refute the magnitude.

### 4.1 Multilabel co-occurrence (species, not just "any bird")

`orcas_dclde2026`'s ecotype label is single-valued per annotation and (with 0.07% documented exception, §3.4 of `ecotype_classifier.md`) effectively single-valued per recording. PteroSet's species label is not: two co-occurring `CYAVIO` events in `annotations_species.json:6913-6933` were already found in Round 1 `[VERIFIED: repo]`, and — more importantly — PteroSet's own existing report `reports/window_counts_by_fold.md` already shows, at the **identification (any-bird)** annotation level, mean 1.6–1.9 and up to 10 overlapping annotations per positive test window `[VERIFIED: reports/window_counts_by_fold.md §"Annotation density per positive test window", v3-era]`. This is annotation-count, not distinct-species-count — it does not yet tell us the true co-occurrence rate — but it is strong prior evidence that forcing a single winner-take-all species label per window (the Orcas `multi_class="multinomial"` design) would discard real information at non-trivial frequency. **This must be measured, not assumed** (Gate A, §6.1).

### 4.2 Incomplete/ambiguous species annotation coverage

Orcas drops the 10.4% of KW annotations lacking an ecotype from the ecotype-specific dataset entirely (`ecotype_classifier.md` §3.1/§4.1) — a clean design because "missing ecotype" only ever removes an annotation from consideration, it never has to interact with a multilabel target vector. PteroSet's species annotations sit inside a *superset* of "any-bird" (`AVEVOC`) annotations: a window can be bird-positive (per the existing binary task) while its overlapping annotation(s) carry no resolvable species code (rank-mixed placeholders like `PSITTACIDAE`, `TYRANN_SP1`, or simply absent from `species.csv` — Round 1 §4). Silently treating such windows as **all-zero multilabel targets** — i.e., "negative for every species" — would inject an unknown number of false negatives into every species' negative class simultaneously (already flagged as the single largest correctness risk by the `architect_minimalist_proposal.md` draft, §4.2: *"treating 'bird present, species unknown' windows as negatives for every species class... would poison every class's negative set"*). Round 1's six-state label schema (`no_bird / clean_single / clean_multi / ambiguous_unresolved / ...`) already exists to prevent this; Round 2 reaffirms it as **mandatory**, not optional, precisely because the Orcas drop-if-missing pattern cannot be applied at the window level without corrupting the multilabel target (it *can* be applied at the single-annotation level, which is exactly what the primary-annotation reduction in §7.2 does).

### 4.3 Leave-one-project-out vs. `StratifiedGroupKFold`

Orcas' split (`ecotype_classifier.md` §5.1) is `StratifiedGroupKFold(n_splits=20, shuffle=True)`, grouped by `sound_filepath`, **stratified on `ecotype_label`** — chosen from a 300-seed sweep (§5.2) to balance both split-size and per-class representation simultaneously. This is only possible because Orcas is not required to hold out an entire *site/project*; the grouping unit (file) and the stratification target (ecotype) are independent enough that a seed search can jointly satisfy both.

PteroSet's fold design is fixed and non-negotiable: leave-one-**project**-out (`data/folds_segmented_v4/fold_{i}_{PROJECT}_segmented/`). Project is confounded with recording site, equipment, habitat, and species pool. There is **no seed to sweep** — the held-out project's species composition is whatever it is; some species may be far more common in the held-out project than in the remaining four (an under-represented-in-train problem, same direction as Orcas' SAR/OKW imbalance) and some species may occur in **only** the held-out project or **only** the training projects (an absent-not-just-imbalanced problem Orcas does not have to solve, because stratification guarantees every class appears in every split by construction). **Do not port `StratifiedGroupKFold` to PteroSet's primary folds.** If a species-stratified, non-LOPO split is ever produced for comparability with the Orcas methodology, it must be a clearly-labeled *secondary, in-distribution-only* experiment, explicitly disclosed as not testing cross-project generalization, and never substituted for or blended with the LOPO result in a headline claim (leakage rule L6, §11).

### 4.4 Unseen species per fold

A direct consequence of §4.3: for a species whose entire PteroSet occurrence is concentrated in one project (analogous to Orcas' SAR being concentrated in 2 of ~30 datasets, `ecotype_classifier.md` §2.3), the LOPO fold that holds out that project has **zero training examples** for that species — not "few," zero. That species cannot be evaluated in that fold at all (any reported score would be undefined or a degenerate `nan`/0 that looks like a real number if not gated). Symmetrically, a species that is rare enough to appear *only* in the held-out project's test windows and nowhere in the other four training projects is a **zero-shot** case, not a generalization test. Both directions must be computed explicitly per fold (Gate B, §6.2) and excluded from that fold's supervised macro-averaged metrics, reported separately as "structurally unseen," never silently dropped or silently scored as zero.

### 4.5 Recording-level reporting: PteroSet "recordings" are not acoustically coherent units

Orcas' recording-level pooling (mean-pool all windows sharing `sound_filepath`, L2-normalize, majority-vote the ecotype label) is defensible because an Orca encounter recording is very likely to contain one ecotype throughout (0.07% mixed-file rate, §3.4). A PteroSet audio file is fundamentally different: `[VERIFIED: repo — CLAUDE.md]` each file is **48 concatenated 10-second time-lapse segments** (a duty-cycled recorder sampling ~10 s snapshots at intervals, not one continuous recording), so a whole-file mean-pool would average together up to ~288 5-second windows (48 segments × up to 6 overlapping 5 s windows per 10 s segment at 1 s hop) drawn from acoustically unrelated moments — potentially hours or days apart within the same file — then assign the whole thing one majority-vote species label. For a dawn-chorus soundscape with a long-tailed species pool this is not a mild simplification, it is close to meaningless as a *classification* unit, though "does file F contain species X at all" is a legitimate, different question (soundscape occupancy). **Recording/file-level majority-vote reporting is rejected outright for PteroSet** (§7.3 gives the substitute).

---

## 5. Window-geometry check: does "load the window's own boundaries" actually match Orcas' centering formula?

`[VERIFIED: data/config.yaml:23]` PteroSet window duration is exactly 5.0 s, matching Perch's native window (`PERCH_WINDOW_SEC = 5.0`). Given a window `[ws, we)` with `we - ws = 5.0`, Orcas' `center_sec = (ws+we)/2` and `offset = center_sec - 2.5, duration = 5.0` yields `offset = ws` exactly. So for PteroSet, "center on the window midpoint" and "load the window's own boundaries" are the **same operation** — there is no approximation to worry about, unlike Orcas itself (whose underlying windows are reportedly shorter than 5 s per `ecotype_classifier.md` §3.6, spectrograms use "3 s windows" for the CNN pipeline while Perch is fed a wider 5 s context centered on that 3 s window — an intentional context-expansion Orcas relies on and PteroSet does not need). `[RECOMMENDATION]` Implement PteroSet's loader with the identical `librosa.load(filepath, sr=32000, offset=window_start_sec, duration=5.0)` call (not `offset=center-2.5`) — algebraically identical here, but named for what it actually does in this repo, and it avoids a silent bug if window duration is ever changed from 5.0 s in a future segmented version. One edge case requires a policy: the **last window of a file** whose nominal `we` slightly exceeds the file's actual duration (float rounding) must pad with zeros rather than raise — mirror Orcas' `try/except → zeros(PERCH_WINDOW_SAMPLES)` fallback, but (unlike Orcas) **log every fallback event to the extraction manifest** rather than only emitting a `warnings.warn` — a bird-species linear probe silently trained on some fraction of all-silence embeddings is a subtler failure than an orca detector with the same issue, because a silent window is a well-defined "no bird" case for the binary task but an ambiguous one for a multilabel species head (is a zeroed embedding a valid negative for every species, or a data error that should be excluded?). `[RECOMMENDATION]`: treat it as excluded (not zero-labeled), and gate on the exclusion rate in Phase 0.

---

## 6. Phase 0 audit protocol — exact queries and decision gates

All of the following run on CPU, no Perch/TensorFlow dependency, against the existing `windows_mapping_4.0overlap_segmented_v4.json`, `data/annotations_species.json`, `data/species.csv`, and `data/folds_segmented_v4/*/{train,val,test}_split.csv` — i.e., this is pure label-provenance analysis and can be finished before any embedding is extracted, exactly like the "Phase 0 feasibility spike" already scoped in all three Round‑1 drafts.

### 6.1 Gate A — co-occurrence rate (decides single-label vs. multilabel primacy)

**Definition**: `co_occurrence_rate = N(species-positive windows with ≥2 distinct canonical species codes overlapping, after applying the Round‑1 species_taxonomy_crosswalk.csv to collapse duplicates like RAMTUC/RHATUC) / N(species-positive windows)`. Must be computed with the **same overlap rule** used for the multilabel target (`t_min < we and t_max > ws`, any temporal overlap — Round 1 §5, `architect_pipeline_proposal.md` line 64) so the number is apples-to-apples with the target construction it is meant to justify or invalidate.

Also compute, for transparency (mirroring Orcas' own "4 mixed-ecotype files, 0.07%" disclosure, `ecotype_classifier.md` §3.4): the count and % of windows with ≥2 distinct species, broken down by project, and the max distinct-species count observed in a single window.

**Decision gate**:
- `co_occurrence_rate < 0.05` (5%, `[RECOMMENDATION — threshold configurable, pre-registered before running the audit]`) → **Primary baseline = single-label**, using the primary-annotation reduction of §7.2, mirroring the Orcas `multi_class="multinomial"` recipe exactly. Multilabel one-vs-rest is still trained and reported as a **secondary/diagnostic** run whose purpose is to quantify what the single-label reduction discarded (report the delta in macro-AP between "OvR on full multilabel target" and "OvR on primary-annotation-only target").
- `co_occurrence_rate >= 0.05` → **Primary baseline = multilabel one-vs-rest**, evaluated with AP/ROC-AUC per species (§8), because a forced single winner would misclassify a non-trivial, measured fraction of true positives as negatives by construction. The single-label primary-annotation reduction is still computed and reported, explicitly labeled `[SECONDARY — comparability with orcas_dclde2026 methodology only; known lower bound, not a standalone scientific claim]`.
- **Both branches are always run.** The gate decides which one is the paper's/report's headline number, not whether the other exists — this directly answers the user's "decide whether baseline should be single-label filtered subset, one-vs-rest multilabel, or both": **both, always; the gate decides primacy.**

### 6.2 Gate B — per-fold × per-species eligibility table (mandatory regardless of Gate A's outcome)

Compute, for each of the 5 LOPO folds and each canonical species (post-crosswalk), a table with columns: `train_support, val_support, test_support`. Derive three per-fold species sets:
- **Eligible**: `train_support >= min_support` (`[RECOMMENDATION: min_support = 10 positive windows]`, same default as Round 1) **and** `test_support >= 1`. Only eligible species contribute to that fold's macro-averaged metrics.
- **Structurally unseen (zero-shot)**: `train_support == 0` and `test_support >= 1`. Reported separately per fold; never included in "macro-F1"/"macro-AP" denominators, never silently scored.
- **Untestable**: `train_support >= min_support` and `test_support == 0`. Reported as "no held-out evidence this fold," excluded from that fold's test metrics, still contributes to global support totals.

This table is the direct PteroSet analogue of Orcas' §2.3/§3.3 imbalance tables, generalized from "5 classes, all present everywhere" to "K species, unevenly present across 5 disjoint project-defined folds" — the eligibility computation itself is the safeguard that Orcas never needed because `StratifiedGroupKFold` made it structurally impossible for a class to vanish from a split.

### 6.3 Gate C — segment-level co-occurrence (decides the recording-level substitute, §7.3)

Recompute Gate A's co-occurrence statistic restricted to within a single 10 s time-lapse segment (the ≤6 windows sharing one segment index) rather than across the whole file. This decides whether the PteroSet analogue of "recording-level" reporting can use majority-vote (if segment-level co-occurrence is also low) or must use "any-positive-in-segment" multilabel aggregation (if not).

### 6.4 Gate D — primary-annotation tie rate (needed only if the single-label branch is used or reported)

When two or more overlapping annotations have identical temporal overlap with a window (a tie in "largest-overlap wins," §7.2), report the tie rate and the deterministic tie-break rule applied (`[RECOMMENDATION]`: break ties by earliest `annotation_id`, documented in code, never by random choice).

**Phase 0 exit criterion**: a single markdown report (`reports/species_phase0_audit.md`, mirroring the format of `reports/window_counts_by_fold.md` and `orcas_dclde2026/reports/ecotype_classifier.md` §2–3) containing Gates A–D's numbers, before any line of embedding-extraction code is run. This report is reviewed by the Inquisitor before Phase 1 begins (decision gate, §14).

---

## 7. Label derivation — exact algorithm (both branches)

### 7.1 Multilabel target (primary if Gate A ≥ 5%, always computed as secondary otherwise)

Unchanged from Round 1: for window `w = [ws, we)` and species annotation `a = [t_min, t_max)` on the same `sound_id`, `w` is positive for `a.canonical_species` iff `t_min < we and t_max > ws`. Six-state per-window label: `no_bird` (existing binary label 0 → all-zero target, retained as negatives), `clean_single`, `clean_multi`, `ambiguous_unresolved` (overlapping annotation exists but species is unresolved/rank-mixed/duplicate-code — **excluded from training and from the negative set of every species, not zeroed**), plus the two Round‑1 edge states for padding/silence fallback (§5). `min_overlap_frac` exposed, defaults to 0.0 (any overlap), consistent with the existing binary derivation.

### 7.2 Single-label "primary-annotation" reduction (mirrors Orcas' `source_annotation_idx`)

For each species-positive window, select the overlapping annotation with the **largest temporal overlap** (`min(we, t_max) - max(ws, t_min)`), ties broken by earliest `annotation_id` (Gate D). That annotation's canonical species becomes the window's single label. Windows whose only overlapping annotation(s) are all `ambiguous_unresolved` are **dropped from the single-label dataset entirely** (exact PteroSet analogue of Orcas dropping "missing-ecotype" annotations, `ecotype_classifier.md` §3.1). This produces a strict subset of species-positive windows with one label each — directly loadable into `sklearn.linear_model.LogisticRegression(..., multi_class="multinomial")` exactly as `run_fulldata()` does in `eval_perch_ecotype.py`.

### 7.3 Recording-level substitute: segment-level pooling

`[RECOMMENDATION, new in Round 2]` Replace whole-file mean-pooling with **10 s time-lapse-segment mean-pooling**: pool the ≤6 windows sharing one segment index (derived from `windows_mapping_4.0overlap_segmented_v4.json`'s segment/window geometry — each ~480 s file (~433 s for PPA1) is composed of 48 (~48 for PPA1 at 9 s stride) discrete 10 s segments per `[VERIFIED: repo — CLAUDE.md]`), L2-normalize the pooled vector — same two-step recipe as `aggregate_to_recordings()`, applied to a coherent acoustic unit instead of an incoherent one. Labeling within a segment: majority-vote if Gate C shows low segment-level co-occurrence, else "any-positive-in-segment" multilabel target (per-species binary: did this species occur anywhere in this 10 s snapshot). Report this explicitly as `segment-level`, not `recording-level`, in all tables and code (naming discipline — a reviewer who knows the Orcas report must not assume the two are the same statistical unit).

---

## 8. Metrics — exact, both branches

### 8.1 Window-level (primary reporting unit, both branches, always computed)

**Single-label branch** (primary-annotation subset, restricted to that fold's Gate‑B-eligible species): macro F1, macro ROC-AUC (`roc_auc_score(..., multi_class="ovr", average="macro")`), per-class F1, confusion matrix, per-project breakdown — the exact `_compute_full_metrics()` contract from `eval_perch_ecotype.py`, ported verbatim, with the only change being the eligible-species mask applied before averaging.

**Multilabel branch** (full multilabel target): per-species **average precision (AP)** as the primary per-species metric (robust to the severe class imbalance already documented for the binary task, `reports/window_counts_by_fold.md` positive rates 0.17–0.35, and expected to be far worse per-species), macro-AP over Gate‑B-eligible species as the headline scalar, micro-AP as a secondary pooled view, per-species ROC-AUC as a secondary metric (kept for comparability with the single-label branch's use of ROC-AUC), label-ranking average precision (LRAP) as a genuinely multilabel-specific summary with no single-label analogue. All computed with `sklearn.metrics`, all restricted to species meeting Gate B's per-fold eligibility.

### 8.2 Segment-level (secondary; substitute for Orcas' "recording-level")

Same metric pair as §8.1, computed on segment-pooled embeddings/targets from §7.3. Reported as a distinct table, never merged into the window-level table.

### 8.3 Cross-fold statistical summary (mandatory, generalizes Orcas' single-split report to PteroSet's 5-fold LOPO)

Orcas reports one train/val/test split; PteroSet has 5 LOPO folds. For every scalar metric in §8.1–8.2: report the per-fold value, the mean ± standard deviation across the 5 folds (only over folds/species where that species is Gate‑B-eligible), and a bootstrap 95% CI (window-level resampling, stratified by `sound_id` to respect the grouping structure — 1,000 resamples, `[RECOMMENDATION]`, carried over from Round 1 §8). Per-species results additionally report **how many of the 5 folds that species was eligible in** — a species eligible in only 1–2 folds gets a wide, low-trust interval flagged as such, never presented with the same visual weight as a species eligible in all 5.

### 8.4 Perch's own zero-shot classifier (secondary baseline, unchanged from Round 1)

Perch v2 ships its own class logits (a 10,932-class-scale output per the Orcas script's v1 comment, analogous large label space for v2). Reporting whether PteroSet's target species appear in Perch's own label space, and if so what zero-shot AP its raw logits achieve before any PteroSet-specific training, remains a useful, cheap secondary baseline (Round 1 §8, retained unchanged) — not present in the Orcas script (which only compares Perch+LogReg against its own from-scratch ResNet-18), but a natural and inexpensive extension.

---

## 9. Exact files, CLI, artifacts

```
species_probe/
  build_species_labels.py       # Phase 1 — multilabel + primary-annotation reduction,
                                 # writes species_labels_segmented_v4.csv (6-state schema)
                                 # and species_labels_primary_segmented_v4.csv (single-label subset)
  audit_phase0.py                # Gates A-D; writes reports/species_phase0_audit.md
  extract_perch_embeddings.py    # Phase 2 — raw-SavedModel extraction (see §10)
  train_species_probe.py         # Phase 3 — both branches, gated by Gate A's stored verdict
  eval_species_probe.py          # Phase 4 — window-level, segment-level, cross-fold summary
```

CLI, mirroring `eval_perch_ecotype.py`'s flag surface where the concepts transfer:

```
python species_probe/audit_phase0.py \
    --windows_mapping data/windows_mapping_4.0overlap_segmented_v4.json \
    --annotations_species data/annotations_species.json \
    --species_csv data/species.csv \
    --crosswalk data/species_taxonomy_crosswalk.csv \
    --folds_dir data/folds_segmented_v4 \
    --min_support 10 \
    --cooccurrence_threshold 0.05 \
    --out reports/species_phase0_audit.md

python species_probe/build_species_labels.py \
    --windows_mapping ... --annotations_species ... --crosswalk ... \
    --out_multilabel data/species_labels_segmented_v4.csv \
    --out_primary data/species_labels_primary_segmented_v4.csv \
    --min_overlap_frac 0.0

python species_probe/extract_perch_embeddings.py \
    --folds_dir data/folds_segmented_v4 \
    --audio_dir data/audios_48khz \
    --model_dir checkpoints/perch/model_v2 \
    --model_source kaggle:google/bird-vocalization-classifier/tensorFlow2/perch_v2 \
    --sr 32000 --window_sec 5.0 --target_peak 0.25 \
    --batch_size 64 \
    --emb_dir checkpoints/perch_species/

python species_probe/train_species_probe.py \
    --branch {single_label,multilabel,both}    # default: both; primary decided by audit_phase0's stored verdict
    --emb_dir checkpoints/perch_species/ \
    --labels_multilabel data/species_labels_segmented_v4.csv \
    --labels_primary data/species_labels_primary_segmented_v4.csv \
    --folds_dir data/folds_segmented_v4 \
    --eligibility_table reports/species_phase0_audit.md \
    --C 1.0 --solver lbfgs --max_iter 1000 \
    --calibration_split val \
    --out_dir checkpoints/species_probe/

python species_probe/eval_species_probe.py \
    --probe_dir checkpoints/species_probe/ \
    --level {window,segment,both} \
    --out_dir reports/species_probe_results/
```

**Artifact naming** (mirrors the Orcas NPZ contract, §3): `checkpoints/perch_species/emb_{project}_{split}.npz` with keys `embeddings [N,1536] float32, window_id [N] str, sound_id [N] str, segment_id [N] str, project [N] str, label_state [N] str (six-state), species_multilabel [N,K] uint8, species_primary [N] int32 (or -1 if excluded from single-label subset)`. A single embeddings artifact serves **both** branches (leakage rule L5, §11) — only the label/target columns differ between what's fed to the single-label vs. multilabel classifier heads.

---

## 10. Embedding extraction: raw-SavedModel-direct vs. `perch_hoplite` — explicit decision

**Context**: Round 1 (and both sibling Round‑1 drafts) recommended `perch_hoplite.zoo.model_configs.load_model_by_name('perch_v2')` → `model.embed(waveform)`. `eval_perch_ecotype.py` instead calls `kagglehub.model_download(...)` directly, loads the resulting SavedModel with `tf.saved_model.load()`, and invokes `model.signatures["serving_default"]` directly, with a small heuristic (`inspect_model_outputs`) to pick the embedding output — no `perch_hoplite` dependency anywhere in `orcas_dclde2026`'s `pip-requirements.txt` `[VERIFIED]`.

**Options considered:**

| | `perch_hoplite.zoo` wrapper (Round 1 choice) | Raw SavedModel + `kagglehub` (Orcas' choice) |
|---|---|---|
| Dependency footprint | New package for this repo, actively developed upstream, less in-house track record | Zero new dependency beyond what two sibling projects already run in production (`tensorflow==2.21.0`, `kagglehub==1.0.0`) |
| Preprocessing correctness | Delegated to the library's own `target_peak`/resample/window logic — less code to get wrong, but opaque if it changes upstream | Reimplemented manually (peak-norm, windowing) — more code, but transparent, inspectable, and already validated end-to-end on two real projects at this org |
| Numeric comparability with the sibling repo's published numbers | Not guaranteed identical | By construction identical (same call sequence) |
| Pooling utilities | `pooled_embeddings(time_pooling, channel_pooling)` helper built in | None — must implement mean/L2-norm pooling manually (already required anyway, see §7.3) |
| Officially-supported forward path | Yes — this is Google's blessed inference wrapper | No — a thinner, DIY layer over the same underlying SavedModel |

**Decision**: adopt the **raw SavedModel + `kagglehub`** path as the primary implementation for `extract_perch_embeddings.py`, for three reasons: (1) it is the exact mechanism the user asked to mirror, (2) it has zero net new dependency risk given `orcas_dclde2026` already runs it in production, (3) it makes numeric comparability with the cited sibling-repo results a construction fact rather than an assumption to verify later. **Mitigation for the loss of `perch_hoplite`'s vetted abstraction**: add a one-time cross-check test (`tests/test_embedding_equivalence.py`, §13) that runs both the raw-SavedModel path and `perch_hoplite.zoo.model_configs.load_model_by_name('perch_v2').embed(...)` on a small fixed set of windows and asserts the two embeddings agree within floating-point tolerance (e.g., cosine similarity > 0.999) — this is cheap (perch_hoplite need only be installed in a throwaway verification environment, never in the production extraction path) and catches the case where the two libraries' internal preprocessing has silently diverged. `[RECOMMENDATION]` This is a genuine, disclosed revision from Round 1, not a silent contradiction — flagged again in §12.

---

## 11. Train/validation/test and leakage policy — exact, final

Carried over from Round 1 and reaffirmed as mandatory (L1–L4), plus new rules forced by this round's Orcas comparison (L5–L7):

- **L1.** Reuse `data/folds_segmented_v4/fold_{i}_{PROJECT}_segmented/{train,val,test}_split.csv` verbatim. No new splitting logic. Species labels/embeddings are joined onto these files by `window_id`, never used to re-derive a split.
- **L2.** Train/val leakage control = `GroupShuffleSplit` grouped by `sound_id` (already in `prepare_dataset.py::run_splits`, unchanged) — no recording contributes windows to both train and val.
- **L3.** Test = held-out project's non-overlapping windows only (`start % window_size_samples == 0`), unchanged.
- **L4.** Calibration/threshold fitting happens on the **validation split only**, per fold, never on test.
- **L5.** *(new)* Embedding extraction is leakage-blind and fold-agnostic: one embeddings artifact per window, computed once, reused by both the single-label and multilabel branches, and across all 5 LOPO folds' train/val/test roles for that window. No branch or fold gets a differently-preprocessed embedding.
- **L6.** *(new, from §4.3)* Any species-stratified, non-LOPO split (an Orcas-style `StratifiedGroupKFold` re-split, if ever produced for direct methodology comparability) MUST be clearly labeled `[SECONDARY — in-distribution only, does not test cross-project generalization]` in every table/figure it appears in, and must never be blended with, averaged into, or substituted for the LOPO cross-fold summary (§8.3) in any headline claim.
- **L7.** *(new, from §4.4/Gate B)* Per-fold species eligibility (§6.2) is computed strictly from that fold's own train+val partitions, never peeking at test composition; a species's eligibility label is fixed before any model is fit or evaluated for that fold.

---

## 12. Calibration and thresholding

Unchanged in substance from Round 1 (per-species isotonic or Platt calibration fit on the validation split per fold; operating thresholds chosen on validation via F1-maximization or a fixed-recall target, reported alongside precision/recall at that threshold) — with one addition: for the single-label branch, `predict_proba`'s multinomial output is **already** a proper simplex (sums to 1 across classes) so per-class calibration must be interpreted with that constraint in mind (calibrating one class's probabilities in isolation can break the simplex property; if per-class calibration is needed, calibrate on the one-vs-rest decomposition of the multinomial output, not on the raw softmax vector, and document this explicitly wherever the single-label branch's calibrated probabilities are reported). `[RECOMMENDATION]`

---

## 13. Ablation matrix

| Axis | Levels | Purpose |
|---|---|---|
| Label branch | single-label (primary-annotation) / multilabel OvR | Directly answers Gate A's question with a measured performance delta, not just a rate |
| Embedding source | raw-SavedModel-direct / `perch_hoplite` wrapper | Validates §10's equivalence assumption empirically, not just by construction |
| Reporting granularity | window / segment | Quantifies how much smoothing segment-level pooling buys vs. its majority-vote or any-positive label noise |
| Perch model variant | `perch_v2` (1536-d) / `perch_8` a.k.a. original Perch (1280-d) | Carried from Round 1; cheap because embeddings are cached, informs whether the newer, larger model is worth its extra storage/compute |
| Few-shot k | {4,8,16,32} recordings-equivalent per eligible species, 5 seeds | Mirrors Orcas §8.3 exactly; run only on the single-label branch's eligible-species subset per fold |
| min_support threshold | {5,10,20,50} | Sensitivity of Gate B's eligible-species set and of macro-averaged metrics to this one configurable number |
| Overlap rule (`min_overlap_frac`) | {0.0 (any overlap), 0.25, 0.5} | Sensitivity of both the co-occurrence rate (Gate A) and the multilabel target itself to the boundary-overlap definition (flagged as a Round‑1 open question, `architect_pipeline_proposal.md` line 64) |

---

## 14. Reproducibility requirements

Unchanged from Round 1 (pinned `perch_v2` model artifact hash recorded in extraction manifest, `random_state` fixed for any stochastic step — few-shot sampling — deterministic `lbfgs` fits left unseeded per Orcas' own convention and documented as such, git commit SHA + input file SHA-256 recorded per run, config-driven not hardcoded). New: the Phase 0 audit report (`reports/species_phase0_audit.md`) itself is a required, versioned artifact — Gate A/B/C/D's measured values must be committed to the repo (or its designated non-gitignored report location) before Phase 3 training begins, so the branch decision is auditable after the fact, not just asserted in a design doc.

---

## 15. Phased milestones and acceptance criteria

**Phase 0 — Audit (no ML, no Perch dependency; target ≤1 day, matching all three Round‑1 drafts' estimate).**
Acceptance: `reports/species_phase0_audit.md` exists with Gates A–D's numbers, reviewed by the Inquisitor. Go/no-go: if Gate A's co-occurrence rate is ambiguous relative to the 5% threshold (e.g., within ±1pp with small per-project variance), escalate to a design discussion rather than auto-deciding — a borderline number should not be silently rounded into a confident branch choice.

**Phase 1 — Label pipeline.** Builds `species_labels_segmented_v4.csv` (multilabel, six-state) and `species_labels_primary_segmented_v4.csv` (single-label subset). Acceptance: row counts reconcile against Phase 0's numbers exactly (no silent drift between audit and label-build); a fail-loud assertion checks every fold's test rows' project matches that fold's held-out project (Round 1 pattern, retained).

**Phase 2 — Embedding extraction.** Acceptance: raw-SavedModel path produces 1536-d embeddings for a 200-window smoke test; the `perch_hoplite` equivalence cross-check (§10) passes at cosine similarity > 0.999 on the same 200 windows; full extraction completes for all 160,244 windows with a logged, bounded fallback-to-silence rate (<0.1%, `[RECOMMENDATION]`, else investigate before proceeding).

**Phase 3 — Linear probe training (the actual go/no-go).** Acceptance: both branches train without error on all 5 folds; the primary branch (decided by Gate A) beats the trivial baseline (per-species prevalence-only classifier) on macro-AP (multilabel) or macro-F1 (single-label) by a pre-registered margin `[RECOMMENDATION: +5pp]`; Gate B's eligible-species tables are consumed correctly (no metric computed for a structurally-unseen species).

**Phase 4 — Evaluation and reporting.** Acceptance: `reports/species_probe_results/` contains window-level, segment-level, and cross-fold-summary tables for both branches, the Perch-zero-shot secondary baseline, and the ablation matrix (§13) at least partially populated (embedding-source and reporting-granularity axes are cheap and must be complete; few-shot and min_support sweeps may be partial if compute-constrained, but must be disclosed as such).

---

## 16. Tests

- `tests/test_primary_annotation_reduction.py` — synthetic windows with known overlaps assert the largest-overlap-wins rule and the tie-break rule (Gate D) produce the expected single label.
- `tests/test_eligibility_gate.py` — synthetic per-fold support tables assert eligible/unseen/untestable classification matches §6.2's exact boundary conditions (`train_support == min_support` is eligible; `min_support - 1` is not).
- `tests/test_embedding_equivalence.py` — raw-SavedModel vs. `perch_hoplite` cosine-similarity cross-check (§10), run once per Perch model-artifact version, not on every CI run (network/Kaggle-dependent).
- `tests/test_window_geometry.py` — asserts `librosa.load(..., offset=window_start_sec, duration=5.0)` returns exactly `PERCH_WINDOW_SAMPLES` samples for interior windows and a correctly-flagged (not silently zero-padded-and-unflagged) short read for the last window of a file.
- `tests/test_leakage_join.py` — the fail-loud project-mismatch assertion from Phase 1 (Round 1 pattern), extended to also assert no `sound_id` appears in both a fold's train and val label rows.
- `tests/test_metric_masking.py` — asserts macro-AP/macro-F1 computations exclude Gate-B-ineligible species from both the numerator and the denominator (a species silently included with an undefined/`nan` score would corrupt the macro average silently).

---

## 17. Failure modes (new/updated for this round)

| Failure | Detection | Mitigation |
|---|---|---|
| Gate A's co-occurrence rate is measured near the 5% boundary and flips between minor code changes (e.g. crosswalk updates) | Re-run audit_phase0.py after any crosswalk/label change; diff against the committed report | Treat the branch decision as re-opened whenever the audit report changes materially; never hardcode "we chose multilabel" without a live, re-checkable number |
| A species is Gate-B-eligible in only 1 of 5 folds | Eligibility count column in per-species results (§8.3) | Flag visually (e.g., asterisk + footnote), never present alongside 5-fold-eligible species without the distinction |
| Raw-SavedModel and `perch_hoplite` embeddings diverge beyond tolerance | `test_embedding_equivalence.py` fails | Investigate before trusting either extraction path; do not silently pick one |
| Segment-level majority-vote hides real within-segment co-occurrence (Gate C not actually low) | Gate C's measured rate | If Gate C ≥ threshold, segment-level must use "any-positive" multilabel target, not majority vote — same logic as Gate A, applied one level down |
| `kagglehub` auth/network failure blocks model download | `download` CLI mode fails loudly (mirrors Orcas' `SystemExit`) | Vendor/cache the downloaded SavedModel directory as a durable artifact after first successful download (Round 1 mitigation, retained) |

---

## 18. Decision gates (consolidated)

1. **Gate A** (§6.1): co-occurrence rate vs. 5% → single-label vs. multilabel primacy. Both always computed.
2. **Gate B** (§6.2): per-fold × per-species support → eligible / structurally-unseen / untestable. Mandatory for both branches, every fold.
3. **Gate C** (§6.3): segment-level co-occurrence → majority-vote vs. any-positive at the segment reporting level.
4. **Gate D** (§6.4): primary-annotation tie rate → confirms the tie-break rule is a minor correction, not a dominant effect (if tie rate is high, the single-label reduction's labels are effectively arbitrary for a large fraction of windows, which would itself argue against using the single-label branch as primary regardless of Gate A).
5. **Phase 3 go/no-go** (§15): primary branch must beat the trivial prevalence baseline by the pre-registered margin before any further compute (fine-tuning, temporal modeling, additional ablations) is justified.

---

## 19. Convergence verdict

**NEW-IDEAS.**

The three Round‑1 drafts (mine, `architect_minimalist_proposal.md`, `architect_pipeline_proposal.md`) already converged, independently, on the core architecture this round reaffirms: multilabel-by-default target construction, six-state label schema to protect against poisoning every species' negative set with unresolved annotations, `sklearn.linear_model.LogisticRegression` one-vs-rest on frozen 1536-d Perch v2 embeddings, joining onto `folds_segmented_v4` verbatim rather than re-splitting, and an isolated Perch/TensorFlow dependency environment. Round 2 does not overturn any of that; it is not a redesign.

What is genuinely new, driven specifically by inspecting `orcas_dclde2026` at the user's request, and present in none of the three Round‑1 drafts:

1. **Raw-SavedModel + `kagglehub` extraction in place of `perch_hoplite`** (§10) — a disclosed, justified reversal of Round 1's unanimous choice, made for dependency-footprint and cross-repo-comparability reasons, with an explicit equivalence test as mitigation.
2. **Segment-level (10 s time-lapse snapshot) pooling as the correct PteroSet substitute for "recording-level" reporting** (§7.3, §4.5) — none of the Round‑1 drafts addressed recording/file-level aggregation at all; this round identifies that PteroSet's duty-cycled file structure makes whole-file pooling actively misleading and proposes a specific, smaller, acoustically-coherent substitute unit.
3. **A quantitative, pre-registered decision gate (Gate A, 5% co-occurrence threshold) deciding single-label-vs-multilabel primacy**, rather than Round 1's assumption of "multilabel by default" — Round 1 was correct that multilabel is the safe default, but this round is the first to specify exactly what measurement would justify running (and headlining) a single-label comparability arm instead, and answers the user's explicit "both, or which one" question with a concrete mechanism rather than an architect's judgment call.
4. **The LOPO-vs-`StratifiedGroupKFold` incompatibility (§4.3) and the resulting per-fold species-eligibility gate (Gate B, §6.2)** as a first-class, mandatory artifact — Round 1 mentioned "absent species in held-out projects" as an edge case; this round elevates it to a required per-fold table with three explicit categories (eligible/unseen/untestable) directly motivated by contrasting PteroSet's fixed project-holdout design against Orcas' seed-swept stratified splits.
5. **The primary-annotation ("largest-overlap-wins") single-label reduction algorithm** (§7.2) as the exact, specified PteroSet analogue of Orcas' `source_annotation_idx` join — a concrete algorithm, not present in Round 1, needed specifically to run the single-label branch at all.

---

## Appendix A — Verified-fact source log (Round 2 additions)

- `orcas_dclde2026/eval_perch_ecotype.py` — read in full (constants, `load_audio_segment`, `inspect_model_outputs`, `extract_embeddings`, `run_extraction`/`run_extraction_csv`, `run_unassigned_eval`, `aggregate_to_recordings`, `run_fewshot`, `run_fulldata`, `_compute_full_metrics`, `main`/argparse) — direct source read, this session.
- `orcas_dclde2026/reports/ecotype_classifier.md` §1–8 — read in full, direct source read, this session.
- `orcas_dclde2026/docs/experiments/log.md` — S1.4 and S2.3 entries read, direct source read, prior session turn (summarized above, re-cited here).
- `orcas_dclde2026/pip-requirements.txt` — grepped for `tensorflow`/`kaggle`/`librosa`/`scikit-learn` versions, direct source read, this session.
- `birds_bioacoustics/reports/window_counts_by_fold.md` — read in full, direct source read, this session (existing, already-computed annotation-density evidence cited in §4.1).
- `birds_bioacoustics/docs/design/round_01/architect_minimalist_proposal.md`, `architect_pipeline_proposal.md` — headers grepped, key sections (multilabel/single-label decisions, split logic, `perch_hoplite` usage) grepped and read in context, this session — sufficient to confirm the convergence claims in §19 without re-deriving their full content.

## Appendix B — Explicit unresolved items carried forward

- Exact numeric values for Gates A–D are **not yet measured** — this document specifies the protocol and thresholds, not the outcome. Running `species_probe/audit_phase0.py` is the immediate next action, before any embedding extraction.
- The 5% Gate‑A threshold, 10‑window Gate‑B `min_support`, and 0.1% Phase‑2 silence-fallback bound are all `[RECOMMENDATION]`-tagged, pre-registered defaults, intentionally exposed as CLI flags so they can be revised by evidence (the ablation matrix, §13, sweeps `min_support`; the audit script itself takes `--cooccurrence_threshold` as a flag) without editing code.
- Weights license for `perch_v2` remains `[UNRESOLVED FROM ROUND 1]` — re-verify against the live Kaggle model card before any publication-facing use, unchanged from Round 1's flag.

STATUS: DONE
