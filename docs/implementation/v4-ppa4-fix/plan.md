# Plan: v4 — re-apply PPA4 annotation fix on top of v3

## Context

During a CLAUDE.md documentation audit (2026-04-22), we discovered that `windows_mapping_4.0overlap_segmented_v3.json` has incorrect labels for 1 615 PPA4 windows: the v1→v2 annotation fix was not carried over into v3. All 34 280 PPA4 windows in v3 match v1 labels exactly (v1 positives = v3 positives = 7 334 vs. v2 positives = 8 944). v3 was already trained (`checkpoints_v3/`, `outputs_v3/`) before the regression was detected, so those results are on buggy PPA4 labels.

MAP1, PPA1, PPA2, PPA3 labels are unaffected. PPA1 window set in v3 is correctly re-segmented for the 1-second crossfade / 9-second stride (paper §Data Records) and should be kept.

**Source annotation timestamps**: `data/annotations_identification.json` and `data/annotations_species.json` are both dated 2026-03-10, predating v2 (Mar 19) and v3 (Apr 15). This is the primary root-cause hypothesis: the v2 fix was applied downstream of these JSONs and never propagated back, so v3's regeneration from source silently lost it.

## Objective

Produce `folds_segmented_v4` equal to v3 on MAP1/PPA1/PPA2/PPA3 and equal to v2 on PPA4 labels. Retrain CV. Adopt v4 as the shipping version.

## Acceptance criteria

- `windows_mapping_4.0overlap_segmented_v4.json`:
  - 160 244 windows total (same as v3)
  - PPA4 positives = 8 944 (matches v2), **not** 7 334 (= v3)
  - Windows in MAP1/PPA1/PPA2/PPA3 are identical in count and labels to v3
  - The specific 1 615 PPA4 windows flipped 0→1 in v1→v2 are also 0→1 in v4; the 5 flipped 1→0 in v1→v2 are also 1→0 in v4
- `data/folds_segmented_v4/fold_{0..4}_{MAP1,PPA1,PPA2,PPA3,PPA4}_segmented/` with `{train,val,test}_split.csv`
- `checkpoints_v4/fold_{0..4}/` trained with hyperparameters **mirroring v3 exactly**
- `outputs_v4/cv_results.csv`, `precision_recall_curves.png`, and comparison tables vs. v2 and v3

## Phase 1 — Root cause investigation

Must complete before writing any fix. Without root cause, a future v5 can re-regress.

1.1 Search repo and git history for any PPA4-specific fixup:
   - `git log --all -S 'PPA4'` and `git log --all -S '1615'`
   - `find . -type f \( -name '*ppa4*' -o -name '*parex*' -o -name '*fix*' \)` (case-insensitive)
   - Check for one-off notebooks or scripts that could have patched windows mapping

1.2 Compare source annotations to the known 1 615 fix windows:
   - Load `annotations_identification.json` (Mar 10)
   - Load v2 windows mapping, extract the 1 615 PPA4 windows with `label=1` where v1 had `label=0`
   - For each such window, check whether any annotation in the source JSON overlaps `[window.start, window.end]` with a PPA4 `sound_id`
   - Count how many of the 1 615 are "covered" by source vs. orphaned

1.3 If annotations JSON lacks them, check RAVEN label files in `data/labels_48khz/` for PPA4 (PAREX) recordings:
   - Match `sound_id` → RAVEN `.txt` file → grep for time ranges covering the 1 615 windows
   - Are the missing annotations present in the raw labels but filtered out by `data_reader.py`?

1.4 Verify `prepare_dataset.py --steps windows segment_windows` is deterministic and fully source-driven:
   - Read through `prepare_dataset.py` and `data/data_reader.py`
   - Check for any hard-coded PPA4 handling or external patch file references

1.5 **Decision gate** — classify root cause:
   - **Case A**: missing from `annotations_*.json` → port fix to source annotations
   - **Case B**: missing from RAVEN label files → fix labels, regenerate JSONs
   - **Case C**: present in source but pipeline filters them → fix pipeline bug

## Phase 2 — Pipeline fix (Option B: re-derive labels in `run_segment_windows`)

**Phase 1 result: Case C (pipeline caching bug).** See `phase1_findings.md`. `run_segment_windows` was copying labels verbatim from a stale Feb 16 unsegmented cache. Source `annotations_identification.json` already contains all 1 615 PPA4 annotations.

**Chosen fix**: re-derive labels inside `run_segment_windows` for each retained window, using the already-loaded `annotations_data`. Rationale: `run_segment_windows` is the function that produces each versioned file (v2/v3/v4/…); making it label-self-sufficient means every version automatically reflects the annotations JSON at time-of-derivation, regardless of upstream cache state. The unsegmented cache (`windows_mapping_4.0overlap.json`) is no longer label-authoritative — no backup or invalidation needed.

2.1 Patch `prepare_dataset.py` `run_segment_windows()`:
   - Build `sound_to_anns: dict[sound_id, list[(t_min, t_max)]]` right after `annotations_data` is loaded.
   - Inside the fits-branch (after `w_copy = dict(w)` and `w_copy["dataset"] = ...`), overwrite `w_copy["label"]` using overlap against `sound_to_anns` with proper samples→seconds conversion via `w_copy["sample_rate"]`.

2.2 Dry-run verification (no file write, no training): simulate the patched label logic on the existing v3 window set and confirm:
   - Total positives = 35 073 (= v3 + 1 615 − 5)
   - PPA4 positives = 8 944 (matches v2)
   - MAP1/PPA1/PPA2/PPA3 positives unchanged from v3
   - Flip counts: 1 615 PPA4 (0→1), 5 PPA4 (1→0), 0 elsewhere

**Status: DONE (2026-04-23)** — patch applied; dry-run matched all expected counts exactly.

## Phase 3 — Build v4 artifacts

3.1 Version suffix: `run_segment_windows(config, windows, version="v3")` and `run_splits(config, windows, folds_subdir="folds_segmented_v3")` default to v3. For v4 either:
   - Change the defaults in `prepare_dataset.py` to "v4" / "folds_segmented_v4", OR
   - Add a `--version` CLI flag to the main pipeline and thread it through.

   Suggested: add `--version` flag (minimal, non-destructive to existing v3 defaults if someone re-runs).

3.2 Regenerate:
   ```
   python prepare_dataset.py --config data/config.yaml --steps segment_windows splits --version v4
   ```
   (`windows` step not needed — the unsegmented cache is still valid as window-set input; the patched `run_segment_windows` re-derives labels regardless of input labels.)

3.3 **Verification gate** (mandatory before Phase 4) — load `windows_mapping_4.0overlap_segmented_v4.json` and confirm:
   - Total windows = 160 244
   - Total positives = 35 073
   - PPA4 positives = 8 944
   - MAP1/PPA1/PPA2/PPA3 positives unchanged from v3
   - Diff v4 vs. v3 on `(sound_id, start, end)`: identical window sets (no PPA1 re-churn)
   - Label flips vs. v3: 1 615 PPA4 (0→1), 5 PPA4 (1→0), 0 elsewhere

3.4 Spectrograms: window identity is unchanged from v3, so the existing `.npy` cache covers v4. No `spectrograms` step needed.

3.5 Fold directories `data/folds_segmented_v4/fold_{0..4}_{MAP1,PPA1,PPA2,PPA3,PPA4}_segmented/` created by the `splits` step; CSVs reflect v4 labels.

## Phase 4 — Cross-validation training

4.1 Hyperparameters: **mirror v3 exactly.** No changes to:
   - backbone (`resnet18`), num_classes (2), batch size, epochs
   - augmentation flags (`--use_specaug`, MixUp)
   - optimizer, learning rate schedule, backbone freeze strategy
   - random seeds

   v4 ↔ v3 metric delta must isolate the PPA4 annotation fix and nothing else.

4.2 Command:
   ```
   python train.py --config data/config.yaml \
       --cross_validation --fold_dir data/folds_segmented_v4 \
       --ckpt_dir checkpoints_v4
   ```
   (Previous versions of `train.py` hardcoded `ckpt_dir = "checkpoints/fold_N"`; a new `--ckpt_dir` CLI flag was added to parameterize the output root, defaulting to `checkpoints` for backward compatibility.)

4.3 All 5 folds must be retrained — PPA4 is present in train/val for folds 0–3 and held out as test for fold 4, so every fold's data changes. Cannot cherry-pick.

4.4 Monitor per-fold val loss; flag any fold whose training diverges significantly from v3's loss curve for that fold (would indicate a broader issue beyond the annotation fix).

## Phase 5 — Evaluation and comparison

5.1 Generate v4 results:
   ```
   python plot_cv_results.py --fold_dir data/folds_segmented_v4 \
       --config data/config.yaml --checkpoint_dir checkpoints_v4
   ```

5.2 Three-way comparison v2 / v3 / v4 using the `comparison_*.csv` tooling already present in `outputs_v3/`:
   - v4 vs. v3: isolates PPA4 annotation fix
   - v4 vs. v2: isolates PPA1 crossfade re-segmentation (both have correct PPA4)
   - Per-fold and per-project metrics

5.3 Write `docs/implementation/v4-ppa4-fix/results.md` with:
   - Root-cause finding from Phase 1
   - Fix applied in Phase 2 (Case A / B / C + details)
   - Quantitative impact table: fold × version × metric (AP, AUROC, precision@recall)
   - Decision: is v4 the new shipping version for publication?
   - If Case C, note the regression test added

## Phase 6 — Documentation and cleanup

6.1 Update `CLAUDE.md` Dataset Versions table:
   - Fill v4 row: total windows = 160 244, positives = ~35 073 (v3 + 1 615 − 5)
   - Mark v4 as "current shipped"

6.2 Update "Current shipped experiment" pointer in CLAUDE.md: `outputs_v4/` + `checkpoints_v4/`.

6.3 Annotate v3 row: "superseded by v4; retained for reference only (PPA4 labels incorrect)".

6.4 Merge branches into master:
   - `docs/clarify-dataset-versions` (CLAUDE.md clarifications — though CLAUDE.md is gitignored, any other changes on the branch)
   - v4 work branch (new branch for the fix + training)

## Risks and alternatives considered

- **Alternative: skip v4, publish v2.** v2 lacks PPA1 crossfade correction, so its PPA1 fold metrics are on misaligned segment boundaries. Quantify impact in Phase 5 before committing.
- **Alternative: patch `windows_mapping_v4.json` directly without fixing source.** Faster but perpetuates the bug. Only acceptable if Phase 1 finds the source-of-truth fix is impractical; must be flagged as tech debt.
- **Retraining cost**: 5 folds of ResNet18 on ~160 k spectrograms × epochs. Budget GPU hours before starting Phase 4.
- **Config mechanism**: the current `config.yaml` may not parameterize output version; Phase 3.1 may require a small script change.
- **Fold naming comparison with v2**: v2 folds use original project codes (GEOPARK_PUTUMAYO_2024, GeoPark_II_T2/T3/T4_2024-25, PAREX). Mapping to v4 naming (MAP1, PPA1–PPA4) per CLAUDE.md; align before side-by-side metric tables.

## Log

- **2026-04-22**: plan created; Phase 1 starting.
- **2026-04-23**: Phase 1 complete — root cause = Case C (pipeline caching bug in `prepare_dataset.py:107-111`). See `phase1_findings.md`.
- **2026-04-23**: Phase 2 complete — applied Option B patch: `run_segment_windows` now re-derives labels from current annotations instead of copying from stale input. Dry-run on v3 window set produced exactly the expected v4 state (PPA4 positives 7 334 → 8 944, zero deltas elsewhere).
- **2026-04-23**: Phase 3 complete — added `--version` CLI flag threaded through `run_segment_windows`, `load_segmented_windows_if_exists`, and `run_splits`. Ran `python prepare_dataset.py --config data/config.yaml --steps segment_windows splits --version v4`. All 9 verification-gate checks passed.
- **2026-04-23**: Phase 4 complete — added `--ckpt_dir` CLI flag to `train.py`. User ran CV training on `folds_segmented_v4` with `--ckpt_dir checkpoints_v4`; 5 folds trained.
- **2026-04-23**: Phase 5 complete — ran `plot_cv_results.py` for v4; built project-aligned 3-way comparison (`outputs_v4/version_comparison_by_project.csv`). PPA4 AUPRC +0.0512 vs. v3; other projects within training-variance noise. See `results.md` for full analysis.
- **2026-04-23**: Phase 6 complete — CLAUDE.md Dataset Versions table updated: v4 marked as current shipping version; v3 marked as superseded (buggy PPA4 labels). Changelog describes pipeline fix and new CLI flags (`--version`, `--ckpt_dir`).
