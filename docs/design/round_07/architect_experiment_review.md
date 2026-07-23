# Round 7 — Clean Convergence Review

**Author**: architect-experiment (Round 7 review)
**Input read**: `docs/design/perch2_species_linear_probe_plan.md`, full current revision, re-read end to end this round.

**Task**: search only for genuinely new load-bearing scientific flaws; no repeats, optional enhancements, or style comments.

---

## New finding: `pool_manifest.json`'s actual `identity_hash_inputs` schema was not updated to match the pool-embedding hash's own stated definition

Since Round 6, the "Cache invalidation" section's definition of the **pool embedding hash** was expanded. It now reads (lines 319-325):

> "a function of `windows_mapping_4.0overlap_segmented_v4.json`'s SHA-256, **the generated `pool.csv` content SHA-256, the `build_embedding_pool_csv.py` source hash and relevant config values**, the vendored model directory's content hash, and the extraction params."

The `fold_manifest.json` example was updated to match this prose:
```json
"pool_embedding_hash": "<hash of windows-json + pool.csv + pool-builder source/config + model-dir + extraction params>"
```
And `test_cache_hash_independence.py` (test #6) was correspondingly updated to require: "changing `pool.csv` or `build_embedding_pool_csv.py` must change `pool_embedding_hash`."

However, the **only place in the document that defines the actual JSON schema `extract_embeddings_pteroset.py` writes** — the `pool_manifest.json` example in the "Failure handling" section (lines 118-122) — was **not updated to match**:

```json
"identity_hash_inputs": {
  "windows_mapping_json_sha256": "<sha256 of windows_mapping_4.0overlap_segmented_v4.json>",
  "model_local_dir_sha256": "<sha256 over a sorted listing of the vendored SavedModel dir's files>",
  "extraction_params": {"target_sample_rate": 32000, "window_sec": 5.0, "target_peak": 0.25, "batch_size": 64}
}
```

This block still lists only the three inputs that were sufficient under the *old* (Round 6) definition of the pool embedding hash. It has no `pool_csv_sha256` key, no `build_embedding_pool_csv.py`-source-hash key, and no place to record "relevant config values" for that script. If `extract_embeddings_pteroset.py` is implemented literally against this schema (the only schema the document gives for what that script actually writes), the resulting `pool_embedding_hash` will **not** change when `pool.csv` or `build_embedding_pool_csv.py` changes — directly contradicting the Cache invalidation section's own stated behavior and failing test #6 as specified.

**Why this is load-bearing, not cosmetic**: this is the exact failure mode the two-hash-lineage mechanism exists to prevent (Round 3 Finding 3 / Round 4-5's confirmed resolution) — a change to an upstream data-preparation step (here, `build_embedding_pool_csv.py`, e.g. a fix to how identity columns are derived, or which windows are included/excluded from the pool) silently fails to invalidate `pool_emb_v2.npz`, so a stale pool could be reused and treated as valid after the very script that produces its input has changed. The fix that was intended to broaden the hash's coverage (adding `pool.csv` and the pool-builder script to its inputs) was only written into the prose description and the downstream `fold_manifest.json` example, not into the one schema block that actually defines what gets hashed at the point of computation.

**Minimal fix**: add `pool_csv_sha256` and `pool_builder_source_sha256` (plus whatever "relevant config values" resolves to, e.g. a hash of the CLI args or `data/config.yaml`'s relevant section) as keys inside `identity_hash_inputs` in the `pool_manifest.json` schema shown in "Failure handling", so that block matches the Cache invalidation section's prose and `fold_manifest.json`'s example verbatim. No new mechanism, phase, or artifact is required — this is a one-block schema correction to make the document internally consistent with itself, the same class of fix as Round 5's ownership-attribution correction, now recurring on the cache-hash-inputs schema instead.

This is the only new issue found; no other section shows a new inconsistency this round.

---

## Verdict

**NEEDS-MORE** — one new, narrow, load-bearing issue: the `pool_manifest.json` schema's `identity_hash_inputs` block does not contain the two additional inputs (`pool.csv` content hash, `build_embedding_pool_csv.py` source hash) that the Cache invalidation section, `fold_manifest.json`'s example, and test #6 all already require the pool embedding hash to depend on. The fix is a one-block schema correction, not a design change.

STATUS: DONE
