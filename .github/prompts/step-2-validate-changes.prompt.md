---
agent: 'agent'
description: 'Stage 2 — verify every documented change against the live source tree and quarantine the ones that are no longer valid into obsolete/'
---

# Stage 2 — Determine which changes are still valid

You are validating stage 1's output against the **live source tree**. This is **stage 2 of 4**.
Docs are a hint, not proof: an optimization recorded as KEPT may have been overwritten by the
upstream merge, refactored away, or made redundant by a later commit. Find the truth in code.

Read `docs/optimization-documentation/README.md` first for the ID namespace and rules.

## Inputs

- `docs/optimization-documentation/manifest.json` — the change inventory to validate.
- `docs/optimization-documentation/files-changed/**` — the before/after records.
- The live source tree on the current branch.

## Hard rules

1. **Do not modify any `.py` file.** You write only under `docs/optimization-documentation/`.
2. **Code wins.** If the live tree contradicts the docs, the live tree is correct. Record the
   contradiction — do not quietly rewrite history.
3. **Never delete.** Invalid changes are *moved* to `obsolete/`, and the original stays
   documented with its reason. Nothing is erased.
4. **No benchmarks.** Validity here is structural (is the code present and reachable), not
   performance-based. A change can be structurally valid and still not worth doing — that is
   Stage 3's job.

## Procedure

For every change in `manifest.json`:

1. **Locate the live code.** Open the module's live file and read the named function in full.
   Do not trust line numbers from the docs — they drift. Search by symbol name.
2. **Classify the "after" state:**
   - `VALID` — the optimized code is present as documented.
   - `SUPERSEDED` — the intent survives but the code was rewritten upstream; the documented
     snippet no longer matches. Record what the live code actually does.
   - `REVERTED` — the live code matches the documented **before** snippet, i.e. the change is
     gone.
   - `NOT_FOUND` — the function, file, or symbol no longer exists.
   - `UNVERIFIED` — you could not determine it. Do not guess; say so.
3. **Check reachability.** A change can be present but dead. Confirm the code path is actually
   executed under the user's config (`batch_size 1`, `cache_text_embeddings: true`,
   `timestep_type: linear`, `flip_x: false`, bf16, `low_vram` quantized). Several fixes are
   known to be dormant under this config (K18, K21) — that is fine, mark them `dormant: true`
   but keep them; correctness fixes stay valid even when dormant.
4. **`X2` vs `K9` — pre-resolved, just confirm.** `X2` was never applied; `K9` is the live
   implementation. Verify the fp32 integration cast is still present in the CFG loop of
   `extensions_built_in/diffusion_models/krea2/src/pipeline.py`
   (`latents = latents + (tprev - tcurr) * v.to(torch.float32)`). If present, close the item
   as resolved. If it has disappeared, that is a **new discrepancy** — report it in
   `manifest.json` under `"discrepancies"`; do not assume `X2` was retro-applied.
5. **Placeholder scan on legacy `C` sources.** Before accepting any `C` verdict, scan
   `docs/optimization/README.md` for placeholder tokens (`X.XX`, `[detailed analysis of
   results]`, `PENDING / REVERTED / INCONCLUSIVE`). A `C` change whose only evidence is a
   placeholder becomes `UNVERIFIED`, not `valid`.
6. **Quarantine.** For `REVERTED`, `NOT_FOUND`, and `SUPERSEDED`: move the change file to
   `obsolete/<module-path>/<ID>-<slug>.md` and write `obsolete/<module-path>/OBSOLETE.md`
   entries.
   - `REVERTED` / `NOT_FOUND` → removed from the active set.
   - `SUPERSEDED` → **keep in the active set** with `"status": "superseded"`, because the
     optimization intent may still be re-appliable. Flag it loudly for Stage 3.
7. **Update `manifest.json`** (see schema below). Do not rewrite stage-1 fields you did not
   verify; carry them forward.

## Known live anchors (verify, do not assume)

Confirmed present when this prompt was written — re-check, the tree may have moved:

| ID | Anchor to grep for | File |
|----|--------------------|------|
| K16 | `def ropeapply` applying rotation in `xq.dtype` | `krea2/src/mmdit.py` |
| K14 | `_padlen = (-combined.shape[1]) % 32` | `krea2/src/mmdit.py` |
| K10 | `fused_context` param on `forward` / `predict_velocity` | `krea2/src/mmdit.py`, `pipeline.py` |
| K9 | `lat_d = latents.to(dtype)` cast once per CFG step | `krea2/src/pipeline.py` |
| K6 | `def _cache_vae_norm_constants`, `_vae_latents_mean` | `krea2/krea2.py` |
| K1 | per-image loop in `encode_images` | `krea2/krea2.py` |
| K5 | `timestep.to(self.device_torch, dtype=self.torch_dtype)` | `krea2/krea2.py` |
| K3+K19 | vectorized fast path **and** ragged `for` branch | `krea2/src/pipeline.py` |
| K4 | `checkpoint(self._forward, x, mask)` in fusion blocks | `krea2/src/mmdit.py` |
| K18 | `(len(base) - 1) - torch.searchsorted(torch.flip(...))` | `toolkit/samplers/custom_flowmatch_sampler.py` |
| K20 | `def _devices_match`, used by `to_device_if_needed` | `extensions_built_in/sd_trainer/SDTrainer.py` |
| K21 | restored `copy.deepcopy(file_item)` in the `flip_x` block | `toolkit/data_loader.py` |

Note on K4 vs K17: `checkpoint(...)` calls at the fusion blocks are K4 (kept). The DiT block
call carrying `use_reentrant=False` belongs to the K17 lineage, which was **reverted** — if you
find `use_reentrant=False` on the fusion blocks, that is a discrepancy worth reporting.

## Output

### `obsolete/<module-path>/OBSOLETE.md`

```markdown
# Obsolete changes — <module path>

| ID | Title | Status | Reason | Verified against |
|----|-------|--------|--------|------------------|

## <ID>: <title>
**Status:** REVERTED / NOT_FOUND / SUPERSEDED
**Documented after:** `<quoted snippet>`
**Live code now:** `<what is actually there, quoted>`
**Why obsolete:** <one paragraph>
**Re-appliable?** <yes/no + why>
```

### `manifest.json` updates

Set `"stage": 2`. Per change, add:

```json
"K14": {
  "...carried forward from stage 1...": "",
  "status": "valid",
  "verified": true,
  "verified_against": "extensions_built_in/diffusion_models/krea2/src/mmdit.py",
  "live_anchor": "_padlen = (-combined.shape[1]) % 32",
  "dormant": false,
  "notes": ""
}
```

`status` ∈ `valid` | `superseded` | `obsolete` | `unverified`.
Add `"obsolete_moved": ["K7", ...]` and resolve `"ambiguous"` with findings.

## Handoff to next step

Print: total validated / superseded / obsolete / unverified, the list of files moved to
`obsolete/`, and the `X2`-vs-`K9` finding. Stage 3 analyzes only changes whose status is
`valid` or `superseded`.
