# Test Inventory

This file tracks the FA2-related test suites and what they validate.

## Coverage Map

| Test file | Category | Key checks |
|---|---|---|
| `tests/test_prefill_attention_mode_routing.py` | Routing / mode switch | `NANOVLLM_FA2_MODE` dispatch for `varlen_official`, `varlen_man`, `batch_official`, `batch_man`; varlen block_table normalization contract |
| `tests/test_varlen_api_contract.py` | Varlen API contract | `fa2_varlen_fwd` argument validation (`block_table`, batch dim, dtypes) and output match vs `flash_attn_varlen_func` on empty block table |
| `tests/test_prefill_attention_batch_view.py` | Batch-view correctness | varlen<->padded roundtrip, backend equivalence, padding invariance, causal masking behavior |
| `tests/test_batch_man_pad64_hotfix.py` | Batch-man hotfix | non-64 seqlen triggers pad-to-64 + `[WARNING][FA2_BATCH_MAN_PAD64]`; aligned seqlen does not warn |
| `tests/test_flash_attn_correctness.py` | Numerical correctness | SDPA vs flash-attn numerical agreement across multiple `(B, H, N, D)` shapes |

## Shape-focused checks

- `tests/test_flash_attn_correctness.py`
  - `(B, H, N, D) = (1,8,64,64), (1,8,128,64), (1,8,192,64), (1,8,512,64), (1,8,1024,64)`
- `tests/test_prefill_attention_batch_view.py`
  - mixed sequence lengths (`q_lens/k_lens`) for varlen packing and reconstruction
  - GQA-like head expansion path (`q_heads != kv_heads`)
- `tests/test_batch_man_pad64_hotfix.py`
  - current explicit contract cases: unaligned `N=63` (pad+warning), aligned `N=64` (no warning)
  - note: this file does **not** yet include explicit `batch_man` numerical checks for `N=128/192/512`

## Correctness-focused checks

- Batch-view backend result matches flash-attn baseline (`allclose` tolerances)
- Varlen API empty-block-table path matches flash-attn varlen baseline
- Causal masking blocks future-token influence in batch-view flow
- Mode routing chooses the intended implementation branch
- batch_man temporary pad64 hotfix emits warning on unaligned shapes
- flash-attn numerical agreement includes `N=64/128/192/512/1024` (separate from batch_man hotfix contract tests)

## Run commands

```bash
python3 -m pytest -q tests/test_prefill_attention_mode_routing.py
python3 -m pytest -q tests/test_varlen_api_contract.py
python3 -m pytest -q tests/test_prefill_attention_batch_view.py
python3 -m pytest -q tests/test_batch_man_pad64_hotfix.py
python3 -m pytest -q tests/test_flash_attn_correctness.py
```
