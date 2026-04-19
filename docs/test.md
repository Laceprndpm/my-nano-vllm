# Test Inventory

This file tracks the FA2-related test suites and what they validate.

## Coverage Map

| Test file | Category | Key checks |
|---|---|---|
| `tests/test_prefill_attention_mode_routing.py` | Routing / mode switch | `NANOVLLM_FA2_MODE` dispatch for `varlen_official`, `varlen_man`, `batch_official`, `batch_man`; varlen block_table normalization contract |
| `tests/test_varlen_api_contract.py` | Varlen API contract | `fa2_varlen_fwd` argument validation (`block_table`, batch dim, dtypes) and output match vs `flash_attn_varlen_func` on empty block table |
| `tests/test_prefill_attention_batch_view.py` | Batch-view correctness | varlen<->padded roundtrip, backend equivalence, padding invariance, causal masking behavior |
| `tests/test_flash_attn_correctness.py` | Numerical correctness | SDPA vs flash-attn numerical agreement across multiple `(B, H, N, D)` shapes |

## Shape-focused checks

- `tests/test_flash_attn_correctness.py`
  - `(B, H, N, D) = (1,8,128,64), (1,8,512,64), (1,8,1024,64)`
- `tests/test_prefill_attention_batch_view.py`
  - mixed sequence lengths (`q_lens/k_lens`) for varlen packing and reconstruction
  - GQA-like head expansion path (`q_heads != kv_heads`)

## Correctness-focused checks

- Batch-view backend result matches flash-attn baseline (`allclose` tolerances)
- Varlen API empty-block-table path matches flash-attn varlen baseline
- Causal masking blocks future-token influence in batch-view flow
- Mode routing chooses the intended implementation branch

## Run commands

```bash
python3 -m pytest -q tests/test_prefill_attention_mode_routing.py
python3 -m pytest -q tests/test_varlen_api_contract.py
python3 -m pytest -q tests/test_prefill_attention_batch_view.py
python3 -m pytest -q tests/test_flash_attn_correctness.py
```
