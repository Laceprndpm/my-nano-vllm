# Test Inventory

This file tracks the FA2-related test suites and what they validate.

## Coverage Map

| Test file | Category | Key checks |
|---|---|---|
| `tests/test_prefill_attention_mode_routing.py` | Routing / mode switch | `NANOVLLM_FA2_MODE` dispatch for `varlen_official`, `varlen_man`, `varlen_debug`, `batch_official`, `batch_man`, `batch_debug`; varlen block_table normalization; fallback boundary checks (invalid mode no-fallback, runtime error fallback on/off, debug no-fallback) |
| `tests/test_batch_man_correctness.py` | Batch-man correctness | handwritten batch kernel vs SDPA across shape matrix, dtype matrix (`fp16`/`bf16`) |
| `tests/test_batch_debug_mode.py` | Batch-debug behavior | debug mode runs `batch_man` + `batch_official`, returns man output on close match, raises on diff, rejects non-empty `block_table` |
| `tests/test_varlen_debug_mode.py` | Varlen-debug behavior | debug mode runs `varlen_man` + `varlen_official`, returns man output on close match, raises on diff/shape mismatch, and validates varlen_debug mode dispatch |
| `tests/test_varlen_api_contract.py` | Varlen API contract | `fa2_varlen_fwd` argument validation (`block_table`, batch dim, dtypes) and output match vs `flash_attn_varlen_func` on empty block table |
| `tests/test_varlen_man_correctness.py` | Varlen-man correctness | handwritten varlen kernel vs reference on dense/paged paths across `head_dim={64,128}` and dtype matrix (`fp16`/`bf16`) |
| `tests/test_varlen_man_pad64_hotfix.py` | Varlen-man hotfix | non-64 max seqlen triggers align-to-64 + `[WARNING][FA2_VARLEN_MAN_PAD64]`; aligned max seqlen does not warn |
| `tests/test_prefill_attention_batch_view.py` | Batch-view correctness | varlen<->padded roundtrip, backend equivalence, padding invariance, causal masking behavior |
| `tests/test_batch_man_pad64_hotfix.py` | Batch-man hotfix | non-64 seqlen triggers pad-to-64 + `[WARNING][FA2_BATCH_MAN_PAD64]`; aligned seqlen does not warn |
| `tests/test_prefill_attention_nvtx.py` | NVTX annotation | `NANOVLLM_NVTX` off/on behavior, batch/varlen FA2-call range names, and push/pop exception safety |
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
- `tests/test_batch_man_correctness.py`
  - `dtype in {fp16, bf16}`, `(B, H, N, D) = (1,8,64,64), (1,8,128,64), (1,8,192,64), (1,8,512,64)`

## Correctness-focused checks

- Batch-view backend result matches flash-attn baseline (`allclose` tolerances)
- Batch-man kernel matches SDPA baseline across fp16/bf16 correctness matrix
- Varlen API empty-block-table path matches flash-attn varlen baseline
- Varlen-man kernel matches reference on dense/paged paths across fp16/bf16
- Causal masking blocks future-token influence in batch-view flow
- Mode routing chooses the intended implementation branch
- Invalid/typo mode raises directly (no fallback), even when fallback switch is enabled
- Runtime backend `RuntimeError` falls back only when fallback switch is enabled
- Debug mode runtime/assertion failures do not fallback (`batch_debug`/`varlen_debug` hard-fail)
- batch_man temporary pad64 hotfix emits warning on unaligned shapes
- varlen_man temporary max-seqlen align hotfix emits warning on unaligned shapes
- batch_debug compares handwritten and official batch outputs and fails fast on mismatch
- varlen_debug compares handwritten and official varlen outputs and fails fast on mismatch
- flash-attn numerical agreement includes `N=64/128/192/512/1024` (separate from batch_man hotfix contract tests)

## Run commands

```bash
python3 -m pytest -q tests/test_prefill_attention_mode_routing.py
python3 -m pytest -q tests/test_batch_man_correctness.py
python3 -m pytest -q tests/test_varlen_debug_mode.py
python3 -m pytest -q tests/test_varlen_api_contract.py
python3 -m pytest -q tests/test_varlen_man_correctness.py
python3 -m pytest -q tests/test_varlen_man_pad64_hotfix.py
python3 -m pytest -q tests/test_prefill_attention_batch_view.py
python3 -m pytest -q tests/test_batch_man_pad64_hotfix.py
python3 -m pytest -q tests/test_prefill_attention_nvtx.py
python3 -m pytest -q tests/test_batch_debug_mode.py
python3 -m pytest -q tests/test_flash_attn_correctness.py
```

## NCU + NVTX Usage

- NVTX is off by default. Enable with `NANOVLLM_NVTX=1`.
- FA2 call-level ranges:
  - `prefill.batch_official.fa2_call`
  - `prefill.batch_man.fa2_call`
  - `prefill.varlen_official.fa2_call`
  - `prefill.varlen_man.fa2_call`

Profile handwritten batch path only:

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_man \
ncu --target-processes all --nvtx --nvtx-include "prefill.batch_man.fa2_call/" \
python3 example.py
```

Profile official batch path only:

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_official \
ncu --target-processes all --nvtx --nvtx-include "prefill.batch_official.fa2_call/" \
python3 example.py
```

Profile handwritten varlen path only:

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=varlen_man \
ncu --target-processes all --nvtx --nvtx-include "prefill.varlen_man.fa2_call/" \
python3 example.py
```
