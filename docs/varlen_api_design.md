# Varlen/Batch FA2 Runtime Design (NanoVLLM)

## Human Readable

### Goal
Keep both batch-view and varlen FA2 paths available behind one runtime mode switch, while preserving a small-project API surface.

### Design Choices
- Runtime mode switch: `NANOVLLM_FA2_MODE`.
- Supported mode values:
  - `varlen_official`
  - `varlen_man`
  - `batch_official`
  - `batch_man`
  - `batch_debug`
- Legacy typo values (`varlen_offical`, `batch_offical`) are no longer accepted.
- Keep only inference-focused arguments: `q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_table, causal, softmax_scale`.
- Exclude advanced options: dropout/window/alibi/softcap/deterministic/attn-prob outputs.
- `block_table` is required for varlen modes.
- If no prefix cache is used, pass an empty CUDA `int32` tensor with shape `(batch, 0)`.

### Current Implementation State
- `batch_official`: batch-view + official flash-attn batch kernel path.
- `batch_man`: batch-view + handwritten torch-bind `fa2_batch_fwd` path.
- `batch_debug`: debug mode; run both `batch_man` and `batch_official`, then assert diff within threshold.
- `varlen_official`: varlen API path via `fa2_varlen_fwd_official` (flash-attn varlen reference).
- `varlen_man`: handwritten CUDA varlen path via torch extension symbol `fa2_varlen_fwd`.
- `varlen_man` currently applies Python-side max-seqlen align-to-64 hotfix before kernel launch.

### Mode Matrix
| Mode | Function | Current backend |
|---|---|---|
| `varlen_official` | `_run_cuda_varlen_fa2_official` | `flash_attn_varlen_func` placeholder via wrapper |
| `varlen_man` | `_run_cuda_varlen_fa2_man` | handwritten torch bind `fa2_varlen_fwd` with temporary max-seqlen align-to-64 hotfix and `[WARNING][FA2_VARLEN_MAN_PAD64]` |
| `batch_official` | `_run_cuda_batch_fa2_official` | `flash_attn_func` per-sequence batch-view |
| `batch_man` | `_run_cuda_batch_fa2_man` | handwritten torch bind `fa2_batch_fwd` per-sequence batch-view with temporary Python pad-to-64 hotfix and `[WARNING][FA2_BATCH_MAN_PAD64]` |
| `batch_debug` | `_run_cuda_batch_fa2_debug` | run `batch_man` + `batch_official`, compare varlen outputs with `allclose(atol=1e-2, rtol=1e-2)`, raise on mismatch |

### API
```python
def fa2_varlen_fwd(
    q: torch.Tensor,                 # [total_q, nheads_q, headdim]
    k: torch.Tensor,                 # [total_k, nheads_kv, headdim]
    v: torch.Tensor,                 # [total_k, nheads_kv, headdim]
    *,
    cu_seqlens_q: torch.Tensor,      # [batch+1], int32, cuda
    cu_seqlens_k: torch.Tensor,      # [batch+1], int32, cuda
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_table: torch.Tensor,       # [batch, num_blocks], int32, cuda (required)
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor                    # [total_q, nheads_q, headdim]
```

### Constraints
- CUDA only; `q/k/v` must be fp16 or bf16.
- `cu_seqlens_*` must be rank-1 int32.
- `block_table` must be rank-2 int32 and batch dimension must match `len(cu_seqlens_q)-1`.

## Agent Readable

```yaml
api_name: fa2_runtime_modes
layer: nanovllm.layers.prefill_attention
status:
  mode_switch: implemented
  varlen_cpp_symbol: implemented
  handwritten_varlen_cuda: implemented
mode_env:
  name: NANOVLLM_FA2_MODE
  values:
    - varlen_official
    - varlen_man
    - batch_official
    - batch_man
    - batch_debug
  default_resolution:
    - if NANOVLLM_FA2_MODE is set: use it
    - else if NANOVLLM_FA2_USE_TORCH_EXT==1: varlen_official
    - else: batch_official
inputs:
  q:
    shape: [total_q, nheads_q, headdim]
    dtype: [float16, bfloat16]
    device: cuda
  k:
    shape: [total_k, nheads_kv, headdim]
    dtype: [float16, bfloat16]
    device: cuda
  v:
    shape: [total_k, nheads_kv, headdim]
    dtype: [float16, bfloat16]
    device: cuda
  cu_seqlens_q:
    shape: [batch_plus_1]
    dtype: int32
    device: cuda
  cu_seqlens_k:
    shape: [batch_plus_1]
    dtype: int32
    device: cuda
  max_seqlen_q:
    type: int
  max_seqlen_k:
    type: int
  block_table:
    shape: [batch, num_blocks]
    dtype: int32
    device: cuda
    required: true
    empty_when_no_prefix: true
  causal:
    type: bool
  softmax_scale:
    type: float
outputs:
  out:
    shape: [total_q, nheads_q, headdim]
routing:
  varlen_official: flash_attn_varlen_func_placeholder
  varlen_man: handwritten_fa2_varlen_fwd_with_python_max_seqlen_align64_hotfix
  batch_official: flash_attn_func_batch_view
  batch_man: handwritten_fa2_fwd_batch_view_with_python_pad64_hotfix
  batch_debug: compare_batch_man_vs_batch_official_and_assert_allclose
  target_varlen_man: implemented
excluded_features:
  - dropout
  - return_attn_probs
  - window_size
  - alibi_slopes
  - softcap
  - deterministic
```
