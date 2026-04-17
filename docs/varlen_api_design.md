# Varlen API Design (NanoVLLM)

## Human Readable

### Goal
Add a minimal varlen forward API at Python + extension boundary, aligned with flash-attn2 core semantics, but intentionally small for this project.

### Design Choices
- Keep only inference-focused arguments: `q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_table, causal, softmax_scale`.
- Exclude advanced options: dropout/window/alibi/softcap/deterministic/attn-prob outputs.
- `block_table` is required in the API contract.
- If no prefix cache is used, pass an empty CUDA `int32` tensor with shape `(batch, 0)`.

### Current Implementation State
- Python varlen API is implemented and wired into prefill torch-ext path.
- C++ extension symbol `fa2_varlen_fwd` exists as a placeholder entrypoint.
- Runtime varlen execution currently routes to `flash_attn_varlen_func` as a placeholder while handwritten CUDA varlen kernel is pending.

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
api_name: fa2_varlen_fwd
layer: nanovllm.layers.cuda_fa2_torch_ext
status:
  python: implemented
  cpp_symbol: implemented_placeholder
  cuda_kernel: pending
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
  current_runtime: flash_attn_varlen_func_placeholder
  target_runtime: handwritten_cuda_varlen_kernel
excluded_features:
  - dropout
  - return_attn_probs
  - window_size
  - alibi_slopes
  - softcap
  - deterministic
```
