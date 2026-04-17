import pytest
import torch
from flash_attn import flash_attn_varlen_func

from nanovllm.layers.cuda_fa2_torch_ext import fa2_varlen_fwd


flash_attn = pytest.importorskip("flash_attn")


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _build_cu_seqlens(lengths: list[int], device: str) -> torch.Tensor:
    out = [0]
    for x in lengths:
        out.append(out[-1] + x)
    return torch.tensor(out, dtype=torch.int32, device=device)


def _build_varlen_qkv(
    q_lens: list[int],
    k_lens: list[int],
    q_heads: int,
    kv_heads: int,
    dim: int,
    dtype=torch.float16,
):
    assert len(q_lens) == len(k_lens)
    device = "cuda"
    q = torch.randn(sum(q_lens), q_heads, dim, dtype=dtype, device=device)
    k = torch.randn(sum(k_lens), kv_heads, dim, dtype=dtype, device=device)
    v = torch.randn(sum(k_lens), kv_heads, dim, dtype=dtype, device=device)
    cu_q = _build_cu_seqlens(q_lens, device)
    cu_k = _build_cu_seqlens(k_lens, device)
    return q, k, v, cu_q, cu_k


def test_varlen_api_requires_block_table():
    _require_cuda()
    q, k, v, cu_q, cu_k = _build_varlen_qkv([4], [4], q_heads=4, kv_heads=4, dim=64)
    with pytest.raises(ValueError, match="block_table is required"):
        fa2_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=4,
            max_seqlen_k=4,
            block_table=None,
            causal=True,
            softmax_scale=1.0 / (64 ** 0.5),
        )


def test_varlen_api_accepts_empty_block_table_and_matches_flash_attn():
    _require_cuda()
    q_lens = [7, 5, 3]
    k_lens = [7, 5, 3]
    q, k, v, cu_q, cu_k = _build_varlen_qkv(q_lens, k_lens, q_heads=8, kv_heads=4, dim=64)
    scale = 1.0 / (64 ** 0.5)
    block_table = torch.empty((len(q_lens), 0), dtype=torch.int32, device="cuda")

    out_api = fa2_varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        block_table=block_table,
        causal=True,
        softmax_scale=scale,
    )
    out_ref = flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=max(q_lens),
        cu_seqlens_q=cu_q,
        max_seqlen_k=max(k_lens),
        cu_seqlens_k=cu_k,
        softmax_scale=scale,
        causal=True,
        block_table=None,
    )
    assert torch.allclose(out_api, out_ref, atol=1e-2, rtol=1e-2)


def test_varlen_api_validates_block_table_batch_dim():
    _require_cuda()
    q, k, v, cu_q, cu_k = _build_varlen_qkv([4, 4], [4, 4], q_heads=4, kv_heads=4, dim=64)
    bad_block_table = torch.empty((1, 0), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="batch mismatch"):
        fa2_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=4,
            max_seqlen_k=4,
            block_table=bad_block_table,
            causal=True,
            softmax_scale=1.0 / (64 ** 0.5),
        )
