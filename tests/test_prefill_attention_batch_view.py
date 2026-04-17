import pytest
import torch

flash_attn = pytest.importorskip("flash_attn")

from nanovllm.layers.prefill_attention import (
    _padded_to_varlen,
    _run_cuda_fa2_placeholder,
    _varlen_to_padded,
    configure_prefill_attention_backend,
    run_prefill_attention,
)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flash-attn tests")


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


def test_varlen_padded_roundtrip_q():
    _require_cuda()
    q_lens = [3, 5, 2]
    k_lens = [4, 6, 3]
    q, k, v, cu_q, cu_k = _build_varlen_qkv(q_lens, k_lens, q_heads=8, kv_heads=4, dim=64)
    padded = _varlen_to_padded(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
    )
    q_back = _padded_to_varlen(padded.q_pad, padded.q_lens, cu_q)
    assert torch.allclose(q, q_back, atol=0, rtol=0)


def test_batch_view_backend_matches_flash_attn_baseline():
    _require_cuda()
    q_lens = [7, 5, 3]
    k_lens = [7, 5, 3]
    q, k, v, cu_q, cu_k = _build_varlen_qkv(q_lens, k_lens, q_heads=8, kv_heads=4, dim=64)

    configure_prefill_attention_backend("flash_attn", True)
    out_baseline = run_prefill_attention(
        q,
        k,
        v,
        max_seqlen_q=max(q_lens),
        cu_seqlens_q=cu_q,
        max_seqlen_k=max(k_lens),
        cu_seqlens_k=cu_k,
        softmax_scale=1.0 / (64 ** 0.5),
        causal=True,
        block_table=None,
    )

    configure_prefill_attention_backend("cuda_fa2", True)
    out_batch_view = run_prefill_attention(
        q,
        k,
        v,
        max_seqlen_q=max(q_lens),
        cu_seqlens_q=cu_q,
        max_seqlen_k=max(k_lens),
        cu_seqlens_k=cu_k,
        softmax_scale=1.0 / (64 ** 0.5),
        causal=True,
        block_table=None,
    )
    assert torch.allclose(out_baseline, out_batch_view, atol=1e-2, rtol=1e-2)


def test_padding_invariance_in_batch_view_path():
    _require_cuda()
    q_lens = [5, 4]
    k_lens = [5, 4]
    q, k, v, cu_q, cu_k = _build_varlen_qkv(q_lens, k_lens, q_heads=8, kv_heads=8, dim=64)
    scale = 1.0 / (64 ** 0.5)

    out_ref = _run_cuda_fa2_placeholder(
        q,
        k,
        v,
        max_seqlen_q=max(q_lens),
        cu_seqlens_q=cu_q,
        max_seqlen_k=max(k_lens),
        cu_seqlens_k=cu_k,
        softmax_scale=scale,
        causal=True,
    )
    out_more_padding = _run_cuda_fa2_placeholder(
        q,
        k,
        v,
        max_seqlen_q=max(q_lens) + 3,
        cu_seqlens_q=cu_q,
        max_seqlen_k=max(k_lens) + 3,
        cu_seqlens_k=cu_k,
        softmax_scale=scale,
        causal=True,
    )
    assert torch.allclose(out_ref, out_more_padding, atol=1e-2, rtol=1e-2)


def test_causal_mask_blocks_future_influence():
    _require_cuda()
    q_lens = [8]
    k_lens = [8]
    q, k, v, cu_q, cu_k = _build_varlen_qkv(q_lens, k_lens, q_heads=4, kv_heads=4, dim=64)
    scale = 1.0 / (64 ** 0.5)
    out_ref = _run_cuda_fa2_placeholder(
        q,
        k,
        v,
        max_seqlen_q=8,
        cu_seqlens_q=cu_q,
        max_seqlen_k=8,
        cu_seqlens_k=cu_k,
        softmax_scale=scale,
        causal=True,
    )

    k_changed = k.clone()
    v_changed = v.clone()
    k_changed[-1] = torch.randn_like(k_changed[-1]) * 50
    v_changed[-1] = torch.randn_like(v_changed[-1]) * 50
    out_changed = _run_cuda_fa2_placeholder(
        q,
        k_changed,
        v_changed,
        max_seqlen_q=8,
        cu_seqlens_q=cu_q,
        max_seqlen_k=8,
        cu_seqlens_k=cu_k,
        softmax_scale=scale,
        causal=True,
    )

    assert torch.allclose(out_ref[:7], out_changed[:7], atol=1e-2, rtol=1e-2)
