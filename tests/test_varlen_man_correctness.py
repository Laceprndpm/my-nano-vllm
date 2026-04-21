import math

import pytest
import torch

flash_attn = pytest.importorskip("flash_attn")
from flash_attn import flash_attn_varlen_func

from nanovllm.layers.cuda_fa2_torch_ext import fa2_varlen_fwd_man, fa2_varlen_fwd_official


DTYPE = torch.float16
Q_HEADS = 4
KV_HEADS = 4
BLOCK_SIZE = 64
PAGED_BLOCK_SIZE = 256
ALIGNED_SIZES = [64, 128, 192, 512]
SUPPORTED_HEAD_DIMS = [64, 128]


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _seed():
    torch.manual_seed(514)
    torch.cuda.manual_seed_all(514)


def _build_cu_seqlens(lengths: list[int], device: str) -> torch.Tensor:
    out = [0]
    for x in lengths:
        out.append(out[-1] + x)
    return torch.tensor(out, dtype=torch.int32, device=device)


def _build_dense_varlen_case(seq_len: int, head_dim: int):
    batch = 2
    device = "cuda"
    q_lens = [seq_len] * batch
    k_lens = [seq_len] * batch
    q = torch.randn(sum(q_lens), Q_HEADS, head_dim, dtype=DTYPE, device=device)
    k = torch.randn(sum(k_lens), KV_HEADS, head_dim, dtype=DTYPE, device=device)
    v = torch.randn(sum(k_lens), KV_HEADS, head_dim, dtype=DTYPE, device=device)
    cu_q = _build_cu_seqlens(q_lens, device)
    cu_k = _build_cu_seqlens(k_lens, device)
    return q, k, v, cu_q, cu_k


def _build_paged_varlen_case(seq_len: int, head_dim: int):
    batch = 2
    device = "cuda"

    if seq_len % BLOCK_SIZE != 0:
        raise ValueError("seq_len must be aligned to 64")

    num_blocks_per_seq = (seq_len + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE
    total_blocks = batch * num_blocks_per_seq
    q, dense_k, dense_v, cu_q, cu_k = _build_dense_varlen_case(seq_len, head_dim)
    dense_k = dense_k.view(batch, seq_len, KV_HEADS, head_dim)
    dense_v = dense_v.view(batch, seq_len, KV_HEADS, head_dim)
    k = torch.zeros(
        total_blocks,
        PAGED_BLOCK_SIZE,
        KV_HEADS,
        head_dim,
        dtype=DTYPE,
        device=device,
    )
    v = torch.zeros_like(k)
    for b in range(batch):
        seq_start = b * num_blocks_per_seq
        for block_idx in range(num_blocks_per_seq):
            token_start = block_idx * PAGED_BLOCK_SIZE
            token_end = min(token_start + PAGED_BLOCK_SIZE, seq_len)
            block_id = seq_start + block_idx
            block_tokens = token_end - token_start
            if block_tokens <= 0:
                continue
            k[block_id, :block_tokens] = dense_k[b, token_start:token_end]
            v[block_id, :block_tokens] = dense_v[b, token_start:token_end]
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).view(batch, num_blocks_per_seq)
    return q, k, v, cu_q, cu_k, block_table


def _assert_output_matches(out_man: torch.Tensor, out_ref: torch.Tensor, expected_shape: torch.Size) -> None:
    assert out_man.shape == expected_shape
    assert out_ref.shape == expected_shape
    assert out_man.dtype == DTYPE
    assert out_ref.dtype == DTYPE
    assert out_man.shape == out_ref.shape
    assert out_man.dtype == out_ref.dtype
    torch.testing.assert_close(out_man, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("head_dim", SUPPORTED_HEAD_DIMS)
@pytest.mark.parametrize("seq_len", ALIGNED_SIZES)
def test_varlen_man_matches_flash_attn_for_dense_kv(seq_len, head_dim):
    _require_cuda()
    _seed()

    softmax_scale = 1.0 / math.sqrt(head_dim)
    q, k, v, cu_q, cu_k = _build_dense_varlen_case(seq_len, head_dim)
    out_man = fa2_varlen_fwd_man(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        block_table=torch.empty((cu_q.numel() - 1, 0), dtype=torch.int32, device="cuda"),
        causal=True,
        softmax_scale=softmax_scale,
    )
    out_ref = flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=seq_len,
        cu_seqlens_q=cu_q,
        max_seqlen_k=seq_len,
        cu_seqlens_k=cu_k,
        softmax_scale=softmax_scale,
        causal=True,
        block_table=None,
    )
    _assert_output_matches(out_man, out_ref, q.shape)


@pytest.mark.parametrize("head_dim", SUPPORTED_HEAD_DIMS)
@pytest.mark.parametrize("seq_len", ALIGNED_SIZES)
def test_varlen_man_matches_official_path_for_paged_kv(seq_len, head_dim):
    _require_cuda()
    _seed()

    softmax_scale = 1.0 / math.sqrt(head_dim)
    q, k, v, cu_q, cu_k, block_table = _build_paged_varlen_case(seq_len, head_dim)
    out_man = fa2_varlen_fwd_man(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        block_table=block_table,
        causal=True,
        softmax_scale=softmax_scale,
    )
    out_ref = fa2_varlen_fwd_official(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        block_table=block_table,
        causal=True,
        softmax_scale=softmax_scale,
    )
    _assert_output_matches(out_man, out_ref, q.shape)


def test_varlen_man_rejects_unsupported_head_dim():
    _require_cuda()
    _seed()

    head_dim = 96
    q, k, v, cu_q, cu_k = _build_dense_varlen_case(seq_len=64, head_dim=head_dim)
    with pytest.raises((ValueError, RuntimeError), match="head_dim|supports"):
        fa2_varlen_fwd_man(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=64,
            max_seqlen_k=64,
            block_table=torch.empty((cu_q.numel() - 1, 0), dtype=torch.int32, device="cuda"),
            causal=True,
            softmax_scale=1.0 / math.sqrt(head_dim),
        )
