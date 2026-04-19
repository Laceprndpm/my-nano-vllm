from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func

from nanovllm.layers.cuda_fa2_torch_ext import (
    fa2_batch_fwd as torch_ext_fa2_batch_fwd,
    fa2_varlen_fwd as torch_ext_fa2_varlen_fwd,
)


@dataclass(slots=True)
class PrefillBackendSettings:
    backend: str = "cuda_fa2"
    fallback_to_flash_attn: bool = False


_SETTINGS = PrefillBackendSettings()
_BATCH_MAN_PAD64_WARNING_EMITTED = False
_BATCH_DEBUG_ATOL = 1e-2
_BATCH_DEBUG_RTOL = 1e-2


def configure_prefill_attention_backend(backend: str, fallback_to_flash_attn: bool) -> None:
    """更新 prefill attention 的全局后端配置与回退策略。"""
    _SETTINGS.backend = backend
    _SETTINGS.fallback_to_flash_attn = fallback_to_flash_attn


def _direct_flash_attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    block_table: torch.Tensor | None,
) -> torch.Tensor:
    """直接调用 flash-attn 执行变长 prefill 计算路径。"""
    return flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=causal,
        block_table=block_table,
    )


def _seq_lens(cu_seqlens: torch.Tensor) -> list[int]:
    """将累计序列偏移量转换为逐条序列长度。"""
    cu = cu_seqlens.to("cpu", dtype=torch.int64).tolist()
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]


def _expand_kv_heads(x: torch.Tensor, q_heads: int) -> torch.Tensor:
    """在 GQA 场景下将 KV 头数扩展到与 Q 头数一致。"""
    kv_heads = x.size(1)
    if kv_heads == q_heads:
        return x
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads})")
    return x.repeat_interleave(q_heads // kv_heads, dim=1)


@dataclass(slots=True)
class PaddedBatch:
    q_pad: torch.Tensor
    k_pad: torch.Tensor
    v_pad: torch.Tensor
    q_valid: torch.Tensor
    k_valid: torch.Tensor
    q_lens: list[int]
    k_lens: list[int]
    max_seqlen_q: int
    max_seqlen_k: int


def _varlen_to_padded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> PaddedBatch:
    """将扁平变长 Q/K/V 打包为定长 padding 批次，并生成有效位掩码。"""
    q_lens = _seq_lens(cu_seqlens_q)
    k_lens = _seq_lens(cu_seqlens_k)
    if len(q_lens) != len(k_lens):
        raise ValueError("q and k batch size mismatch")

    batch = len(q_lens)
    q_heads, dim = q.size(1), q.size(2)
    q_pad = torch.zeros((batch, max_seqlen_q, q_heads, dim), dtype=q.dtype, device=q.device)
    k_pad = torch.zeros((batch, max_seqlen_k, q_heads, dim), dtype=k.dtype, device=k.device)
    v_pad = torch.zeros((batch, max_seqlen_k, q_heads, dim), dtype=v.dtype, device=v.device)
    q_valid = torch.zeros((batch, max_seqlen_q), dtype=torch.bool, device=q.device)
    k_valid = torch.zeros((batch, max_seqlen_k), dtype=torch.bool, device=k.device)

    q_offset = 0
    k_offset = 0
    for b, (ql, kl) in enumerate(zip(q_lens, k_lens)):
        if ql > max_seqlen_q or kl > max_seqlen_k:
            raise ValueError("seqlen exceeds provided max_seqlen")
        q_seq = q[q_offset:q_offset + ql]
        k_seq = _expand_kv_heads(k[k_offset:k_offset + kl], q_heads)
        v_seq = _expand_kv_heads(v[k_offset:k_offset + kl], q_heads)
        q_pad[b, :ql] = q_seq
        k_pad[b, :kl] = k_seq
        v_pad[b, :kl] = v_seq
        q_valid[b, :ql] = True
        k_valid[b, :kl] = True
        q_offset += ql
        k_offset += kl
    if q_offset != q.size(0) or k_offset != k.size(0) or k_offset != v.size(0):
        raise ValueError("input varlen tensors do not match cu_seqlens")

    return PaddedBatch(
        q_pad=q_pad,
        k_pad=k_pad,
        v_pad=v_pad,
        q_valid=q_valid,
        k_valid=k_valid,
        q_lens=q_lens,
        k_lens=k_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


def _build_padded_attention_mask(q_valid: torch.Tensor, k_valid: torch.Tensor, causal: bool) -> torch.Tensor:
    """根据有效位掩码与是否因果约束构造布尔 attention mask。"""
    valid = q_valid.unsqueeze(-1) & k_valid.unsqueeze(-2)
    if not causal:
        return valid
    sq = q_valid.size(1)
    sk = k_valid.size(1)
    causal_mask = torch.ones((sq, sk), dtype=torch.bool, device=q_valid.device).tril()
    return valid & causal_mask.unsqueeze(0)


def _padded_to_varlen(padded: torch.Tensor, lens: list[int], cu_seqlens: torch.Tensor) -> torch.Tensor:
    """将 padding 后的批次张量还原为单个变长张量。"""
    if padded.size(0) != len(lens):
        raise ValueError("padded batch and lengths mismatch")
    chunks = []
    for b, seqlen in enumerate(lens):
        chunks.append(padded[b, :seqlen])
    out = torch.cat(chunks, dim=0) if chunks else padded.new_empty((0, padded.size(2), padded.size(3)))
    expected = int(cu_seqlens[-1].item())
    if out.size(0) != expected:
        raise ValueError("reconstructed varlen length mismatch")
    return out


def _run_cuda_batch_fa2_official(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """官方 batch-view 路径：在 batch 视图下调用 flash-attn FA2 进行前向。"""
    padded = _varlen_to_padded(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    _attn_mask = _build_padded_attention_mask(padded.q_valid, padded.k_valid, causal)

    q_pad = padded.q_pad * padded.q_valid.unsqueeze(-1).unsqueeze(-1).to(padded.q_pad.dtype)
    k_pad = padded.k_pad * padded.k_valid.unsqueeze(-1).unsqueeze(-1).to(padded.k_pad.dtype)
    v_pad = padded.v_pad * padded.k_valid.unsqueeze(-1).unsqueeze(-1).to(padded.v_pad.dtype)
    out_pad = torch.zeros_like(q_pad)

    for b, (ql, kl) in enumerate(zip(padded.q_lens, padded.k_lens)):
        if ql == 0 or kl == 0:
            continue
        q_b = q_pad[b:b + 1, :ql]
        k_b = k_pad[b:b + 1, :kl]
        v_b = v_pad[b:b + 1, :kl]
        out_b = flash_attn_func(
            q_b,
            k_b,
            v_b,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        out_pad[b:b + 1, :ql] = out_b

    out_pad = out_pad * padded.q_valid.unsqueeze(-1).unsqueeze(-1).to(out_pad.dtype)
    out_pad = out_pad.masked_fill(~_attn_mask.any(dim=-1).unsqueeze(-1).unsqueeze(-1), 0)
    return _padded_to_varlen(out_pad, padded.q_lens, cu_seqlens_q)


def _run_cuda_batch_fa2_man(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """手写 CUDA batch-view 路径：逐序列调用 torch bind 自写 kernel。"""
    global _BATCH_MAN_PAD64_WARNING_EMITTED
    padded = _varlen_to_padded(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    q_pad = padded.q_pad * padded.q_valid.unsqueeze(-1).unsqueeze(-1).to(padded.q_pad.dtype)
    k_pad = padded.k_pad * padded.k_valid.unsqueeze(-1).unsqueeze(-1).to(padded.k_pad.dtype)
    v_pad = padded.v_pad * padded.k_valid.unsqueeze(-1).unsqueeze(-1).to(padded.v_pad.dtype)
    out_pad = torch.zeros_like(q_pad)
    for b, (ql, kl) in enumerate(zip(padded.q_lens, padded.k_lens)):
        if ql == 0 or kl == 0:
            continue
        ql_pad = ((ql + 63) // 64) * 64
        kl_pad = ((kl + 63) // 64) * 64
        if (ql_pad != ql or kl_pad != kl) and not _BATCH_MAN_PAD64_WARNING_EMITTED:
            warnings.warn(
                "[WARNING][FA2_BATCH_MAN_PAD64] batch_man kernel currently requires 64-aligned "
                "seqlen for stable numerics; applying temporary Python-side pad-to-64 hotfix.",
                RuntimeWarning,
                stacklevel=2,
            )
            _BATCH_MAN_PAD64_WARNING_EMITTED = True
        q_b = torch.zeros((1, ql_pad, q_pad.size(2), q_pad.size(3)), dtype=q_pad.dtype, device=q_pad.device)
        k_b = torch.zeros((1, kl_pad, k_pad.size(2), k_pad.size(3)), dtype=k_pad.dtype, device=k_pad.device)
        v_b = torch.zeros((1, kl_pad, v_pad.size(2), v_pad.size(3)), dtype=v_pad.dtype, device=v_pad.device)
        q_b[:, :ql] = q_pad[b:b + 1, :ql]
        k_b[:, :kl] = k_pad[b:b + 1, :kl]
        v_b[:, :kl] = v_pad[b:b + 1, :kl]
        out_lse = torch_ext_fa2_batch_fwd(
            q_b,
            k_b,
            v_b,
            causal=causal,
            softmax_scale=float(softmax_scale),
        )
        out_b = out_lse[0] if isinstance(out_lse, (tuple, list)) else out_lse
        out_pad[b:b + 1, :ql] = out_b[:, :ql]
    return _padded_to_varlen(out_pad, padded.q_lens, cu_seqlens_q)


def _run_cuda_batch_fa2_debug(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """调试路径：同时运行 batch_man 和 batch_official 并断言输出一致。"""
    out_man = _run_cuda_batch_fa2_man(
        q,
        k,
        v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    out_official = _run_cuda_batch_fa2_official(
        q,
        k,
        v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    if out_man.shape != out_official.shape:
        raise AssertionError(
            "batch_debug mismatch: shape differs between batch_man and batch_official "
            f"(man={tuple(out_man.shape)}, official={tuple(out_official.shape)})"
        )
    diff = (out_man - out_official).abs()
    max_abs_error = float(diff.max().item()) if diff.numel() > 0 else 0.0
    mean_abs_error = float(diff.mean().item()) if diff.numel() > 0 else 0.0
    allclose = torch.allclose(out_man, out_official, atol=_BATCH_DEBUG_ATOL, rtol=_BATCH_DEBUG_RTOL)
    if not allclose:
        raise AssertionError(
            "batch_debug mismatch between batch_man and batch_official: "
            f"shape={tuple(out_man.shape)}, dtype={out_man.dtype}, device={out_man.device}, "
            f"atol={_BATCH_DEBUG_ATOL}, rtol={_BATCH_DEBUG_RTOL}, "
            f"max_abs_error={max_abs_error:.6f}, mean_abs_error={mean_abs_error:.6f}"
        )
    return out_man


def _run_cuda_varlen_fa2_official(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """官方 varlen 路径：当前通过 flash-attn varlen 占位接口贯通。"""
    return torch_ext_fa2_varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        causal=causal,
        softmax_scale=float(softmax_scale),
    )


def _run_cuda_varlen_fa2_man_placeholder(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """手写 CUDA varlen 路径占位：暂时回落到官方 varlen 实现。"""
    return _run_cuda_varlen_fa2_official(
        q,
        k,
        v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def _run_cuda_fwd_placeholder(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """兼容旧命名的 fwd 占位入口：等价于 batch official 路径。"""
    return _run_cuda_batch_fa2_official(
        q,
        k,
        v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def _run_cuda_fa2_placeholder(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Legacy alias kept for compatibility with existing tests/callers."""
    return _run_cuda_batch_fa2_official(
        q,
        k,
        v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def _normalize_block_table_required(
    *,
    cu_seqlens_q: torch.Tensor,
    block_table: torch.Tensor | None,
) -> torch.Tensor:
    batch = int(cu_seqlens_q.numel()) - 1
    if block_table is None:
        return torch.empty((batch, 0), dtype=torch.int32, device=cu_seqlens_q.device)
    if block_table.dtype != torch.int32:
        raise ValueError("block_table must be int32")
    if block_table.device != cu_seqlens_q.device:
        raise ValueError("block_table must be on the same device as cu_seqlens_q")
    if block_table.dim() != 2:
        raise ValueError("block_table must be rank-2 [batch, num_blocks]")
    if block_table.size(0) != batch:
        raise ValueError(f"block_table batch mismatch: expected {batch}, got {block_table.size(0)}")
    return block_table.contiguous()


def _resolve_fa2_mode() -> str:
    """Resolve FA2 runtime mode."""
    mode = os.getenv("NANOVLLM_FA2_MODE")
    if mode:
        return mode
    if os.getenv("NANOVLLM_FA2_USE_TORCH_EXT", "0") == "1":
        return "varlen_official"
    return "batch_official"


def run_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    block_table: torch.Tensor | None,
) -> torch.Tensor:
    """按配置选择 prefill attention 后端，并在需要时执行回退。"""
    if _SETTINGS.backend == "flash_attn":
        return _direct_flash_attn_prefill(
            q,
            k,
            v,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
        )
    if _SETTINGS.backend == "cuda_fa2":
        try:
            mode = _resolve_fa2_mode()
            if mode == "varlen_official":
                required_block_table = _normalize_block_table_required(cu_seqlens_q=cu_seqlens_q, block_table=block_table)
                return _run_cuda_varlen_fa2_official(
                    q,
                    k,
                    v,
                    max_seqlen_q=max_seqlen_q,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_k=max_seqlen_k,
                    cu_seqlens_k=cu_seqlens_k,
                    block_table=required_block_table,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            if mode == "varlen_man":
                required_block_table = _normalize_block_table_required(cu_seqlens_q=cu_seqlens_q, block_table=block_table)
                return _run_cuda_varlen_fa2_man_placeholder(
                    q,
                    k,
                    v,
                    max_seqlen_q=max_seqlen_q,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_k=max_seqlen_k,
                    cu_seqlens_k=cu_seqlens_k,
                    block_table=required_block_table,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            if mode == "batch_official":
                if block_table is not None and block_table.numel() > 0:
                    raise RuntimeError("batch_official mode does not support non-empty block_table")
                return _run_cuda_batch_fa2_official(
                    q,
                    k,
                    v,
                    max_seqlen_q=max_seqlen_q,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_k=max_seqlen_k,
                    cu_seqlens_k=cu_seqlens_k,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            if mode == "batch_man":
                if block_table is not None and block_table.numel() > 0:
                    raise RuntimeError("batch_man mode does not support non-empty block_table")
                return _run_cuda_batch_fa2_man(
                    q,
                    k,
                    v,
                    max_seqlen_q=max_seqlen_q,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_k=max_seqlen_k,
                    cu_seqlens_k=cu_seqlens_k,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            if mode == "batch_debug":
                if block_table is not None and block_table.numel() > 0:
                    raise RuntimeError("batch_debug mode does not support non-empty block_table")
                return _run_cuda_batch_fa2_debug(
                    q,
                    k,
                    v,
                    max_seqlen_q=max_seqlen_q,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_k=max_seqlen_k,
                    cu_seqlens_k=cu_seqlens_k,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            raise ValueError(f"unsupported NANOVLLM_FA2_MODE: {mode}")
        except Exception:
            if _SETTINGS.fallback_to_flash_attn:
                return _direct_flash_attn_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q=max_seqlen_q,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_k=max_seqlen_k,
                    cu_seqlens_k=cu_seqlens_k,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    block_table=block_table,
                )
            raise
    raise ValueError(f"unsupported prefill backend: {_SETTINGS.backend}")
