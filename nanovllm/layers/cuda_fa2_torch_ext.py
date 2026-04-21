from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from flash_attn import flash_attn_varlen_func
from torch.utils.cpp_extension import CUDA_HOME, load


def _ext_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "csrc", "fa2"))


@lru_cache(maxsize=1)
def load_cuda_fa2_extension():
    root = _ext_root()
    cpp_path = os.path.join(root, "fa2_binding.cpp")
    cu_path = os.path.join(root, "fa2_fwd.cu")
    core_cu = os.path.join(root, "batch", "batch_fwd.cu")
    varlen_cu = os.path.join(root, "varlen", "varlen_fwd.cu")
    if not os.path.exists(cpp_path) or not os.path.exists(cu_path) or not os.path.exists(core_cu) or not os.path.exists(varlen_cu):
        raise RuntimeError(f"FA2 torch extension sources not found under {root}")
    repo_root = Path(root).resolve().parents[2]
    include_dirs = [
        str(Path(root)),
        str(repo_root / "deps" / "cutlass" / "include"),
        str(repo_root / "deps" / "cutlass" / "examples" / "common"),
    ]
    cuda_flags = [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-gencode",
        "arch=compute_80,code=sm_80",
    ]
    if CUDA_HOME:
        try:
            raw = subprocess.check_output([os.path.join(CUDA_HOME, "bin", "nvcc"), "-V"], text=True)
            if "release 11.8" in raw or "release 12." in raw:
                cuda_flags += ["-gencode", "arch=compute_90,code=sm_90"]
        except Exception:
            pass
    return load(
        name="nanovllm_fa2_ext",
        sources=[cpp_path, cu_path, core_cu, varlen_cu],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            *cuda_flags,
        ],
        with_cuda=True,
        verbose=False,
    )


def fa2_batch_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Python-side tensors are [B, S, H, D], but the CUDA kernel expects [B, H, S, D].
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4D tensors in [B, S, H, D] layout")
    q_bhsd = q.transpose(1, 2).contiguous()
    k_bhsd = k.transpose(1, 2).contiguous()
    v_bhsd = v.transpose(1, 2).contiguous()

    ext = load_cuda_fa2_extension()
    out_lse = ext.fa2_batch_fwd(q_bhsd, k_bhsd, v_bhsd, causal, softmax_scale)
    if isinstance(out_lse, (tuple, list)) and len(out_lse) == 2:
        out_bhsd, lse_bhs = out_lse
        out_bshd = out_bhsd.transpose(1, 2).contiguous()
        lse_bsh = lse_bhs.transpose(1, 2).contiguous()
        return out_bshd, lse_bsh
    raise RuntimeError("fa2_batch_fwd extension returned invalid output; expected (out, lse)")


def _validate_varlen_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_table: torch.Tensor,
):
    if q.dim() != 3:
        raise ValueError("q must be rank-3 [total_q, q_heads, head_dim]")
    if k.dim() not in (3, 4):
        raise ValueError("k must be rank-3 (dense varlen) or rank-4 (paged cache)")
    if v.dim() != k.dim():
        raise ValueError("k and v rank mismatch")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must have same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q, k, v must be fp16/bf16")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("max_seqlen_q/max_seqlen_k must be positive")
    if block_table is None:
        raise ValueError("block_table is required; pass an empty int32 tensor when no prefix cache")
    if block_table.dim() != 2:
        raise ValueError("block_table must be rank-2 [batch, num_blocks]")
    if block_table.dtype != torch.int32:
        raise ValueError("block_table must be int32")
    if not block_table.is_cuda:
        raise ValueError("block_table must be a CUDA tensor")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("cu_seqlens_q/cu_seqlens_k must be int32")
    if cu_seqlens_q.dim() != 1 or cu_seqlens_k.dim() != 1:
        raise ValueError("cu_seqlens_q/cu_seqlens_k must be rank-1")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same length")
    batch = int(cu_seqlens_q.numel()) - 1
    if block_table.size(0) != batch:
        raise ValueError(f"block_table batch mismatch: expected {batch}, got {block_table.size(0)}")


def fa2_varlen_fwd_official(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    _validate_varlen_inputs(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
    )

    block_table_for_fa = None if block_table.numel() == 0 else block_table

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
        block_table=block_table_for_fa,
    )


def fa2_varlen_fwd_man(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    _validate_varlen_inputs(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
    )
    if q.size(-1) not in (64, 128):
        raise ValueError(f"fa2_varlen_fwd_man currently supports head_dim in {{64, 128}}, got {q.size(-1)}")

    ext = load_cuda_fa2_extension()
    return ext.fa2_varlen_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        int(max_seqlen_q),
        int(max_seqlen_k),
        block_table.contiguous(),
        bool(causal),
        float(softmax_scale),
    )


def fa2_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    # Keep historical API contract: this symbol maps to official/reference path.
    return fa2_varlen_fwd_official(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_table=block_table,
        causal=causal,
        softmax_scale=softmax_scale,
    )
