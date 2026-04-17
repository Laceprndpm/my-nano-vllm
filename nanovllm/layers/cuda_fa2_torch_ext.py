from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


def _ext_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "csrc", "fa2"))


@lru_cache(maxsize=1)
def load_cuda_fa2_extension():
    root = _ext_root()
    cpp_path = os.path.join(root, "fa2_binding.cpp")
    cu_path = os.path.join(root, "fa2_fwd.cu")
    core_cu = os.path.join(root, "flash_attention.cu")
    if not os.path.exists(cpp_path) or not os.path.exists(cu_path) or not os.path.exists(core_cu):
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
        sources=[cpp_path, cu_path, core_cu],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            *cuda_flags,
        ],
        with_cuda=True,
        verbose=False,
    )


def fa2_fwd(
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
    out_lse = ext.fa2_fwd(q_bhsd, k_bhsd, v_bhsd, causal, softmax_scale)
    if isinstance(out_lse, (tuple, list)) and len(out_lse) == 2:
        out_bhsd, lse_bhs = out_lse
        out_bshd = out_bhsd.transpose(1, 2).contiguous()
        lse_bsh = lse_bhs.transpose(1, 2).contiguous()
        return out_bshd, lse_bsh
    raise RuntimeError("fa2_fwd extension returned invalid output; expected (out, lse)")
