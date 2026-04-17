from __future__ import annotations

import os
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load


def _ext_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "csrc", "fa2"))


@lru_cache(maxsize=1)
def load_cuda_fa2_extension():
    root = _ext_root()
    cpp_path = os.path.join(root, "fa2_binding.cpp")
    cu_path = os.path.join(root, "fa2_fwd.cu")
    if not os.path.exists(cpp_path) or not os.path.exists(cu_path):
        raise RuntimeError(f"FA2 torch extension sources not found under {root}")
    return load(
        name="nanovllm_fa2_ext",
        sources=[cpp_path, cu_path],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
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
) -> torch.Tensor:
    ext = load_cuda_fa2_extension()
    return ext.fa2_fwd(q, k, v, causal, softmax_scale)
