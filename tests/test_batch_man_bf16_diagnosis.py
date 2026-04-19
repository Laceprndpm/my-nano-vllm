import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.cuda_fa2_torch_ext import fa2_batch_fwd


torch.manual_seed(514)


def _run_kernel_vs_sdpa(dtype: torch.dtype, *, B: int = 1, H: int = 8, N: int = 128, D: int = 64):
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    out_kernel_bshd, _lse = fa2_batch_fwd(
        q_bshd,
        k_bshd,
        v_bshd,
        causal=True,
        softmax_scale=1.0 / (D ** 0.5),
    )
    out_kernel = out_kernel_bshd.transpose(1, 2).contiguous()

    diff = (out_kernel - out_ref).abs()
    return diff.max().item(), diff.mean().item(), torch.allclose(out_kernel, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp16_kernel_matches_sdpa_baseline():
    max_abs_error, mean_abs_error, allclose = _run_kernel_vs_sdpa(torch.float16)
    assert allclose, (
        "fp16 kernel should remain close to sdpa baseline: "
        f"max_abs_error={max_abs_error:.6f}, mean_abs_error={mean_abs_error:.6f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bf16_kernel_matches_sdpa_baseline():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 is not supported on this GPU")

    max_abs_error, mean_abs_error, allclose = _run_kernel_vs_sdpa(torch.bfloat16)

    assert allclose, (
        "bf16 kernel should remain close to sdpa baseline: "
        f"max_abs_error={max_abs_error:.6f}, mean_abs_error={mean_abs_error:.6f}"
    )
