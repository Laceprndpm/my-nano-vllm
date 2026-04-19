import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.cuda_fa2_torch_ext import fa2_batch_fwd

torch.manual_seed(514)

DEVICE = "cuda"
DTYPE = torch.float16


def sdpa_ref(q, k, v, causal=True):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def kernel_ref(q, k, v, causal=True):
    # fa2_batch_fwd 吃 [B, N, H, D]，这里从 [B, H, N, D] 转过去再转回。
    _, _, _, D = q.shape
    q_ = q.transpose(1, 2).contiguous()
    k_ = k.transpose(1, 2).contiguous()
    v_ = v.transpose(1, 2).contiguous()
    o_, _lse = fa2_batch_fwd(
        q_,
        k_,
        v_,
        softmax_scale=1.0 / (D**0.5),
        causal=causal,
    )
    return o_.transpose(1, 2).contiguous()


def run_case(B, H, N, D):
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)

    out_sdpa = sdpa_ref(q, k, v, causal=True)
    out_kernel = kernel_ref(q, k, v, causal=True)

    diff = (out_sdpa - out_kernel).abs()
    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()
    allclose = torch.allclose(out_sdpa, out_kernel, atol=1e-2, rtol=1e-2)

    return max_abs_error, mean_abs_error, allclose


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "B,H,N,D",
    [
        (1, 8, 64, 64),
        (1, 8, 128, 64),
        (1, 8, 192, 64),
        (1, 8, 512, 64),
    ],
)
def test_kernel_matches_sdpa(B, H, N, D):
    max_abs_error, mean_abs_error, allclose = run_case(B, H, N, D)
    assert allclose, (
        f"kernel mismatch for B={B}, H={H}, N={N}, D={D}: "
        f"max_abs_error={max_abs_error:.6f}, mean_abs_error={mean_abs_error:.6f}"
    )
