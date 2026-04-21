import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.cuda_fa2_torch_ext import fa2_batch_fwd

torch.manual_seed(514)

DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16]


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


def run_case(B, H, N, D, dtype: torch.dtype):
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)

    out_sdpa = sdpa_ref(q, k, v, causal=True)
    out_kernel = kernel_ref(q, k, v, causal=True)

    diff = (out_sdpa - out_kernel).abs()
    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()
    allclose = torch.allclose(out_sdpa, out_kernel, atol=1e-2, rtol=1e-2)

    return max_abs_error, mean_abs_error, allclose


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "B,H,N,D",
    [
        (1, 8, 64, 64),
        (1, 8, 128, 64),
        (1, 8, 192, 64),
        (1, 8, 512, 64),
    ],
)
def test_kernel_matches_sdpa(B, H, N, D, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 is not supported on this GPU")

    max_abs_error, mean_abs_error, allclose = run_case(B, H, N, D, dtype)
    assert allclose, (
        f"kernel mismatch for dtype={dtype}, B={B}, H={H}, N={N}, D={D}: "
        f"max_abs_error={max_abs_error:.6f}, mean_abs_error={mean_abs_error:.6f}"
    )
