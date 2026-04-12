import torch
import torch.nn.functional as F

torch.manual_seed(514)

DEVICE = "cuda"
DTYPE = torch.float16


def sdpa_ref(q, k, v, causal=True):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def flash_attn_ref(q, k, v, causal=True):
    from flash_attn import flash_attn_func
    # flash-attn 要 [B, N, H, D] 而不是[B, H, N, D]
    q_ = q.transpose(1, 2).contiguous()
    k_ = k.transpose(1, 2).contiguous()
    v_ = v.transpose(1, 2).contiguous()
    o_ = flash_attn_func(q_, k_, v_, causal=causal)
    return o_.transpose(1, 2).contiguous()


def run_case(B, H, N, D):
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=DTYPE)

    out_sdpa = sdpa_ref(q, k, v, causal=True)
    out_fa = flash_attn_ref(q, k, v, causal=True)

    diff = (out_sdpa - out_fa).abs()
    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()
    allclose = torch.allclose(out_sdpa, out_fa, atol=1e-2, rtol=1e-2)

    print(f"[B={B}, H={H}, N={N}, D={D}]")
    print(f"  max_abs_error : {max_abs_error:.6f}")
    print(f"  mean_abs_error: {mean_abs_error:.6f}")
    print(f"  allclose      : {allclose}")
    print()


if __name__ == "__main__":
    run_case(1, 8,  128, 64)
    run_case(1, 8,  512, 64)
    run_case(1, 8, 1024, 64)