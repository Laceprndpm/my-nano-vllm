import torch
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda"
dtype = torch.float16

B, H, N, D = 1, 8, 128, 64
q = torch.randn(B, H, N, D, device=device, dtype=dtype)
k = torch.randn(B, H, N, D, device=device, dtype=dtype)
v = torch.randn(B, H, N, D, device=device, dtype=dtype)

o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

print("shape:", o.shape)
print("dtype:", o.dtype)
print("mean:", o.float().mean().item())
print("std:", o.float().std().item())