import pytest
import torch

import nanovllm.layers.prefill_attention as prefill_attention


def _sample_inputs(device: str = "cpu"):
    q = torch.zeros((3, 2, 4), dtype=torch.float16, device=device)
    k = torch.zeros((3, 2, 4), dtype=torch.float16, device=device)
    v = torch.zeros((3, 2, 4), dtype=torch.float16, device=device)
    cu = torch.tensor([0, 3], dtype=torch.int32, device=device)
    return q, k, v, cu, cu


def test_batch_debug_returns_man_output_when_close(monkeypatch):
    q, k, v, cu_q, cu_k = _sample_inputs()

    out_man = torch.randn((3, 2, 4), dtype=torch.float16)
    out_official = out_man + 1e-4

    monkeypatch.setattr(prefill_attention, "_run_cuda_batch_fa2_man", lambda *args, **kwargs: out_man)
    monkeypatch.setattr(prefill_attention, "_run_cuda_batch_fa2_official", lambda *args, **kwargs: out_official)

    out = prefill_attention._run_cuda_batch_fa2_debug(
        q,
        k,
        v,
        max_seqlen_q=3,
        cu_seqlens_q=cu_q,
        max_seqlen_k=3,
        cu_seqlens_k=cu_k,
        softmax_scale=1.0,
        causal=True,
    )

    assert torch.equal(out, out_man)


def test_batch_debug_raises_on_mismatch(monkeypatch):
    q, k, v, cu_q, cu_k = _sample_inputs()

    out_man = torch.zeros((3, 2, 4), dtype=torch.float16)
    out_official = torch.ones((3, 2, 4), dtype=torch.float16)

    monkeypatch.setattr(prefill_attention, "_run_cuda_batch_fa2_man", lambda *args, **kwargs: out_man)
    monkeypatch.setattr(prefill_attention, "_run_cuda_batch_fa2_official", lambda *args, **kwargs: out_official)

    with pytest.raises(AssertionError, match="max_abs_error=.*mean_abs_error"):
        prefill_attention._run_cuda_batch_fa2_debug(
            q,
            k,
            v,
            max_seqlen_q=3,
            cu_seqlens_q=cu_q,
            max_seqlen_k=3,
            cu_seqlens_k=cu_k,
            softmax_scale=1.0,
            causal=True,
        )


def test_batch_debug_mode_rejects_non_empty_block_table(monkeypatch):
    q, k, v, cu_q, cu_k = _sample_inputs()
    monkeypatch.setenv("NANOVLLM_FA2_MODE", "batch_debug")
    prefill_attention.configure_prefill_attention_backend("cuda_fa2", False)

    with pytest.raises(RuntimeError, match="batch_debug mode does not support non-empty block_table"):
        prefill_attention.run_prefill_attention(
            q,
            k,
            v,
            max_seqlen_q=3,
            cu_seqlens_q=cu_q,
            max_seqlen_k=3,
            cu_seqlens_k=cu_k,
            softmax_scale=1.0,
            causal=True,
            block_table=torch.zeros((1, 1), dtype=torch.int32),
        )
