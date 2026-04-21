import pytest
import torch

import nanovllm.layers.prefill_attention as prefill_attention


def _sample_inputs(device: str = "cpu"):
    q = torch.zeros((3, 2, 4), dtype=torch.float16, device=device)
    k = torch.zeros((3, 2, 4), dtype=torch.float16, device=device)
    v = torch.zeros((3, 2, 4), dtype=torch.float16, device=device)
    cu = torch.tensor([0, 3], dtype=torch.int32, device=device)
    block_table = torch.empty((1, 0), dtype=torch.int32, device=device)
    return q, k, v, cu, cu, block_table


def test_varlen_debug_returns_man_output_when_close(monkeypatch):
    q, k, v, cu_q, cu_k, block_table = _sample_inputs()

    out_man = torch.randn((3, 2, 4), dtype=torch.float16)
    out_official = out_man + 1e-4

    monkeypatch.setattr(prefill_attention, "_run_cuda_varlen_fa2_man", lambda *args, **kwargs: out_man)
    monkeypatch.setattr(prefill_attention, "_run_cuda_varlen_fa2_official", lambda *args, **kwargs: out_official)

    out = prefill_attention._run_cuda_varlen_fa2_debug(
        q,
        k,
        v,
        max_seqlen_q=3,
        cu_seqlens_q=cu_q,
        max_seqlen_k=3,
        cu_seqlens_k=cu_k,
        block_table=block_table,
        softmax_scale=1.0,
        causal=True,
    )

    assert torch.equal(out, out_man)


def test_varlen_debug_raises_on_mismatch(monkeypatch):
    q, k, v, cu_q, cu_k, block_table = _sample_inputs()

    out_man = torch.zeros((3, 2, 4), dtype=torch.float16)
    out_official = torch.ones((3, 2, 4), dtype=torch.float16)

    monkeypatch.setattr(prefill_attention, "_run_cuda_varlen_fa2_man", lambda *args, **kwargs: out_man)
    monkeypatch.setattr(prefill_attention, "_run_cuda_varlen_fa2_official", lambda *args, **kwargs: out_official)

    with pytest.raises(AssertionError, match="max_abs_error=.*mean_abs_error"):
        prefill_attention._run_cuda_varlen_fa2_debug(
            q,
            k,
            v,
            max_seqlen_q=3,
            cu_seqlens_q=cu_q,
            max_seqlen_k=3,
            cu_seqlens_k=cu_k,
            block_table=block_table,
            softmax_scale=1.0,
            causal=True,
        )


def test_varlen_debug_raises_on_shape_mismatch(monkeypatch):
    q, k, v, cu_q, cu_k, block_table = _sample_inputs()

    out_man = torch.zeros((3, 2, 4), dtype=torch.float16)
    out_official = torch.zeros((2, 2, 4), dtype=torch.float16)

    monkeypatch.setattr(prefill_attention, "_run_cuda_varlen_fa2_man", lambda *args, **kwargs: out_man)
    monkeypatch.setattr(prefill_attention, "_run_cuda_varlen_fa2_official", lambda *args, **kwargs: out_official)

    with pytest.raises(AssertionError, match="shape differs"):
        prefill_attention._run_cuda_varlen_fa2_debug(
            q,
            k,
            v,
            max_seqlen_q=3,
            cu_seqlens_q=cu_q,
            max_seqlen_k=3,
            cu_seqlens_k=cu_k,
            block_table=block_table,
            softmax_scale=1.0,
            causal=True,
        )


def test_varlen_debug_mode_dispatches_from_run_prefill_attention(monkeypatch):
    q, k, v, cu_q, cu_k, _ = _sample_inputs()
    monkeypatch.setenv("NANOVLLM_FA2_MODE", "varlen_debug")
    prefill_attention.configure_prefill_attention_backend("cuda_fa2", False)

    out_expected = torch.randn((3, 2, 4), dtype=torch.float16)
    calls = []

    def _stub(*args, **kwargs):
        calls.append(kwargs)
        return out_expected

    monkeypatch.setattr(prefill_attention, "_run_cuda_varlen_fa2_debug", _stub)
    monkeypatch.setattr(
        prefill_attention,
        "_direct_flash_attn_prefill",
        lambda *args, **kwargs: pytest.fail("unexpected fallback to flash-attn in varlen_debug routing test"),
    )

    out = prefill_attention.run_prefill_attention(
        q,
        k,
        v,
        max_seqlen_q=3,
        cu_seqlens_q=cu_q,
        max_seqlen_k=3,
        cu_seqlens_k=cu_k,
        softmax_scale=1.0,
        causal=True,
        block_table=None,
    )

    assert torch.equal(out, out_expected)
    assert len(calls) == 1
    assert "block_table" in calls[0]
    assert calls[0]["block_table"].dtype == torch.int32
    assert calls[0]["block_table"].shape == (1, 0)
