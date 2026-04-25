import pytest
import torch

import nanovllm.layers.prefill_attention as prefill_attention


def _empty_inputs(device: str = "cpu"):
    q = torch.empty((0, 1, 64), dtype=torch.float16, device=device)
    k = torch.empty((0, 1, 64), dtype=torch.float16, device=device)
    v = torch.empty((0, 1, 64), dtype=torch.float16, device=device)
    cu = torch.tensor([0, 0], dtype=torch.int32, device=device)
    return q, k, v, cu, cu


def _one_token_inputs(device: str = "cpu"):
    q = torch.ones((64, 1, 64), dtype=torch.float16, device=device)
    k = torch.ones((64, 1, 64), dtype=torch.float16, device=device)
    v = torch.ones((64, 1, 64), dtype=torch.float16, device=device)
    cu = torch.tensor([0, 64], dtype=torch.int32, device=device)
    block_table = torch.empty((1, 0), dtype=torch.int32, device=device)
    return q, k, v, cu, cu, block_table


def _install_nvtx_spy(monkeypatch):
    calls = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: calls.append(("push", name)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: calls.append(("pop", None)))
    return calls


def test_nvtx_disabled_by_default_no_push_pop(monkeypatch):
    monkeypatch.delenv("NANOVLLM_NVTX", raising=False)
    calls = _install_nvtx_spy(monkeypatch)

    q, k, v, cu_q, cu_k, _ = _one_token_inputs()
    monkeypatch.setattr(prefill_attention, "flash_attn_func", lambda q, k, v, **kwargs: q.clone())
    prefill_attention._run_cuda_batch_fa2_official(
        q,
        k,
        v,
        max_seqlen_q=64,
        cu_seqlens_q=cu_q,
        max_seqlen_k=64,
        cu_seqlens_k=cu_k,
        softmax_scale=1.0,
        causal=True,
    )

    assert calls == []


def test_nvtx_enabled_wraps_batch_official_and_batch_man(monkeypatch):
    monkeypatch.setenv("NANOVLLM_NVTX", "1")
    calls = _install_nvtx_spy(monkeypatch)

    q, k, v, cu_q, cu_k, _ = _one_token_inputs()
    monkeypatch.setattr(prefill_attention, "flash_attn_func", lambda q, k, v, **kwargs: q.clone())
    monkeypatch.setattr(prefill_attention, "torch_ext_fa2_batch_fwd", lambda q, k, v, **kwargs: q.clone())

    prefill_attention._run_cuda_batch_fa2_official(
        q,
        k,
        v,
        max_seqlen_q=64,
        cu_seqlens_q=cu_q,
        max_seqlen_k=64,
        cu_seqlens_k=cu_k,
        softmax_scale=1.0,
        causal=True,
    )
    prefill_attention._run_cuda_batch_fa2_man(
        q,
        k,
        v,
        max_seqlen_q=64,
        cu_seqlens_q=cu_q,
        max_seqlen_k=64,
        cu_seqlens_k=cu_k,
        softmax_scale=1.0,
        causal=True,
    )

    assert calls == [
        ("push", "prefill.batch_official.fa2_call"),
        ("pop", None),
        ("push", "prefill.batch_man.fa2_call"),
        ("pop", None),
    ]


def test_nvtx_enabled_wraps_varlen_official_and_varlen_man(monkeypatch):
    monkeypatch.setenv("NANOVLLM_NVTX", "1")
    calls = _install_nvtx_spy(monkeypatch)

    q, k, v, cu_q, cu_k, block_table = _one_token_inputs()
    monkeypatch.setattr(prefill_attention, "torch_ext_fa2_varlen_fwd_official", lambda *args, **kwargs: q.clone())
    monkeypatch.setattr(prefill_attention, "torch_ext_fa2_varlen_fwd_man", lambda *args, **kwargs: q.clone())

    prefill_attention._run_cuda_varlen_fa2_official(
        q,
        k,
        v,
        max_seqlen_q=64,
        cu_seqlens_q=cu_q,
        max_seqlen_k=64,
        cu_seqlens_k=cu_k,
        block_table=block_table,
        softmax_scale=1.0,
        causal=True,
    )
    prefill_attention._run_cuda_varlen_fa2_man(
        q,
        k,
        v,
        max_seqlen_q=64,
        cu_seqlens_q=cu_q,
        max_seqlen_k=64,
        cu_seqlens_k=cu_k,
        block_table=block_table,
        softmax_scale=1.0,
        causal=True,
    )

    assert calls == [
        ("push", "prefill.varlen_official.fa2_call"),
        ("pop", None),
        ("push", "prefill.varlen_man.fa2_call"),
        ("pop", None),
    ]


def test_nvtx_pop_on_exception(monkeypatch):
    monkeypatch.setenv("NANOVLLM_NVTX", "1")
    calls = _install_nvtx_spy(monkeypatch)

    with pytest.raises(RuntimeError, match="boom"):
        with prefill_attention._nvtx_range("prefill.batch_man"):
            raise RuntimeError("boom")

    assert calls == [
        ("push", "prefill.batch_man"),
        ("pop", None),
    ]
