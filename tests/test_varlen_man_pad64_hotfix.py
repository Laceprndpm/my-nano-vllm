import warnings

import torch

import nanovllm.layers.prefill_attention as prefill_attention


def _build_cu_seqlens(lengths: list[int], device: str) -> torch.Tensor:
    out = [0]
    for x in lengths:
        out.append(out[-1] + x)
    return torch.tensor(out, dtype=torch.int32, device=device)


def test_varlen_man_pad64_emits_warning_and_aligns_max_seqlen(monkeypatch):
    monkeypatch.setattr(prefill_attention, "_VARLEN_MAN_PAD64_WARNING_EMITTED", False)

    captured = {}

    def _fake_varlen_man(q, k, v, *, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k, block_table, softmax_scale, causal):
        captured["max_seqlen_q"] = max_seqlen_q
        captured["max_seqlen_k"] = max_seqlen_k
        return q.clone()

    monkeypatch.setattr(prefill_attention, "torch_ext_fa2_varlen_fwd_man", _fake_varlen_man)

    q = torch.randn(63, 4, 64, dtype=torch.float16)
    k = torch.randn(63, 4, 64, dtype=torch.float16)
    v = torch.randn(63, 4, 64, dtype=torch.float16)
    cu = _build_cu_seqlens([63], device="cpu")
    block_table = torch.empty((1, 0), dtype=torch.int32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = prefill_attention._run_cuda_varlen_fa2_man_placeholder(
            q,
            k,
            v,
            max_seqlen_q=63,
            cu_seqlens_q=cu,
            max_seqlen_k=63,
            cu_seqlens_k=cu,
            block_table=block_table,
            softmax_scale=1.0 / (64**0.5),
            causal=True,
        )

    assert out.shape == q.shape
    assert captured["max_seqlen_q"] == 64
    assert captured["max_seqlen_k"] == 64
    assert any("[WARNING][FA2_VARLEN_MAN_PAD64]" in str(w.message) for w in caught)


def test_varlen_man_no_warning_when_max_seqlen_aligned(monkeypatch):
    monkeypatch.setattr(prefill_attention, "_VARLEN_MAN_PAD64_WARNING_EMITTED", False)

    captured = {}

    def _fake_varlen_man(q, k, v, *, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k, block_table, softmax_scale, causal):
        captured["max_seqlen_q"] = max_seqlen_q
        captured["max_seqlen_k"] = max_seqlen_k
        return q.clone()

    monkeypatch.setattr(prefill_attention, "torch_ext_fa2_varlen_fwd_man", _fake_varlen_man)

    q = torch.randn(64, 4, 64, dtype=torch.float16)
    k = torch.randn(64, 4, 64, dtype=torch.float16)
    v = torch.randn(64, 4, 64, dtype=torch.float16)
    cu = _build_cu_seqlens([64], device="cpu")
    block_table = torch.empty((1, 0), dtype=torch.int32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = prefill_attention._run_cuda_varlen_fa2_man_placeholder(
            q,
            k,
            v,
            max_seqlen_q=64,
            cu_seqlens_q=cu,
            max_seqlen_k=64,
            cu_seqlens_k=cu,
            block_table=block_table,
            softmax_scale=1.0 / (64**0.5),
            causal=True,
        )

    assert out.shape == q.shape
    assert captured["max_seqlen_q"] == 64
    assert captured["max_seqlen_k"] == 64
    assert not any("[WARNING][FA2_VARLEN_MAN_PAD64]" in str(w.message) for w in caught)
