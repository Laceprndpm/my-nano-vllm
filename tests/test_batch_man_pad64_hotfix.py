import warnings

import torch

import nanovllm.layers.prefill_attention as prefill_attention


def _build_cu_seqlens(lengths: list[int], device: str) -> torch.Tensor:
    out = [0]
    for x in lengths:
        out.append(out[-1] + x)
    return torch.tensor(out, dtype=torch.int32, device=device)


def test_batch_man_pad64_emits_warning_and_pads(monkeypatch):
    monkeypatch.setattr(prefill_attention, "_BATCH_MAN_PAD64_WARNING_EMITTED", False)

    called_shapes = []

    def _fake_fwd(q, k, v, *, causal, softmax_scale):
        called_shapes.append((tuple(q.shape), tuple(k.shape), tuple(v.shape), causal, softmax_scale))
        return q.clone(), torch.zeros((q.size(0), q.size(1), q.size(2)), dtype=torch.float32, device=q.device)

    monkeypatch.setattr(prefill_attention, "torch_ext_fa2_batch_fwd", _fake_fwd)

    q = torch.randn(63, 4, 64, dtype=torch.float16)
    k = torch.randn(63, 4, 64, dtype=torch.float16)
    v = torch.randn(63, 4, 64, dtype=torch.float16)
    cu = _build_cu_seqlens([63], device="cpu")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = prefill_attention._run_cuda_batch_fa2_man(
            q,
            k,
            v,
            max_seqlen_q=63,
            cu_seqlens_q=cu,
            max_seqlen_k=63,
            cu_seqlens_k=cu,
            softmax_scale=1.0 / (64 ** 0.5),
            causal=True,
        )

    assert out.shape == q.shape
    assert called_shapes and called_shapes[0][0][1] == 64
    assert called_shapes[0][1][1] == 64
    assert any("[WARNING][FA2_BATCH_MAN_PAD64]" in str(w.message) for w in caught)


def test_batch_man_no_warning_when_aligned(monkeypatch):
    monkeypatch.setattr(prefill_attention, "_BATCH_MAN_PAD64_WARNING_EMITTED", False)

    called_shapes = []

    def _fake_fwd(q, k, v, *, causal, softmax_scale):
        called_shapes.append((tuple(q.shape), tuple(k.shape), tuple(v.shape), causal, softmax_scale))
        return q.clone(), torch.zeros((q.size(0), q.size(1), q.size(2)), dtype=torch.float32, device=q.device)

    monkeypatch.setattr(prefill_attention, "torch_ext_fa2_batch_fwd", _fake_fwd)

    q = torch.randn(64, 4, 64, dtype=torch.float16)
    k = torch.randn(64, 4, 64, dtype=torch.float16)
    v = torch.randn(64, 4, 64, dtype=torch.float16)
    cu = _build_cu_seqlens([64], device="cpu")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = prefill_attention._run_cuda_batch_fa2_man(
            q,
            k,
            v,
            max_seqlen_q=64,
            cu_seqlens_q=cu,
            max_seqlen_k=64,
            cu_seqlens_k=cu,
            softmax_scale=1.0 / (64 ** 0.5),
            causal=True,
        )

    assert out.shape == q.shape
    assert called_shapes and called_shapes[0][0][1] == 64
    assert not any("[WARNING][FA2_BATCH_MAN_PAD64]" in str(w.message) for w in caught)
