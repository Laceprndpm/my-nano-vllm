import pytest
import torch

pytest.importorskip("flash_attn")

import nanovllm.layers.prefill_attention as prefill_attention


@pytest.fixture(autouse=True)
def _reset_backend_settings():
    prefill_attention.configure_prefill_attention_backend("cuda_fa2", False)
    yield
    prefill_attention.configure_prefill_attention_backend("cuda_fa2", True)


def _sample_inputs():
    q = torch.zeros((1, 1, 64), dtype=torch.float16)
    k = torch.zeros((1, 1, 64), dtype=torch.float16)
    v = torch.zeros((1, 1, 64), dtype=torch.float16)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    return q, k, v, cu, cu


@pytest.mark.parametrize(
    "mode,expected_handler",
    [
        ("varlen_official", "_run_cuda_varlen_fa2_official"),
        ("varlen_man", "_run_cuda_varlen_fa2_man"),
        ("varlen_debug", "_run_cuda_varlen_fa2_debug"),
        ("batch_official", "_run_cuda_batch_fa2_official"),
        ("batch_man", "_run_cuda_batch_fa2_man"),
        ("batch_debug", "_run_cuda_batch_fa2_debug"),
    ],
)
def test_nanovllm_fa2_mode_routing_dispatches_expected_handler(monkeypatch, mode, expected_handler):
    q, k, v, cu_q, cu_k = _sample_inputs()
    monkeypatch.setenv("NANOVLLM_FA2_MODE", mode)

    calls = []

    def _make_stub(name):
        def _stub(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return _stub

    monkeypatch.setattr(prefill_attention, "_run_cuda_varlen_fa2_official", _make_stub("_run_cuda_varlen_fa2_official"))
    monkeypatch.setattr(prefill_attention, "_run_cuda_varlen_fa2_man", _make_stub("_run_cuda_varlen_fa2_man"))
    monkeypatch.setattr(prefill_attention, "_run_cuda_varlen_fa2_debug", _make_stub("_run_cuda_varlen_fa2_debug"))
    monkeypatch.setattr(prefill_attention, "_run_cuda_batch_fa2_official", _make_stub("_run_cuda_batch_fa2_official"))
    monkeypatch.setattr(prefill_attention, "_run_cuda_batch_fa2_man", _make_stub("_run_cuda_batch_fa2_man"))
    monkeypatch.setattr(prefill_attention, "_run_cuda_batch_fa2_debug", _make_stub("_run_cuda_batch_fa2_debug"))
    monkeypatch.setattr(
        prefill_attention,
        "_direct_flash_attn_prefill",
        lambda *args, **kwargs: pytest.fail("unexpected fallback to flash-attn in routing test"),
    )

    out = prefill_attention.run_prefill_attention(
        q,
        k,
        v,
        max_seqlen_q=1,
        cu_seqlens_q=cu_q,
        max_seqlen_k=1,
        cu_seqlens_k=cu_k,
        softmax_scale=1.0,
        causal=True,
        block_table=None,
    )

    assert out == expected_handler
    assert len(calls) == 1
    called_name, _args, kwargs = calls[0]
    assert called_name == expected_handler

    if mode.startswith("varlen"):
        assert "block_table" in kwargs
        assert kwargs["block_table"].dtype == torch.int32
        assert kwargs["block_table"].shape == (1, 0)
    else:
        assert "block_table" not in kwargs


def test_invalid_mode_raises_without_fallback_even_if_enabled(monkeypatch):
    q, k, v, cu_q, cu_k = _sample_inputs()
    monkeypatch.setenv("NANOVLLM_FA2_MODE", "batch_officia")
    prefill_attention.configure_prefill_attention_backend("cuda_fa2", True)

    fallback_calls = []
    monkeypatch.setattr(
        prefill_attention,
        "_direct_flash_attn_prefill",
        lambda *args, **kwargs: fallback_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="unsupported NANOVLLM_FA2_MODE"):
        prefill_attention.run_prefill_attention(
            q,
            k,
            v,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=1,
            cu_seqlens_k=cu_k,
            softmax_scale=1.0,
            causal=True,
            block_table=None,
        )
    assert len(fallback_calls) == 0


def test_runtime_error_fallbacks_when_enabled(monkeypatch):
    q, k, v, cu_q, cu_k = _sample_inputs()
    monkeypatch.setenv("NANOVLLM_FA2_MODE", "batch_official")
    prefill_attention.configure_prefill_attention_backend("cuda_fa2", True)

    monkeypatch.setattr(
        prefill_attention,
        "_run_cuda_batch_fa2_official",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kernel launch failed")),
    )
    monkeypatch.setattr(
        prefill_attention,
        "_direct_flash_attn_prefill",
        lambda *args, **kwargs: "fallback_used",
    )

    out = prefill_attention.run_prefill_attention(
        q,
        k,
        v,
        max_seqlen_q=1,
        cu_seqlens_q=cu_q,
        max_seqlen_k=1,
        cu_seqlens_k=cu_k,
        softmax_scale=1.0,
        causal=True,
        block_table=None,
    )
    assert out == "fallback_used"


def test_runtime_error_raises_when_fallback_disabled(monkeypatch):
    q, k, v, cu_q, cu_k = _sample_inputs()
    monkeypatch.setenv("NANOVLLM_FA2_MODE", "batch_official")
    prefill_attention.configure_prefill_attention_backend("cuda_fa2", False)

    monkeypatch.setattr(
        prefill_attention,
        "_run_cuda_batch_fa2_official",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kernel launch failed")),
    )
    monkeypatch.setattr(
        prefill_attention,
        "_direct_flash_attn_prefill",
        lambda *args, **kwargs: pytest.fail("unexpected fallback when disabled"),
    )

    with pytest.raises(RuntimeError, match="kernel launch failed"):
        prefill_attention.run_prefill_attention(
            q,
            k,
            v,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=1,
            cu_seqlens_k=cu_k,
            softmax_scale=1.0,
            causal=True,
            block_table=None,
        )


def test_debug_mode_runtime_error_does_not_fallback_even_if_enabled(monkeypatch):
    q, k, v, cu_q, cu_k = _sample_inputs()
    monkeypatch.setenv("NANOVLLM_FA2_MODE", "batch_debug")
    prefill_attention.configure_prefill_attention_backend("cuda_fa2", True)

    monkeypatch.setattr(
        prefill_attention,
        "_run_cuda_batch_fa2_debug",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("debug kernel failed")),
    )
    monkeypatch.setattr(
        prefill_attention,
        "_direct_flash_attn_prefill",
        lambda *args, **kwargs: pytest.fail("unexpected fallback in debug mode"),
    )

    with pytest.raises(RuntimeError, match="debug kernel failed"):
        prefill_attention.run_prefill_attention(
            q,
            k,
            v,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=1,
            cu_seqlens_k=cu_k,
            softmax_scale=1.0,
            causal=True,
            block_table=None,
        )
