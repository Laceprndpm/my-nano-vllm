# Nano-vLLM + Custom CUDA FA2

This repository is based on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and extended with a custom CUDA FlashAttention2 (FA2) integration workflow.

The goal of this project is to keep the lightweight nano-vllm runtime while iterating on handwritten CUDA attention kernels, correctness checks, and profiling/debug tooling.

## What Is Added in This Repo

- Runtime backend routing for prefill attention (`flash_attn` / `cuda_fa2`).
- FA2 mode switch via `NANOVLLM_FA2_MODE`:
  - `varlen_official`
  - `varlen_man` (placeholder path today)
  - `batch_official`
  - `batch_man` (handwritten CUDA path via Torch extension)
  - `batch_debug` (run official + manual and assert diff)
- Handwritten CUDA extension entry points under `nanovllm/csrc/fa2`.
- CUDA profiling support with optional NVTX ranges:
  - `prefill.batch_official`
  - `prefill.batch_man`

## Installation

```bash
python3 -m pip install -e .
```

Use a CUDA-enabled environment with compatible `torch`, `triton`, and `flash-attn`.

## Quick Start

```bash
python3 example.py
```

You can select FA2 runtime mode at launch:

```bash
NANOVLLM_FA2_MODE=batch_man python3 example.py
```

## Test

Run project tests (recommended command for this repo):

```bash
python3 -m pytest -q tests
```

## Profiling (NCU + NVTX)

Enable NVTX only when needed:

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_man \
ncu --target-processes all --nvtx --nvtx-include "prefill.batch_man/" \
python3 example.py
```

For the official batch path:

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_official \
ncu --target-processes all --nvtx --nvtx-include "prefill.batch_official/" \
python3 example.py
```

## Notes

- The handwritten **batch** kernel path is implemented and tested.
- The handwritten **varlen** CUDA kernel is not implemented yet; `varlen_man` currently remains a placeholder route.
