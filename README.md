# Nano-vLLM + Custom CUDA FA2

This repository is based on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and extended with a custom CUDA FlashAttention2 (FA2) integration workflow.

The goal of this project is to keep the lightweight nano-vllm runtime while iterating on handwritten CUDA attention kernels, correctness checks, and profiling/debug tooling.

## What Is Added in This Repo

- Runtime backend routing for prefill attention (`flash_attn` / `cuda_fa2`).
- FA2 mode switch via `NANOVLLM_FA2_MODE`:
  - `varlen_official`
  - `varlen_man` (handwritten CUDA varlen path via Torch extension)
  - `varlen_debug` (run official + manual and assert diff)
  - `batch_official`
  - `batch_man` (handwritten CUDA path via Torch extension)
  - `batch_debug` (run official + manual and assert diff)
- Handwritten CUDA extension entry points under `nanovllm/csrc/fa2`.
- CUDA profiling support with optional NVTX ranges:
  - `prefill.batch_official.fa2_call`
  - `prefill.batch_man.fa2_call`
  - `prefill.varlen_official.fa2_call`
  - `prefill.varlen_man.fa2_call`

## Installation

```bash
python3 -m pip install -e .
```

Use a CUDA-enabled environment with compatible `torch`, `triton`, and `flash-attn`.

For RTX 4060 (sm_89), this repo now defaults `TORCH_CUDA_ARCH_LIST=8.9` when loading the local CUDA extension if the variable is unset. You can still override it explicitly.

## Quick Start

```bash
python3 example.py
```

You can select FA2 runtime mode at launch:

```bash
NANOVLLM_FA2_MODE=batch_man python3 example.py
```

Varlen debug example:

```bash
NANOVLLM_FA2_MODE=varlen_debug python3 example.py
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
ncu --target-processes all --nvtx --nvtx-include "prefill.batch_man.fa2_call/" \
python3 example.py
```

Detailed profile for one representative batch launch (`id=9`, `--set full`):

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_man \
ncu --set full --target-processes all --nvtx \
--nvtx-include "prefill.batch_man.fa2_call/" \
--launch-skip 9 --launch-count 1 \
-o reports/batch_man_id9_full \
python3 example.py
```

Fast full-window scan across all matching batch launches (`--set basic`):

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_man \
ncu --set basic --target-processes all --nvtx \
--nvtx-include "prefill.batch_man.fa2_call/" \
-o reports/batch_man_all_basic \
python3 example.py
```

For the official batch path:

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_official \
ncu --target-processes all --nvtx --nvtx-include "prefill.batch_official.fa2_call/" \
python3 example.py
```

For handwritten varlen path:

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=varlen_man \
ncu --target-processes all --nvtx --nvtx-include "prefill.varlen_man.fa2_call/" \
python3 example.py
```

Detailed profile for one representative launch (`id=1`, `--set full`):

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=varlen_man \
ncu --set full --target-processes all --nvtx \
--nvtx-include "prefill.varlen_man.fa2_call/" \
--launch-skip 1 --launch-count 1 \
-o reports/varlen_man_id1_full \
python3 example.py
```

Fast full-window scan across all matching launches (`--set basic`):

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=varlen_man \
ncu --set basic --target-processes all --nvtx \
--nvtx-include "prefill.varlen_man.fa2_call/" \
-o reports/varlen_man_all_basic \
python3 example.py
```

## Notes

- The handwritten **batch** kernel path is implemented and tested.
- The handwritten **varlen** kernel path is implemented (minimal kernel) with current constraints:
  - `head_dim in {64, 128}`
  - max sequence length alignment hotfix is applied in Python when needed
- Fallback policy for `cuda_fa2` routing:
  - fallback applies to runtime backend execution errors (`RuntimeError`) only
  - invalid mode / validation errors fail fast (no fallback)
  - `batch_debug` / `varlen_debug` do not fallback on mismatches
