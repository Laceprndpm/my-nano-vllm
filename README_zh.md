# Nano-vLLM + 自定义 CUDA FA2

本仓库基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 构建，并扩展了自定义 CUDA FlashAttention2（FA2）集成工作流。

该项目目标是在保留 nano-vllm 轻量运行时的同时，持续迭代手写 CUDA attention kernel、正确性校验，以及性能分析/调试工具链。

## 本仓库新增内容

- Prefill attention 的运行时后端路由（`flash_attn` / `cuda_fa2`）。
- 通过 `NANOVLLM_FA2_MODE` 切换 FA2 模式：
  - `varlen_official`
  - `varlen_man`（通过 Torch extension 走手写 CUDA varlen 路径）
  - `varlen_debug`（同时运行 official + manual 并断言差异）
  - `batch_official`
  - `batch_man`（通过 Torch extension 走手写 CUDA 路径）
  - `batch_debug`（同时运行 official + manual 并断言差异）
- 手写 CUDA extension 入口位于 `nanovllm/csrc/fa2`。
- 支持 CUDA profiling，并可选启用 NVTX range：
  - `prefill.batch_official.fa2_call`
  - `prefill.batch_man.fa2_call`
  - `prefill.varlen_official.fa2_call`
  - `prefill.varlen_man.fa2_call`

## 安装

```bash
python3 -m pip install -e .
```

请使用启用 CUDA 的环境，并确保 `torch`、`triton`、`flash-attn` 版本兼容。

针对 RTX 4060（sm_89），若未显式设置变量，本仓库在加载本地 CUDA extension 时默认使用 `TORCH_CUDA_ARCH_LIST=8.9`。你仍可手动覆盖该设置。

## 快速开始

```bash
python3 example.py
```

你可以在启动时选择 FA2 运行模式：

```bash
NANOVLLM_FA2_MODE=batch_man python3 example.py
```

Varlen 调试示例：

```bash
NANOVLLM_FA2_MODE=varlen_debug python3 example.py
```

## 测试

运行项目测试（本仓库推荐命令）：

```bash
python3 -m pytest -q tests
```

## 性能分析（NCU + NVTX）

仅在需要时启用 NVTX：

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_man \
ncu --target-processes all --nvtx --nvtx-include "prefill.batch_man.fa2_call/" \
python3 example.py
```

针对一个代表性 batch launch 进行详细分析（`id=9`，`--set full`）：

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_man \
ncu --set full --target-processes all --nvtx \
--nvtx-include "prefill.batch_man.fa2_call/" \
--launch-skip 9 --launch-count 1 \
-o reports/batch_man_id9_full \
python3 example.py
```

对所有匹配的 batch launch 执行快速全窗口扫描（`--set basic`）：

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_man \
ncu --set basic --target-processes all --nvtx \
--nvtx-include "prefill.batch_man.fa2_call/" \
-o reports/batch_man_all_basic \
python3 example.py
```

官方 batch 路径：

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=batch_official \
ncu --target-processes all --nvtx --nvtx-include "prefill.batch_official.fa2_call/" \
python3 example.py
```

手写 varlen 路径：

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=varlen_man \
ncu --target-processes all --nvtx --nvtx-include "prefill.varlen_man.fa2_call/" \
python3 example.py
```

针对一个代表性 launch 进行详细分析（`id=1`，`--set full`）：

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=varlen_man \
ncu --set full --target-processes all --nvtx \
--nvtx-include "prefill.varlen_man.fa2_call/" \
--launch-skip 1 --launch-count 1 \
-o reports/varlen_man_id1_full \
python3 example.py
```

对所有匹配 launch 执行快速全窗口扫描（`--set basic`）：

```bash
NANOVLLM_NVTX=1 NANOVLLM_FA2_MODE=varlen_man \
ncu --set basic --target-processes all --nvtx \
--nvtx-include "prefill.varlen_man.fa2_call/" \
-o reports/varlen_man_all_basic \
python3 example.py
```

## 说明

- 手写 **batch** kernel 路径已实现并经过测试。
- 手写 **varlen** kernel 路径已实现（最小 kernel），当前约束如下：
  - `head_dim in {64, 128}`
  - 必要时会在 Python 侧应用最大序列长度对齐热修复
- `cuda_fa2` 路由的回退策略：
  - 仅对运行时后端执行错误（`RuntimeError`）生效
  - 非法模式/校验错误会快速失败（不回退）
  - `batch_debug` / `varlen_debug` 在 mismatch 时不回退
