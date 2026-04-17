# Project Status

Last updated (UTC): 2026-04-17T08:28:08Z

## Snapshot
- Branch: `master`
- Latest commit: `684692f` (`Save progress on CUDA FA2 integration`)
- Previous commit: `d87e769` (`Fix flash-attn correctness test for pytest collection`)

## Completed Milestones
- Added prefill batch-view adapter and backend dispatch for `cuda_fa2` with flash-attn fallback path.
- Added torch cpp extension scaffold and integrated FA2 extension loading pipeline.
- Landed CUTLASS/CUTE-compatible handwritten FA2 source set under `nanovllm/csrc/fa2`.
- Fixed test collection for flash-attn correctness by converting script-style checks to pytest tests.

## Validation Summary
- `python3 -m pytest -q tests/test_flash_attn_correctness.py` -> `3 passed`
- `python3 -m pytest -q tests/test_prefill_attention_batch_view.py` -> `4 passed`
- `python3 example.py` with `NANOVLLM_FA2_USE_TORCH_EXT=1` completed generation successfully.
- Numerical check (`fa2_fwd` vs `flash_attn_func`) passed with tight match after layout fix.

## Current Workspace State
- Untracked directories remain: `deps/`, `third_party/`.
- No other tracked-file modifications pending at snapshot time.

## Open Items / Risks
- Handwritten CUDA FA2 kernel is integrated but still treated as in-progress hardening work.
- Shape support is currently conservative in the torch-ext route to avoid unsafe launches.
- Need broader performance and stability sweep before making CUDA path default.

## Next Actions
1. Expand shape/head-dim coverage and add targeted correctness tests.
2. Add stress/regression tests for long sequence and multi-head/multi-batch cases.
3. Decide policy for tracking or ignoring `deps/` and `third_party/` in this repo.

---
Update rule:
- Keep `docs/STATUS.md` (human) and `docs/status.json` (agent) aligned in the same commit.
