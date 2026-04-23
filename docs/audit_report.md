# Documentation Audit Report

Audit date (UTC): 2026-04-23
Branch: `varlen-fa2`
HEAD: `714eec7` (`rewrite varlen_fwd to CUTE-tile kernel structure`)

## Summary
- Result: **Pass with risks** (meets gate: `P0=0`).
- Score: **81/100**.
- Findings: `P1=4`, `P2=6`, `INFO=2`.
- Method: 4 parallel subagent inspections + local evidence checks (`pytest --collect-only`, routing grep, git snapshot).

## Top Findings (Prioritized)

### P1
1. `AGENTS.md` test command may not execute pytest cases.
- Evidence: `AGENTS.md:21` uses `python tests/test_flash_attn_correctness.py`; test file has no `__main__` runner (`tests/test_flash_attn_correctness.py`).
- Impact: false sense of test pass.
- Recommendation: switch to `python3 -m pytest -q tests/test_flash_attn_correctness.py`.

2. `docs/STATUS.md` and `docs/status.json` snapshot is stale vs current HEAD.
- Evidence: both still reference `89972df` and `2026-04-21`; current HEAD is `714eec7` (`2026-04-22`).
- Impact: operational confusion during debugging and handoff.
- Recommendation: update status snapshot in same change whenever project state is reported.

3. Invalid/typo `NANOVLLM_FA2_MODE` can be swallowed by fallback path.
- Evidence: `prefill_attention.py` raises unsupported mode then broad `except` + fallback branch.
- Impact: config errors may silently degrade to flash-attn.
- Recommendation: keep mode validation outside broad fallback or let `ValueError` pass through.

4. `debug` mismatch assert can be bypassed when fallback is enabled.
- Evidence: debug paths raise `AssertionError`, outer fallback catches and continues.
- Impact: numeric regressions may be hidden.
- Recommendation: disable fallback for `*_debug` modes or re-raise assertion-class errors.

### P2
1. NVTX include pattern inconsistent (`prefill.batch_man/` vs `prefill.batch_man`).
2. `python` vs `python3` command style inconsistent.
3. `docs/test.md` overstates dtype validation for `test_varlen_api_contract.py`.
4. `docs/test.md` under-describes existing varlen GQA and unsupported head-dim negative test coverage.
5. `batch_man` correctness tests currently do not directly cover GQA/MQA in that specific file.
6. `block_table required` wording differs from runtime behavior (`None` normalized to empty tensor in varlen path).

### INFO
1. `docs/test.md` file inventory matches actual `tests/` files and collect count (`137`).
2. FA2 mode matrix/default routing in docs and code is mostly aligned.

## Recommended Fix Order
1. Fix command correctness + snapshot freshness (`AGENTS.md`, `STATUS.md`, `status.json`).
2. Clarify mode/fallback/debug semantics in runtime design docs.
3. Normalize command/NVTX style across docs.
4. Refresh test coverage statements in `docs/test.md`.

## Evidence Sources
- `AGENTS.md`, `README.md`, `docs/STATUS.md`, `docs/status.json`, `docs/test.md`, `docs/varlen_api_design.md`
- `nanovllm/layers/prefill_attention.py`, `nanovllm/layers/cuda_fa2_torch_ext.py`
- `tests/*.py`, `python3 -m pytest -q tests --collect-only` (137 collected)
