# Project Status

Last updated (UTC): 2026-04-23T02:10:00Z

## Snapshot
- Branch: `main`
- Latest commit: `8b8b21b` (`polish release docs before merging varlen-fa2`)
- Previous commit: `ee1d515` (`set default TORCH_CUDA_ARCH_LIST for RTX 4060`)

## Completed Milestones
- Added FA2 runtime mode switch (`NANOVLLM_FA2_MODE`) with:
  - `varlen_official`, `varlen_man`, `varlen_debug`
  - `batch_official`, `batch_man`, `batch_debug`
- Handwritten batch-view path (`batch_man`) and handwritten varlen path (`varlen_man`) are both wired through torch extension.
- Added debug compare modes (`batch_debug`, `varlen_debug`) and focused correctness/routing tests.
- Fixed varlen GQA head mapping bug (`q_head -> kv_head`) in handwritten varlen kernel.
- Tightened fallback semantics:
  - fallback now applies only to runtime backend execution errors (`RuntimeError`)
  - invalid mode / validation errors no longer fallback
  - `batch_debug` / `varlen_debug` mismatches always hard-fail
- Added default arch behavior for extension loading on this workstation:
  - if `TORCH_CUDA_ARCH_LIST` is unset, default to `8.9` (RTX 4060)

## Temporary Hotfix (Current)
- `batch_man` now applies Python-side per-sequence pad-to-64 before calling handwritten kernel.
- Unaligned input triggers one-time warning tag:
  - `[WARNING][FA2_BATCH_MAN_PAD64]`
- `varlen_man` now applies Python-side max-seqlen align-to-64 before kernel launch when needed.
- Unaligned max-seqlen triggers one-time warning tag:
  - `[WARNING][FA2_VARLEN_MAN_PAD64]`
- These are temporary mitigations until kernel-side tail handling is fully completed.

## Open Items / Risks
- Kernel-level tail/predicate handling for both batch and varlen paths still needs hardening to remove Python-side align hotfixes.
- `varlen_debug` may report BF16 point-wise outliers even when generation succeeds; tolerance strategy may need refinement.

## Next Actions
1. Replace Python align/pad hotfixes with kernel-level boundary/predicate fixes.
2. Continue expanding correctness coverage (dtype/head_dim/shape/GQA) and keep `docs/test.md` in sync.
3. Prepare merge-to-main notes after final mode verification (completed on this branch).

## Validation Snapshot
- Mode test status: completed, no semantic collapse observed in final mode checks.
- Pytest status: `143 passed` on `python3 -m pytest -q tests`.

## Audit Snapshot (2026-04-23)
- Added audit artifacts:
  - `docs/audit_acceptance.md`
  - `docs/audit_report.md`
  - `docs/audit_report.json`
- Audit result: **Pass with risks** (`P0=0`, `P1=4`, `P2=6`).
- Primary risks are doc-command correctness, fallback/debug semantics clarity, and stale status synchronization.

---
Update rule:
- Keep `docs/STATUS.md` (human) and `docs/status.json` (agent) aligned in the same commit.
- Treat status as a snapshot contract: if a change updates milestones, risks, next actions, or reported commit baseline, refresh both files in that same commit.
