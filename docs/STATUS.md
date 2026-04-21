# Project Status

Last updated (UTC): 2026-04-21T16:40:00Z

## Snapshot
- Branch: `varlen-fa2`
- Latest commit: `89972df` (`fix GQA kv-head mapping in varlen_man kernel`)
- Previous commit: `1b6d65d` (`add GQA varlen_man correctness regression tests`)

## Completed Milestones
- Added FA2 runtime mode switch (`NANOVLLM_FA2_MODE`) with:
  - `varlen_official`, `varlen_man`, `varlen_debug`
  - `batch_official`, `batch_man`, `batch_debug`
- Handwritten batch-view path (`batch_man`) and handwritten varlen path (`varlen_man`) are both wired through torch extension.
- Added debug compare modes (`batch_debug`, `varlen_debug`) and focused correctness/routing tests.
- Fixed varlen GQA head mapping bug (`q_head -> kv_head`) in handwritten varlen kernel.

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

---
Update rule:
- Keep `docs/STATUS.md` (human) and `docs/status.json` (agent) aligned in the same commit.
