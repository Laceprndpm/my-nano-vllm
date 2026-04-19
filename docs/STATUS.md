# Project Status

Last updated (UTC): 2026-04-19T07:38:51Z

## Snapshot
- Branch: `master`
- Latest commit: `4005e1c` (`Checkpoint before batch_man padding hotfix`)
- Previous commit: `cf54991` (`Add minimal varlen FA2 API contract and integration`)

## Completed Milestones
- Added FA2 runtime mode switch (`NANOVLLM_FA2_MODE`) with `varlen_official`, `varlen_man`, `batch_official`, `batch_man`.
- Restored handwritten batch-view path (`batch_man`) and kept official batch-view path.
- Added mode routing tests and test inventory docs.

## Temporary Hotfix (Current)
- `batch_man` now applies Python-side per-sequence pad-to-64 before calling handwritten kernel.
- Unaligned input triggers one-time warning tag:
  - `[WARNING][FA2_BATCH_MAN_PAD64]`
- This is a temporary mitigation for non-64-aligned sequence degradation.

## Open Items / Risks
- Handwritten CUDA varlen kernel remains pending.
- Kernel-level tail/predicate handling in handwritten batch kernel is still not fully fixed.

## Next Actions
1. Replace Python pad64 hotfix with kernel-level boundary/predicate fixes.
2. Keep validating mode matrix correctness and quality after each kernel change.

---
Update rule:
- Keep `docs/STATUS.md` (human) and `docs/status.json` (agent) aligned in the same commit.
