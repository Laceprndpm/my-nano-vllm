# Project Status

Last updated (UTC): 2026-04-19T07:13:31Z

## Snapshot
- Branch: `master`
- Latest commit: `cf54991` (`Add minimal varlen FA2 API contract and integration`)
- Previous commit: `7582dd0` (`Add project status docs`)

## Completed Milestones
- Added minimal varlen FA2 API contract at Python/extension boundary.
- Added varlen API contract tests and docs for agent/human readers.
- Added runtime mode switch in prefill path via `NANOVLLM_FA2_MODE`.
- Restored handwritten batch-view path (`batch_man`) using torch bind `fa2_fwd`.
- Kept official batch-view path (`batch_official`) and varlen placeholders.

## Runtime Modes (Current)
- `varlen_official`: enabled, routes via varlen wrapper (placeholder backend).
- `varlen_man`: placeholder, currently forwards to `varlen_official`.
- `batch_official`: enabled, batch-view via official flash-attn path.
- `batch_man`: enabled, batch-view via handwritten torch-bind `fa2_fwd`.

## Current Workspace State
- Working tree contains local edits for mode routing + docs + tests updates.
- Untracked directories remain: `deps/`, `third_party/`.

## Open Items / Risks
- Handwritten CUDA varlen kernel remains pending.
- `varlen_man` is currently placeholder behavior, not independent kernel path yet.
- Expanded shape/perf validation should be done after routing stabilization.

## Next Actions
1. Wire true handwritten CUDA varlen kernel to replace `varlen_man` placeholder.
2. Run full regression + mode-matrix tests after integration freeze.
3. Decide repository policy for `deps/` and `third_party/` tracking.

---
Update rule:
- Keep `docs/STATUS.md` (human) and `docs/status.json` (agent) aligned in the same commit.
