# TODO

## Testing Refactor Backlog

This backlog tracks test-suite cleanup and coverage hardening for FA2 modes and handwritten kernels.

## Top Priority

### P0-0: Implement `varlen_man` handwritten CUDA path
- Priority: P0
- Owner: TBD
- Status: DONE
- Why: `varlen_man` now routes to handwritten CUDA varlen forward end-to-end, with dedicated correctness and pad64-hotfix tests.
- Exit Condition: Satisfied (`NANOVLLM_FA2_MODE=varlen_man` no longer forwards to official path).

## Priority Queue

### P0-1: Merge BF16 diagnosis into main correctness matrix
- Priority: P0
- Owner: TBD
- Status: DONE
- Why: Batch correctness matrix is now dtype-parameterized (`fp16`/`bf16`) in one file.
- Exit Condition: Satisfied (`test_batch_man_bf16_diagnosis.py` removed, coverage merged into `test_batch_man_correctness.py`).

### P0-2: Expand batch-man correctness shape coverage
- Priority: P0
- Owner: TBD
- Status: TODO
- Why: Current kernel correctness is biased toward aligned/equal-length cases.
- Exit Condition: correctness matrix includes non-64-aligned lengths and representative tail cases (for example 65/127/255), with stable pass criteria.

### P0-3: Extend routing negative-path tests
- Priority: P0
- Owner: TBD
- Status: TODO
- Why: routing tests mainly check happy-path dispatch and miss fallback/error branches.
- Exit Condition: tests cover unsupported mode behavior and fallback on/off branches explicitly.

### P1-1: Split batch-view mixed-responsibility tests
- Priority: P1
- Owner: TBD
- Status: TODO
- Why: `test_prefill_attention_batch_view.py` currently mixes adapter, masking, and backend-equivalence concerns.
- Exit Condition: adapter/roundtrip checks and masking/behavior checks are separated into focused files.

### P1-2: Add real CUDA integration case for `batch_debug`
- Priority: P1
- Owner: TBD
- Status: TODO
- Why: current `batch_debug` tests are mostly monkeypatched behavior checks.
- Exit Condition: at least one non-mock CUDA integration case validates `batch_debug` end-to-end for an aligned shape.

### P1-3: Keep docs/test matrix synchronized
- Priority: P1
- Owner: TBD
- Status: TODO
- Why: test inventory drifts quickly when mode/dtype coverage changes.
- Exit Condition: `docs/test.md` reflects active files and effective coverage (mode/dtype/shape) after each test refactor.

### P2-1: Retire temporary pad64-hotfix test after kernel fix
- Priority: P2
- Owner: TBD
- Status: TODO
- Why: `test_batch_man_pad64_hotfix.py` validates a temporary Python workaround.
- Exit Condition: after kernel tail/predicate fix lands, remove or downgrade hotfix-specific contract tests.

## Done Criteria

A backlog item is marked `DONE` only when:
- test behavior is covered by stable regression checks,
- related docs are updated in the same change,
- old redundant assertions are removed or clearly scoped as legacy.

## Notes

- Use `python3 -m pytest -q tests` for repo tests to avoid collecting vendored examples under `deps/`.
- Keep this file as a live execution backlog (not a changelog).
