# Documentation Audit Acceptance Criteria

Last updated (UTC): 2026-04-23T00:00:00Z

## Purpose
Define objective acceptance criteria for repository **规范性 / 完整性 / 一致性** audit, and keep conclusions reproducible.

## Scope
- In scope: `AGENTS.md`, `README.md`, `docs/*.md`, `docs/*.json`, referenced commands/paths, and cross-check with `tests/` + key runtime routing docs.
- Out of scope: implementing code fixes in non-doc files.

## Evaluation Dimensions

### A. 规范性 (Style & Convention)
- Heading structure clear, terminology consistent.
- Command style follows one convention (for example `python3 -m ...`).
- Severity levels use unified taxonomy: `P0/P1/P2`.

### B. 完整性 (Coverage)
- Must cover: project structure, run/build/test commands, FA2 mode matrix, known constraints/hotfix, test inventory.
- Test docs must reflect real file inventory and key coverage axes (mode/dtype/shape/GQA/paged/debug).

### C. 一致性 (Doc ↔ Repo Truth)
- Mode names, API semantics, and fallback/debug behavior in docs match implementation.
- Referenced paths exist.
- Example commands are executable in principle (static validation + pytest collect evidence).
- Status snapshots (date/commit) align with current HEAD.

### D. 可验证性 (Evidence Quality)
- Every finding includes evidence (file path + line or command output).
- Findings include impact and actionable recommendation.

## Severity Definition
- `P0`: misleading or wrong content causing unsafe operation or invalid conclusion.
- `P1`: important mismatch that can hide failures or block reliable reproduction.
- `P2`: non-blocking inconsistency/maintenance issue.

## Pass Criteria
- `P0 == 0`.
- Each `P1` has a concrete remediation recommendation.
- Audit report includes machine-readable mirror (`docs/audit_report.json`) with matching counts.
