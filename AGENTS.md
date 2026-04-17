# Repository Guidelines

## Project Structure & Module Organization
Core Python source lives in `nanovllm/`:
- `engine/` handles scheduling, sequence lifecycle, and model execution.
- `layers/` contains model building blocks (attention, norms, linear layers, sampling).
- `models/` contains architecture-specific model definitions (for example `qwen3.py`).
- `utils/` contains loading/context helpers.

Top-level scripts:
- `example.py` for quick inference usage.
- `bench.py` for throughput benchmarking.
- `tests/` currently contains GPU correctness checks.

Native CUDA experiments are in `gemm_project/csrc/`. Vendored dependencies live in `deps/` and `third_party/`; avoid editing them unless you are intentionally updating upstream code.

## Build, Test, and Development Commands
- `python -m pip install -e .` installs the project in editable mode.
- `python example.py` runs a local generation smoke test (requires a local model path update).
- `python bench.py` runs benchmark code.
- `python tests/test_flash_attn_correctness.py` runs the current flash-attention correctness check.

Use a CUDA-enabled environment with compatible `torch`, `triton`, and `flash-attn` versions (see `pyproject.toml` and `.devcontainer/Dockerfile`).

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8 style, type hints where practical.
- Naming: `snake_case` for functions/variables/files, `PascalCase` for classes, `UPPER_CASE` for constants.
- Keep modules focused and readable; this repo favors small, explicit implementations over heavy abstraction.
- CUDA/C++ (`gemm_project/csrc`): follow existing brace/indent style and lowercase underscore filenames.

## Testing Guidelines
- Add tests under `tests/` with `test_*.py` naming.
- Prefer deterministic setups (`torch.manual_seed(...)`) for numeric comparisons.
- For kernel/attention changes, include both correctness checks and a reproducible performance command.

## Commit & Pull Request Guidelines
- Recent history uses short, imperative commit subjects (for example: `import nanovllm`, `update docker`).
- Keep commit messages concise and scoped to one logical change.
- PRs should include:
  - What changed and why.
  - How to run/reproduce (`python ...` commands).
  - Hardware/software context for performance-related changes.
  - Linked issue(s) when applicable.

## Security & Configuration Tips
- Do not commit model weights, secrets, or local absolute paths.
- Keep large artifacts in ignored directories and document required env vars/paths in the PR description.
