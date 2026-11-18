---
trigger: model_decision
description: Guide for LaTeX Compilation
---

## Basic Compilation & Error Handling
- Whenever you create or modify a `.tex` file, immediately compile it to PDF using the terminal tool:
  `tectonic -X compile <filename>.tex`
- If compilation fails, read the terminal error output, fix the LaTeX syntax, and re-run `tectonic` until a `.pdf` is cleanly generated.
- To prevent silent warnings or broken references from slipping through, treat warnings as errors during final checks by running:
  `tectonic -X compile --untrusted -Z continue-on-errors=false <filename>.tex`

## Output Management & Workspace Hygiene
- If the user asks to keep the workspace clean or place PDFs in a specific folder, use the `--outdir` flag:
  `tectonic -X compile --outdir build <filename>.tex`
- Tectonic automatically cleans up intermediate files (`.aux`, `.log`, `.bbl`) by default. Only pass `--keep-intermediates` if you specifically need to inspect raw `.log` or `.aux` files to debug a complex package issue.
- Only pass `--keep-logs` when deep debugging is required (`tectonic -X compile --keep-logs <filename>.tex`), and delete the generated `.log` file afterward.

## Multi-File Projects & Modern Workflows
- If the workspace contains a `Tectonic.toml` file, build the entire project instead of individual `.tex` files by running:
  `tectonic -X build`
- If starting a large multi-chapter document, initialize a structured project with `tectonic -X new <project_name>`.

## Offline & Shell-Escape Rules
- Once packages are cached, you can force strictly offline builds (zero network checks) using `-C`:
  `tectonic -X compile -C <filename>.tex`
- If the document uses packages requiring external shell execution (such as `minted` for code syntax highlighting or `svg`), pass `--untrusted` (which enables shell-escape in Tectonic):
  `tectonic -X compile --untrusted <filename>.tex`