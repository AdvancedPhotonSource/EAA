# Packages

This repository is a `uv` workspace with package-local source trees under
`packages/`.

- `packages/eaa-core`: shared runtime, generic tools, memory, WebUI, MCP
  helpers, and reusable task-manager infrastructure.
- `packages/eaa-imaging`: imaging and microscopy task managers, imaging tools,
  prompts, and skills.
- `packages/eaa-spectroscopy`: spectroscopy tools, acquisition functions, and
  spectroscopy task managers.

Source lives under package-local Python namespaces:

- `packages/eaa-core/src/eaa_core`
- `packages/eaa-imaging/src/eaa_imaging`
- `packages/eaa-spectroscopy/src/eaa_spectroscopy`

The root `pyproject.toml` aggregates all workspace members for local
development.
