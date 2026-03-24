# Package Split

This repository now has an incremental multi-package layout under `packages/`.

- `packages/eaa-core`: shared runtime, generic tools, generic BO task manager.
- `packages/eaa-imaging`: imaging and microscopy task managers, imaging tools,
  and APS-specific adapters.

Source now lives only under package-local trees:

- `packages/eaa-core/src/eaa`
- `packages/eaa-imaging/src/eaa`

The root `pyproject.toml` aggregates both package-local source roots for the
full-framework install, while each sub-package also has its own installable
manifest.
