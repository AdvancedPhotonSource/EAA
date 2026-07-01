# Installation

## PyPI

Install the core runtime:

```bash
pip install eaa-core
```

Install the core runtime plus the domain packages:

```bash
pip install eaa-core eaa-imaging eaa-spectroscopy
```

## Source development with `uv`

At the repository root:

```bash
uv sync --all-extras
source .venv/bin/activate
which python
```

After activation, `which python` should resolve to `.venv/bin/python`.

To sync a smaller source-development environment, select a workspace package.
Domain packages depend on `eaa-core`, so selecting one domain package also
installs the core runtime:

```bash
# Core runtime only
uv sync --package eaa-core

# Core runtime plus imaging
uv sync --package eaa-imaging

# Core runtime plus spectroscopy
uv sync --package eaa-spectroscopy

# Core runtime plus both domain packages
uv sync --all-packages
```

Optional extras are declared in `pyproject.toml`. The most relevant ones are:

- `docs` for the Material for MkDocs documentation toolchain
- `bo` for Bayesian optimization dependencies
- `memory_chroma` for the built-in Chroma-backed memory stack

For normal repository development, keep using `uv sync --all-extras`. If you
intentionally want a reduced environment for a narrow task, install a targeted
extra set with:

```bash
uv sync --extra docs
```

## Editable source install with `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e packages/eaa-core
pip install -e packages/eaa-imaging
pip install -e packages/eaa-spectroscopy
```

With docs dependencies:

```bash
pip install -r docs/requirements.txt
```

## Documentation build

```bash
uv run --extra docs mkdocs build --strict
```

The built HTML site is written to `site/`.

For local preview:

```bash
uv run --extra docs mkdocs serve
```

## GitHub Pages

This repository includes `mkdocs.yml` and
`.github/workflows/docs.yml`. Pushes to `main` build the MkDocs site and
publish it with GitHub Pages at:

```text
https://advancedphotonsource.github.io/EAA/
```

In the repository settings, configure GitHub Pages to use **GitHub Actions** as
the source.

## Status note on memory dependencies

The built-in `MemoryManager` path in the current repository creates a
Chroma-backed store. Install the `memory_chroma` extra when using that built-in
memory stack.
