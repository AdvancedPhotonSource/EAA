# Installation

## Recommended: `uv`

At the repository root:

```bash
uv sync --all-extras
source .venv/bin/activate
which python
```

After activation, `which python` should resolve to `.venv/bin/python`.

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

## Editable install with `pip`

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
