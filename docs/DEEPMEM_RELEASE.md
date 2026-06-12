# DeepMem Release

This branch is the DeepMem fork of the MemPalace SDK. DeepMem releases are cut
from `deepmem-main`, built locally, and distributed without GitHub Actions.

## Version Format

Use one version string everywhere:

```text
<upstream-mempalace-version>+deepmem.<deepmem-sdk-version>
```

Example:

```text
3.4.0+deepmem.0.1.0
```

Rules:

- Keep the package name as `mempalace`; DeepMem depends on this SDK API.
- Use the same string in `pyproject.toml`, `mempalace/version.py`, plugin
  manifests, and the Git tag.
- Use PEP 440 local-version syntax with `+deepmem...`; do not use
  `deepmem-3.4.0-0.1.0` as the Python package version.
- Do not publish these fork versions to upstream PyPI. They are for local
  builds or direct Git installs.

## GitHub Actions

GitHub Actions are disabled on `deepmem-main`. Keep workflow files renamed to
`*.yml.disabled` so the original upstream workflow definitions stay available
for reference but do not run in the DeepMem fork.

## Local Release Checklist

Run from the repo root on `deepmem-main`.

```bash
git status --short --branch
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v --ignore=tests/benchmarks
uv build
```

Create the tag after verification:

```bash
git tag 3.4.0+deepmem.0.1.0
git push origin deepmem-main
git push origin 3.4.0+deepmem.0.1.0
```

## Installing From GitHub

External projects can depend on this fork directly from GitHub instead of using
a local wheel. Pin a tag or commit for reproducible installs.

For `pip`:

```bash
python -m pip install "mempalace @ git+https://github.com/wangbooth/mempalace.git@3.4.0+deepmem.0.1.0"
```

For `uv add`:

```bash
uv add "mempalace @ git+https://github.com/wangbooth/mempalace.git@3.4.0+deepmem.0.1.0"
```

For `pyproject.toml` dependencies:

```toml
dependencies = [
    "mempalace @ git+https://github.com/wangbooth/mempalace.git@3.4.0+deepmem.0.1.0",
]
```

If the repository is private, use SSH:

```bash
uv add "mempalace @ git+ssh://git@github.com/wangbooth/mempalace.git@3.4.0+deepmem.0.1.0"
```

Depending on `deepmem-main` is allowed for local development, but release
consumers should pin the tag or a commit SHA.
