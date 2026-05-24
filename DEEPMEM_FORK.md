# DeepMem Custom Fork

The `deepmem-main` branch contains DeepMem-specific modifications on top of the
upstream mempalace project. When upstream releases new versions, fast-forward
the local `develop` branch to upstream and merge `develop` into `deepmem-main`
rather than rebasing, so the custom patches remain clearly separated in git
history.

## Upstream

- Repository: https://github.com/MemPalace/mempalace
- Upstream branch: `develop`
- Local mirror branch: `develop`
- DeepMem custom branch: `deepmem-main`

Recommended sync:

```bash
git switch develop
git pull --ff-only upstream develop
git switch deepmem-main
git merge develop
```

## Custom Changes

### `mempalace/embedding.py` — External embedding function override

Added `set_override_embedding_function(ef)` and a 3-line early-return in
`get_embedding_function()`.  This allows the DeepMem daemon to inject
`text2vec-large-chinese` (1024-dim) at process startup instead of using the
built-in ONNX MiniLM (384-dim).

**Why:** DeepMem targets Chinese-language memory (WeChat, etc.).  The default
MiniLM is English-centric; `text2vec-large-chinese` gives significantly better
Chinese semantic retrieval.

**Merge note:** If upstream refactors `embedding.py`, ensure
`_OVERRIDE_EF` / `set_override_embedding_function` survive the merge.  The
change is additive — no existing behaviour is altered when no override is set.

## Build & Install

DeepMem daemon depends on this branch via editable install or path dependency.
Always use `deepmem-main` for DeepMem builds, never the upstream `develop`
directly.
