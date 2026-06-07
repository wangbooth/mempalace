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

### `mempalace/searcher.py` — Generic search extension points

Added keyword-only optional parameters to `search_memories()`:

- `query_embeddings`: lets callers provide a precomputed query embedding for
  Chroma drawer and closet vector queries. The raw `query` string is still used
  for BM25 reranking, hydration keyword selection, diagnostics, and BM25
  fallback. When omitted, the default upstream-compatible `query_texts=[query]`
  behavior is unchanged.
- `where`: lets callers provide a generic Chroma metadata filter. It is combined
  with legacy `wing` and `room` filters via `$and`; collisions are preserved as
  separate predicates so Chroma validation and caller constraints are not
  weakened. Because the sqlite BM25 fallback only supports legacy `wing` and
  `room` filters, `search_memories(vector_disabled=True, where=...)` returns an
  explicit error and `candidate_strategy="union"` skips BM25-only expansion when
  generic `where` is present.
- `metadata_boost`: lets callers apply an optional metadata-based distance
  reduction after drawer and closet signals are assembled. The hook is clamped,
  ranking-only, and cannot remove results. Public `distance` remains the raw
  vector distance; boosted ranking distance is exposed as `effective_distance`.

**Why:** DeepMem needs to pass Jina v5 query-side embeddings into the generic
mempalace hybrid searcher while keeping product-specific concepts outside the
core engine.

**Merge note:** The extension is additive and keyword-only. Preserve the
existing `search_memories(query, palace_path, ...)` positional contract and the
default `query_texts` vector query path when merging upstream changes.

## Build & Install

DeepMem daemon depends on this branch via editable install or path dependency.
Always use `deepmem-main` for DeepMem builds, never the upstream `develop`
directly.
