# mempalace-searcher-extension Summary

## Files Changed

- `mempalace/searcher.py`
- `tests/test_searcher.py`
- `DEEPMEM_FORK.md`
- `agent-runs/mempalace-searcher-extension/summary.md`

## Commits Created

- `feat(search): support external query embeddings`
- `fix(search): guard generic where in bm25 fallback`
- `fix(search): expose drawer ids in hybrid results`
- `fix(search): enforce where during hydration`

## Tests Run

```bash
python3.11 -m pytest tests/test_searcher.py tests/test_hybrid_search.py tests/test_closets.py -q
```

Result: `131 passed in 37.77s`

```bash
python3.11 -m ruff check mempalace/searcher.py tests/test_searcher.py tests/test_hybrid_search.py tests/test_closets.py
```

Result: `All checks passed!`

## Behavior Intentionally Left Unchanged

- `search_memories(query, palace_path, ...)` keeps its existing positional
  contract and default `query_texts=[query]` Chroma behavior when
  `query_embeddings` is absent.
- Raw `query` remains the source for BM25 reranking, scoped hydration keyword
  selection, diagnostics, and BM25 fallback.
- `vector_disabled=True` still routes through the sqlite BM25-only fallback when
  only legacy `wing`/`room` filters are used.
- Closet boost and scoped drawer hydration remain part of the hybrid retrieval
  path.

## Implementation Notes

- Added keyword-only `query_embeddings`, `where`, and `metadata_boost` extension
  points to `search_memories()`.
- `query_embeddings` is used only for drawer and closet Chroma vector queries.
- `where` is combined with `wing` and `room` filters through `$and` without
  special-casing DeepMem-specific metadata names.
- Generic `where` is not silently ignored by sqlite BM25 paths:
  `vector_disabled=True` returns an explicit error, and
  `candidate_strategy="union"` skips BM25-only candidate expansion when generic
  `where` is present.
- Drawer-derived vector hits expose the Chroma drawer id as `drawer_id`. Scoped
  drawer+closet hydration updates `drawer_id` when keyword hydration selects a
  different scoped drawer/chunk.
- Drawer+closet hydration now preserves the combined hard Chroma `where` scope
  for scoped-id hydration and source-file neighbor hydration. Hydrated hits also
  carry generic raw drawer metadata through the `metadata` field.
- `metadata_boost` is clamped, ranking-only, and represented through
  `effective_distance`; public `distance` remains raw vector distance.

## Risks Or Follow-Up

- Daemon integration still needs to validate query vector cardinality,
  dimension, and drawer/closet embedding identity before calling mempalace.
- Generic sqlite BM25 fallback still only understands legacy `wing`/`room`
  filters; it now fails or skips clearly instead of returning out-of-scope
  candidates when a generic `where` filter is present.
