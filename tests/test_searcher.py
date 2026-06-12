"""
test_searcher.py -- Tests for both search() (CLI) and search_memories() (API).

Uses the real ChromaDB fixtures from conftest.py for integration tests,
plus mock-based tests for error paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from mempalace.searcher import SearchError, _combine_where_filters, search, search_memories


# ── search_memories (API) ──────────────────────────────────────────────


class TestSearchMemories:
    def test_basic_search(self, palace_path, seeded_collection):
        result = search_memories("JWT authentication", palace_path)
        assert "results" in result
        assert len(result["results"]) > 0
        assert result["query"] == "JWT authentication"

    def test_wing_filter(self, palace_path, seeded_collection):
        result = search_memories("planning", palace_path, wing="notes")
        assert all(r["wing"] == "notes" for r in result["results"])

    def test_room_filter(self, palace_path, seeded_collection):
        result = search_memories("database", palace_path, room="backend")
        assert all(r["room"] == "backend" for r in result["results"])

    def test_wing_and_room_filter(self, palace_path, seeded_collection):
        result = search_memories("code", palace_path, wing="project", room="frontend")
        assert all(r["wing"] == "project" and r["room"] == "frontend" for r in result["results"])

    def test_n_results_limit(self, palace_path, seeded_collection):
        result = search_memories("code", palace_path, n_results=2)
        assert len(result["results"]) <= 2

    def test_no_palace_returns_error(self, tmp_path):
        result = search_memories("anything", str(tmp_path / "missing"))
        assert "error" in result

    def test_result_fields(self, palace_path, seeded_collection):
        result = search_memories("authentication", palace_path)
        hit = result["results"][0]
        assert "text" in hit
        assert "wing" in hit
        assert "room" in hit
        assert "source_file" in hit
        assert "similarity" in hit
        assert isinstance(hit["similarity"], float)
        assert "created_at" in hit

    def test_created_at_contains_filed_at(self, palace_path, seeded_collection):
        """created_at surfaces the filed_at metadata from the drawer."""
        result = search_memories("JWT authentication", palace_path)
        hit = result["results"][0]
        assert hit["created_at"] == "2026-01-01T00:00:00"

    def test_created_at_fallback_when_filed_at_missing(self):
        """created_at defaults to 'unknown' when filed_at is absent."""
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "ids": [["drawer_no_date"]],
            "documents": [["Some text without a date"]],
            "metadatas": [[{"wing": "project", "room": "backend", "source_file": "x.py"}]],
            "distances": [[0.1]],
        }

        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            result = search_memories("test", "/fake/path")
        hit = result["results"][0]
        assert hit["created_at"] == "unknown"

    def test_search_memories_query_error(self):
        """search_memories returns error dict when query raises."""
        mock_col = MagicMock()
        mock_col.query.side_effect = RuntimeError("query failed")

        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            result = search_memories("test", "/fake/path")
        assert "error" in result
        assert "query failed" in result["error"]

    def test_search_memories_vector_path_uses_explicit_collection_name(self):
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }

        with patch("mempalace.searcher.get_collection", return_value=mock_col) as get_collection:
            search_memories("test", "/fake/path", collection_name="custom_drawers")

        get_collection.assert_called_once_with(
            "/fake/path",
            collection_name="custom_drawers",
            create=False,
        )

    def test_query_embeddings_feed_drawer_and_closet_vector_queries(self):
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [["drawer doc"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}]],
            "distances": [[0.2]],
            "ids": [["d1"]],
        }
        closets_col = MagicMock()
        closets_col.query.return_value = {
            "documents": [["closet doc"]],
            "metadatas": [[{"source_file": "a.md"}]],
            "distances": [[0.1]],
            "ids": [["c1"]],
        }
        embedding = [[0.1, 0.2, 0.3]]

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch("mempalace.searcher.get_closets_collection", return_value=closets_col),
        ):
            search_memories("raw query text", "/fake/path", query_embeddings=embedding)

        drawer_kwargs = drawers_col.query.call_args.kwargs
        closet_kwargs = closets_col.query.call_args.kwargs
        assert drawer_kwargs["query_embeddings"] == embedding
        assert closet_kwargs["query_embeddings"] == embedding
        assert "query_texts" not in drawer_kwargs
        assert "query_texts" not in closet_kwargs

    def test_query_embeddings_still_use_raw_query_for_bm25_rerank(self):
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [["irrelevant filler", "needle needle exact match"]],
            "metadatas": [
                [
                    {"source_file": "a.md", "wing": "w", "room": "r"},
                    {"source_file": "b.md", "wing": "w", "room": "r"},
                ]
            ],
            "distances": [[0.5, 0.5]],
            "ids": [["d1", "d2"]],
        }

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch(
                "mempalace.searcher.get_closets_collection", side_effect=RuntimeError("no closets")
            ),
        ):
            result = search_memories("needle", "/fake/path", query_embeddings=[[0.1, 0.2]])

        assert result["results"][0]["source_file"] == "b.md"
        assert result["results"][0]["bm25_score"] > 0

    def test_drawer_hits_expose_chroma_drawer_id(self):
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [["first doc", "second doc"]],
            "metadatas": [
                [
                    {"source_file": "a.md", "wing": "w", "room": "r"},
                    {"source_file": "b.md", "wing": "w", "room": "r"},
                ]
            ],
            "distances": [[0.1, 0.2]],
            "ids": [["drawer-a", "drawer-b"]],
        }

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch(
                "mempalace.searcher.get_closets_collection", side_effect=RuntimeError("no closets")
            ),
        ):
            result = search_memories("query", "/fake/path")

        assert result["results"][0]["drawer_id"] == "drawer-a"
        assert result["results"][1]["drawer_id"] == "drawer-b"

    def test_query_embeddings_still_use_raw_query_for_scoped_hydration(self):
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [["initial drawer hit"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}]],
            "distances": [[0.3]],
            "ids": [["d1"]],
        }
        drawers_col.get.return_value = {
            "documents": ["plain context", "needle-rich hydrated context"],
            "metadatas": [{"chunk_index": 0}, {"chunk_index": 1}],
            "ids": ["d1", "d2"],
        }
        closets_col = MagicMock()
        closets_col.query.return_value = {
            "documents": [["closet pointer"]],
            "metadatas": [[{"source_file": "a.md", "drawer_ids_json": '["d1", "d2"]'}]],
            "distances": [[0.1]],
            "ids": [["c1"]],
        }

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch("mempalace.searcher.get_closets_collection", return_value=closets_col),
        ):
            result = search_memories("needle", "/fake/path", query_embeddings=[[0.1, 0.2]])

        hit = result["results"][0]
        assert "needle-rich hydrated context" in hit["text"]
        assert hit["drawer_index"] == 1
        assert hit["drawer_id"] == "d2"

    def test_scoped_hydration_keeps_generic_where_scope(self):
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [["work vector hit"]],
            "metadatas": [[{"source_file": "same.md", "wing": "w", "room": "r", "hall": "work"}]],
            "distances": [[0.3]],
            "ids": [["work-vector-id"]],
        }

        def scoped_get(**kwargs):
            if kwargs.get("ids") == ["work-id", "personal-id"] and kwargs.get("where") == {
                "hall": "work"
            }:
                return {
                    "documents": ["work scoped context"],
                    "metadatas": [{"chunk_index": 0, "hall": "work", "account": "alice"}],
                    "ids": ["work-id"],
                }
            return {
                "documents": ["work scoped context", "needle personal context"],
                "metadatas": [
                    {"chunk_index": 0, "hall": "work", "account": "alice"},
                    {"chunk_index": 1, "hall": "personal", "account": "bob"},
                ],
                "ids": ["work-id", "personal-id"],
            }

        drawers_col.get.side_effect = scoped_get
        closets_col = MagicMock()
        closets_col.query.return_value = {
            "documents": [["closet pointer"]],
            "metadatas": [
                [
                    {
                        "source_file": "same.md",
                        "hall": "work",
                        "drawer_ids_json": '["work-id", "personal-id"]',
                    }
                ]
            ],
            "distances": [[0.1]],
            "ids": [["c1"]],
        }

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch("mempalace.searcher.get_closets_collection", return_value=closets_col),
        ):
            result = search_memories("needle", "/fake/path", where={"hall": "work"})

        hit = result["results"][0]
        assert hit["drawer_id"] == "work-id"
        assert hit["text"] == "work scoped context"
        assert hit["metadata"]["hall"] == "work"
        assert hit["metadata"]["account"] == "alice"

    def test_source_hydration_combines_source_file_with_generic_where(self):
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [["work vector hit"]],
            "metadatas": [[{"source_file": "same.md", "wing": "w", "room": "r", "hall": "work"}]],
            "distances": [[0.3]],
            "ids": [["work-vector-id"]],
        }

        def source_get(**kwargs):
            if kwargs.get("where") == {"$and": [{"source_file": "same.md"}, {"hall": "work"}]}:
                return {
                    "documents": ["work source context"],
                    "metadatas": [{"chunk_index": 0, "hall": "work"}],
                    "ids": ["work-source-id"],
                }
            return {
                "documents": ["work source context", "needle personal context"],
                "metadatas": [
                    {"chunk_index": 0, "hall": "work"},
                    {"chunk_index": 1, "hall": "personal"},
                ],
                "ids": ["work-source-id", "personal-source-id"],
            }

        drawers_col.get.side_effect = source_get
        closets_col = MagicMock()
        closets_col.query.return_value = {
            "documents": [["closet without scoped ids"]],
            "metadatas": [[{"source_file": "same.md", "hall": "work"}]],
            "distances": [[0.1]],
            "ids": [["c1"]],
        }

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch("mempalace.searcher.get_closets_collection", return_value=closets_col),
        ):
            result = search_memories("needle", "/fake/path", where={"hall": "work"})

        hit = result["results"][0]
        assert hit["drawer_id"] == "work-source-id"
        assert hit["text"] == "work source context"
        assert hit["metadata"]["hall"] == "work"

    @pytest.mark.parametrize(
        ("base_where", "extra_where", "expected"),
        [
            (None, None, {}),
            ({"wing": "notes"}, None, {"wing": "notes"}),
            ({"room": "backend"}, None, {"room": "backend"}),
            (
                {"$and": [{"wing": "project"}, {"room": "frontend"}]},
                None,
                {"$and": [{"wing": "project"}, {"room": "frontend"}]},
            ),
            (None, {"source_file": "x.md"}, {"source_file": "x.md"}),
            (
                {"wing": "notes"},
                {"source_file": "x.md"},
                {"$and": [{"wing": "notes"}, {"source_file": "x.md"}]},
            ),
            (
                {"room": "backend"},
                {"source_file": "x.md"},
                {"$and": [{"room": "backend"}, {"source_file": "x.md"}]},
            ),
            (
                {"wing": "notes"},
                {"$and": [{"source_file": "x.md"}, {"room": "backend"}]},
                {"$and": [{"wing": "notes"}, {"source_file": "x.md"}, {"room": "backend"}]},
            ),
            (
                {"wing": "notes"},
                {"wing": "project"},
                {"$and": [{"wing": "notes"}, {"wing": "project"}]},
            ),
            (
                {"room": "backend"},
                {"room": "frontend"},
                {"$and": [{"room": "backend"}, {"room": "frontend"}]},
            ),
        ],
    )
    def test_combine_where_filters(self, base_where, extra_where, expected):
        assert _combine_where_filters(base_where, extra_where) == expected

    def test_search_memories_combines_where_with_wing_and_room_filters(self):
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch(
                "mempalace.searcher.get_closets_collection", side_effect=RuntimeError("no closets")
            ),
        ):
            search_memories(
                "test",
                "/fake/path",
                wing="project",
                room="backend",
                where={"source_file": "x.md"},
            )

        assert drawers_col.query.call_args.kwargs["where"] == {
            "$and": [{"wing": "project"}, {"room": "backend"}, {"source_file": "x.md"}]
        }

    def test_vector_disabled_with_generic_where_returns_clear_error(self):
        with patch("mempalace.searcher._bm25_only_via_sqlite") as bm25_only:
            result = search_memories(
                "test",
                "/fake/path",
                vector_disabled=True,
                where={"source_file": "scoped.md"},
            )

        assert "error" in result
        assert "generic where" in result["error"]
        assert "BM25-only fallback" in result["error"]
        bm25_only.assert_not_called()

    def test_vector_disabled_with_expand_to_burst_logs_warning(self):
        """vector_disabled=True + expand_to_burst=True: BM25 path proceeds but logs a warning."""
        with patch("mempalace.searcher._bm25_only_via_sqlite") as bm25_only, \
             patch("mempalace.searcher.logger") as mock_logger:
            bm25_only.return_value = {"results": []}
            result = search_memories(
                "test",
                "/fake/path",
                vector_disabled=True,
                expand_to_burst=True,
            )

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args.args[0]
        assert "expand_to_burst" in warning_msg
        assert "BM25" in warning_msg or "vector_disabled" in warning_msg
        bm25_only.assert_called_once()
        assert "error" not in result

    def test_union_strategy_skips_bm25_candidates_when_generic_where_is_present(self):
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [["vector scoped doc"]],
            "metadatas": [[{"source_file": "scoped.md", "wing": "w", "room": "r"}]],
            "distances": [[0.4]],
            "ids": [["d1"]],
        }

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch(
                "mempalace.searcher.get_closets_collection", side_effect=RuntimeError("no closets")
            ),
            patch("mempalace.searcher._bm25_only_via_sqlite") as bm25_only,
        ):
            result = search_memories(
                "query",
                "/fake/path",
                candidate_strategy="union",
                where={"source_file": "scoped.md"},
                n_results=3,
            )

        bm25_only.assert_not_called()
        assert [hit["source_file"] for hit in result["results"]] == ["scoped.md"]

    def test_metadata_boost_changes_ordering_without_changing_raw_distance(self):
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [["alpha", "beta"]],
            "metadatas": [
                [
                    {"source_file": "low.md", "wing": "w", "room": "r", "priority": "low"},
                    {"source_file": "high.md", "wing": "w", "room": "r", "priority": "high"},
                ]
            ],
            "distances": [[0.4, 0.6]],
            "ids": [["d1", "d2"]],
        }

        def boost(meta):
            return 99.0 if meta.get("priority") == "high" else -5.0

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch(
                "mempalace.searcher.get_closets_collection", side_effect=RuntimeError("no closets")
            ),
        ):
            result = search_memories("query", "/fake/path", metadata_boost=boost)

        hits = result["results"]
        assert [hit["source_file"] for hit in hits] == ["high.md", "low.md"]
        assert hits[0]["distance"] == 0.6
        assert hits[0]["effective_distance"] == 0.0
        assert hits[0]["metadata_boost"] == 0.6
        assert hits[1]["distance"] == 0.4
        assert hits[1]["effective_distance"] == 0.4
        assert hits[1]["metadata_boost"] == 0.0

    def test_search_memories_filters_in_result(self, palace_path, seeded_collection):
        result = search_memories("test", palace_path, wing="project", room="backend")
        assert result["filters"]["wing"] == "project"
        assert result["filters"]["room"] == "backend"

    def test_search_memories_handles_none_metadata(self):
        """API path: `None` entries in the drawer results' metadatas list must
        fall back to the sentinel strings (wing/room 'unknown', source '?')
        rather than raising `AttributeError: 'NoneType' object has no
        attribute 'get'` while the rest of the result set renders."""
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["first doc", "second doc"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}, None]],
            "distances": [[0.1, 0.2]],
            "ids": [["d1", "d2"]],
        }

        def mock_get_collection(path, collection_name=None, create=False):
            # First call: drawers. Second call: closets — raise so hybrid
            # degrades to pure drawer search (the catch block covers it).
            if not hasattr(mock_get_collection, "_called"):
                mock_get_collection._called = True
                return mock_col
            raise RuntimeError("no closets")

        with patch("mempalace.searcher.get_collection", side_effect=mock_get_collection):
            result = search_memories("anything", "/fake/path")
        assert "results" in result
        assert len(result["results"]) == 2
        # The None-metadata hit renders with sentinel values, not a crash.
        none_hit = result["results"][1]
        assert none_hit["text"] == "second doc"
        assert none_hit["wing"] == "unknown"
        assert none_hit["room"] == "unknown"

    def test_effective_distance_clamped_to_valid_cosine_range(self):
        """A strong closet boost (up to 0.40) applied to a low-distance drawer
        can drive ``dist - boost`` negative. That violates the cosine-distance
        invariant ``[0, 2]``: the API returns ``similarity > 1.0`` and the
        internal ``_sort_key`` sinks below ordinary positive distances,
        inverting the ranking so the best hybrid matches sort last.

        With the clamp, ``effective_distance`` stays in ``[0, 2]``,
        ``similarity`` stays in ``[0, 1]``, and the sort order is stable.
        """
        # Drawer a.md gets a tiny base distance (0.08) — nearly exact match.
        # Drawer b.md gets a larger base distance (0.35).
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [["doc-a", "doc-b"]],
            "metadatas": [
                [
                    {"source_file": "a.md", "wing": "w", "room": "r", "chunk_index": 0},
                    {"source_file": "b.md", "wing": "w", "room": "r", "chunk_index": 0},
                ]
            ],
            "distances": [[0.08, 0.35]],
            "ids": [["d-a", "d-b"]],
        }
        # A strong closet at rank 0 points at a.md → boost = 0.40,
        # which exceeds a.md's base distance and would go negative without
        # the clamp. No closet for b.md.
        closets_col = MagicMock()
        closets_col.query.return_value = {
            "documents": [["closet-preview-a"]],
            "metadatas": [[{"source_file": "a.md"}]],
            "distances": [[0.2]],  # within CLOSET_DISTANCE_CAP (1.5)
            "ids": [["c-a"]],
        }

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch("mempalace.searcher.get_closets_collection", return_value=closets_col),
        ):
            result = search_memories("query", "/fake/path", n_results=5)

        hits = result["results"]
        assert hits, "should return results"

        # Invariants on every hit.
        for h in hits:
            assert 0.0 <= h["similarity"] <= 1.0, (
                f"similarity out of range: {h['similarity']} for {h['source_file']}"
            )
            assert 0.0 <= h["effective_distance"] <= 2.0, (
                f"effective_distance out of range: {h['effective_distance']} for {h['source_file']}"
            )

        # With the clamp, the closet-boosted a.md still ranks ahead of b.md —
        # the boost still wins, but it no longer flips the ranking.
        assert hits[0]["source_file"] == "a.md"
        assert hits[0]["matched_via"] == "drawer+closet"


# ── BM25 internals: None / empty document safety ─────────────────────


class TestBM25NoneSafety:
    """Regression tests for the AttributeError observed in production when
    Chroma returned ``None`` documents inside a hybrid-rerank pass.

    Trace from the daemon log (2026-04-24 21:07:05):
        File "mempalace/searcher.py", line 81, in _bm25_scores
            tokenized = [_tokenize(d) for d in documents]
        File "mempalace/searcher.py", line 52, in _tokenize
            return _TOKEN_RE.findall(text.lower())
        AttributeError: 'NoneType' object has no attribute 'lower'
    """

    def test_tokenize_handles_none(self):
        from mempalace.searcher import _tokenize

        assert _tokenize(None) == []

    def test_tokenize_handles_empty_string(self):
        from mempalace.searcher import _tokenize

        assert _tokenize("") == []

    def test_bm25_scores_does_not_crash_on_none_documents(self):
        """A ``None`` mixed into the corpus must yield score 0.0 for that doc
        and finite scores for the rest, not raise AttributeError."""
        from mempalace.searcher import _bm25_scores

        scores = _bm25_scores(
            "postgres migration", ["postgres migration done", None, "kafka rebalance"]
        )
        assert len(scores) == 3
        assert scores[1] == 0.0
        assert scores[0] > 0.0


# ── search() (CLI print function) ─────────────────────────────────────


@pytest.fixture
def fake_palace_path(tmp_path):
    """tmp_path with chroma.sqlite3 touched so searcher.search's
    filesystem-first state checks (#1498) pass through to the mocked
    backend instead of raising on State A / State B."""
    p = tmp_path / "palace"
    p.mkdir()
    (p / "chroma.sqlite3").touch()
    return str(p)


class TestSearchCLI:
    def test_search_prints_results(self, palace_path, seeded_collection, capsys):
        search("JWT authentication", palace_path)
        captured = capsys.readouterr()
        assert "JWT" in captured.out or "authentication" in captured.out

    def test_search_with_wing_filter(self, palace_path, seeded_collection, capsys):
        search("planning", palace_path, wing="notes")
        captured = capsys.readouterr()
        assert "Results for" in captured.out

    def test_search_with_room_filter(self, palace_path, seeded_collection, capsys):
        search("database", palace_path, room="backend")
        captured = capsys.readouterr()
        assert "Room:" in captured.out

    def test_search_with_wing_and_room(self, palace_path, seeded_collection, capsys):
        search("code", palace_path, wing="project", room="frontend")
        captured = capsys.readouterr()
        assert "Wing:" in captured.out
        assert "Room:" in captured.out

    def test_search_no_palace_raises(self, tmp_path):
        with pytest.raises(SearchError, match="No palace found"):
            search("anything", str(tmp_path / "missing"))

    def test_search_no_results(self, palace_path, collection, capsys):
        """Empty collection returns no results message."""
        # collection is empty (no seeded data)
        result = search("xyzzy_nonexistent_query", palace_path, n_results=1)
        captured = capsys.readouterr()
        # Either prints "No results" or returns None
        assert result is None or "No results" in captured.out

    def test_search_query_error_raises(self, fake_palace_path):
        """search raises SearchError when query fails."""
        mock_col = MagicMock()
        mock_col.query.side_effect = RuntimeError("boom")

        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            with pytest.raises(SearchError, match="Search error"):
                search("test", fake_palace_path)

    def test_search_n_results(self, palace_path, seeded_collection, capsys):
        search("code", palace_path, n_results=1)
        captured = capsys.readouterr()
        # Should have output with at least one result block
        assert "[1]" in captured.out

    def test_search_applies_bm25_hybrid_rerank(self, fake_palace_path, capsys):
        """CLI search must call the same hybrid rerank that the MCP path uses.

        Regression for a bug where the CLI only consulted ChromaDB cosine
        distance: a drawer whose body contained every query term still
        scored zero similarity if its embedding happened to be far from
        the query (e.g. the drawer was a shell-output fragment that
        embeds as "file tree noise"). Hybrid rerank fixes this by
        combining BM25 with cosine — lexical matches rise above pure
        vector noise.

        Simulates: three candidates, all with distance >= 1.0 (cosine = 0);
        candidate 2 contains every query term. After the fix, candidate 2
        should rank first and display a non-zero bm25 score.
        """
        mock_col = MagicMock()
        mock_col.metadata = {"hnsw:space": "cosine"}
        mock_col.query.return_value = {
            "documents": [
                [
                    "unrelated directory listing -rw-rw-r-- file.txt",
                    "foo bar baz is a multi-word phrase",
                    "another unrelated chunk about colors",
                ]
            ],
            "metadatas": [
                [
                    {"source_file": "a.md", "wing": "w", "room": "r"},
                    {"source_file": "b.md", "wing": "w", "room": "r"},
                    {"source_file": "c.md", "wing": "w", "room": "r"},
                ]
            ],
            "distances": [[1.5, 1.5, 1.5]],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search("foo bar baz", fake_palace_path)
        captured = capsys.readouterr()
        first_block, _, _ = captured.out.partition("[2]")
        # Lexical match must rank first
        assert "b.md" in first_block, (
            f"expected lexical match 'b.md' at rank 1, got:\n{captured.out}"
        )
        # Non-zero bm25 reported
        assert "bm25=" in first_block
        assert "bm25=0.0" not in first_block
        # Cosine still reported for transparency
        assert "cosine=" in first_block

    def test_search_warns_when_palace_uses_wrong_distance_metric(self, fake_palace_path, capsys):
        """Legacy palaces created without `hnsw:space=cosine` silently
        use L2, which breaks similarity interpretation. CLI must warn
        the user and point them at `mempalace repair` rather than
        pretending the `Match` scores are meaningful."""
        mock_col = MagicMock()
        mock_col.metadata = {}  # legacy: no hnsw:space set
        mock_col.query.return_value = {
            "documents": [["some drawer content"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}]],
            "distances": [[1.2]],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search("anything", fake_palace_path)
        captured = capsys.readouterr()
        assert "mempalace repair" in captured.err
        assert "cosine" in captured.err.lower()

    def test_search_does_not_warn_when_palace_is_correctly_configured(
        self, fake_palace_path, capsys
    ):
        mock_col = MagicMock()
        mock_col.metadata = {"hnsw:space": "cosine"}
        mock_col.query.return_value = {
            "documents": [["some drawer content"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}]],
            "distances": [[0.3]],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search("anything", fake_palace_path)
        captured = capsys.readouterr()
        assert "mempalace repair" not in captured.err

    def test_search_handles_none_metadata_without_crash(self, fake_palace_path, capsys):
        """ChromaDB can return `None` entries in the metadatas list when a
        drawer has no metadata. The CLI print path must not crash on them
        mid-render — it used to raise `AttributeError: 'NoneType' object has
        no attribute 'get'` after printing earlier results."""
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["first doc", "second doc"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}, None]],
            "distances": [[0.1, 0.2]],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search("anything", fake_palace_path)
        captured = capsys.readouterr()
        assert "[1]" in captured.out
        assert "[2]" in captured.out
        # Second result renders with fallback '?' values instead of crashing
        assert "second doc" in captured.out

    def test_search_handles_none_document_without_crash(self, fake_palace_path, capsys):
        mock_col = MagicMock()
        mock_col.metadata = {"hnsw:space": "cosine"}
        mock_col.query.return_value = {
            "documents": [["first doc", None]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}, None]],
            "distances": [[0.1, 0.2]],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search("anything", fake_palace_path)
        captured = capsys.readouterr()
        assert "[1]" in captured.out
        assert "[2]" in captured.out
