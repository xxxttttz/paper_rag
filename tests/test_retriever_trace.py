"""Retrieval trace tests with all external operations mocked."""

import unittest
from unittest.mock import patch

import retriever


class RetrievalTraceTests(unittest.TestCase):
    def test_chunk_id_survives_vector_bm25_merge(self):
        hit = {"id": "chunk-1", "source": "A.pdf", "page": 2, "text": "method", "score": 0.9}
        bm25_hit = {**hit, "score": 1.2}
        with (
            patch.object(retriever, "embed_query", return_value=[0.0]),
            patch.object(retriever, "_vector_search", return_value=[hit]),
            patch.object(retriever, "_load_all_chunks", return_value=[]),
            patch.object(retriever, "_bm25_search", return_value=[bm25_hit]),
            patch.object(retriever.config, "USE_HYBRID_SEARCH", True),
            patch.object(retriever.config, "ENABLE_RERANK", False),
        ):
            result = retriever._retrieve_with_client(object(), "question", top_k=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "chunk-1")

    def test_vector_search_returns_primary_key(self):
        class FakeClient:
            def load_collection(self, name):
                pass

            def search(self, **kwargs):
                return [[{
                    "id": "chunk-2",
                    "entity": {"text": "text", "source": "B.pdf", "page": 3},
                    "distance": 0.8,
                }]]

        hits = retriever._vector_search(FakeClient(), [0.0], top_k=1)
        self.assertEqual(hits[0]["id"], "chunk-2")

    def test_trace_has_stages_but_no_paper_text(self):
        vector_hits = [
            {"source": "A.pdf", "page": 1, "text": "private text A", "score": 0.9},
            {"source": "B.pdf", "page": 2, "text": "private text B", "score": 0.5},
        ]
        bm25_hits = [
            {"source": "C.pdf", "page": 3, "text": "private text C", "score": 1.2},
        ]
        trace = {}
        with (
            patch.object(retriever, "embed_query", return_value=[0.0]),
            patch.object(retriever, "_vector_search", return_value=vector_hits),
            patch.object(retriever, "_load_all_chunks", return_value=[]),
            patch.object(retriever, "_bm25_search", return_value=bm25_hits),
            patch.object(
                retriever, "rerank", side_effect=lambda query, candidates, top_n: candidates[:top_n]
            ),
            patch.object(retriever.config, "USE_HYBRID_SEARCH", True),
            patch.object(retriever.config, "ENABLE_RERANK", True),
            patch.object(retriever.config, "ENABLE_DIVERSE_RERANK", False),
            patch.object(retriever.config, "RERANK_CANDIDATE_K", 3),
        ):
            result = retriever._retrieve_with_client(object(), "question", top_k=2, trace=trace)

        self.assertEqual(list(trace), ["vector", "bm25", "candidates", "final"])
        self.assertEqual(len(result), 2)
        self.assertTrue(all("text" not in hit for hits in trace.values() for hit in hits))
        self.assertTrue(all(set(hit) == {"source", "page", "score"} for hits in trace.values() for hit in hits))

    def test_diverse_mode_requests_full_rerank_and_keeps_distinct_pages(self):
        vector_hits = [
            {"source": "A.pdf", "page": 1, "text": "same method", "score": 0.9},
            {"source": "A.pdf", "page": 1, "text": "same method", "score": 0.8},
            {"source": "B.pdf", "page": 2, "text": "different result", "score": 0.7},
        ]
        trace = {}
        with (
            patch.object(retriever, "embed_query", return_value=[0.0]),
            patch.object(retriever, "_vector_search", return_value=vector_hits),
            patch.object(retriever, "rerank", side_effect=lambda query, candidates, top_n: candidates[:top_n]) as rerank,
            patch.object(retriever.config, "USE_HYBRID_SEARCH", False),
            patch.object(retriever.config, "ENABLE_RERANK", True),
            patch.object(retriever.config, "ENABLE_DIVERSE_RERANK", True),
            patch.object(retriever.config, "RERANK_CANDIDATE_K", 3),
        ):
            result = retriever._retrieve_with_client(object(), "question", top_k=2, trace=trace)

        self.assertEqual(rerank.call_args.kwargs["top_n"], 3)
        self.assertEqual([(hit["source"], hit["page"]) for hit in result], [("A.pdf", 1), ("B.pdf", 2)])
        self.assertEqual(len(trace["reranked"]), 3)
        self.assertEqual(len(trace["final"]), 2)

    def test_requirement_mode_reuses_one_full_rerank_for_source_coverage(self):
        vector_hits = [
            {"source": "A.pdf", "page": 1, "text": "first", "score": 0.9},
            {"source": "A.pdf", "page": 2, "text": "second", "score": 0.8},
            {"source": "B.pdf", "page": 3, "text": "third", "score": 0.7},
        ]
        trace = {}
        with (
            patch.object(retriever, "embed_query", return_value=[0.0]),
            patch.object(retriever, "_vector_search", return_value=vector_hits),
            patch.object(
                retriever, "rerank",
                side_effect=lambda query, candidates, top_n: candidates[:top_n],
            ) as rerank,
            patch.object(retriever.config, "USE_HYBRID_SEARCH", False),
            patch.object(retriever.config, "ENABLE_RERANK", True),
            patch.object(retriever.config, "ENABLE_DIVERSE_RERANK", False),
            patch.object(retriever.config, "ENABLE_REQUIREMENT_SELECTION", True),
            patch.object(retriever.config, "RERANK_CANDIDATE_K", 3),
        ):
            result = retriever._retrieve_with_client(
                object(), "两篇论文各自如何识别设备？", top_k=2, trace=trace
            )

        self.assertEqual(rerank.call_count, 1)
        self.assertEqual(rerank.call_args.kwargs["top_n"], 3)
        self.assertEqual({hit["source"] for hit in result}, {"A.pdf", "B.pdf"})
        self.assertEqual(len(trace["reranked"]), 3)

    def test_capture_requests_full_rerank_without_changing_default_top_k(self):
        vector_hits = [
            {"source": "A.pdf", "page": page, "text": f"private {page}", "score": 1 / page}
            for page in (1, 2, 3)
        ]
        captured = []
        trace = {}
        with (
            patch.object(retriever, "embed_query", return_value=[0.0]),
            patch.object(retriever, "_vector_search", return_value=vector_hits),
            patch.object(
                retriever, "rerank",
                side_effect=lambda query, candidates, top_n: candidates[:top_n],
            ) as rerank,
            patch.object(retriever.config, "USE_HYBRID_SEARCH", False),
            patch.object(retriever.config, "ENABLE_RERANK", True),
            patch.object(retriever.config, "ENABLE_DIVERSE_RERANK", False),
            patch.object(retriever.config, "ENABLE_REQUIREMENT_SELECTION", False),
            patch.object(retriever.config, "RERANK_CANDIDATE_K", 3),
        ):
            result = retriever._retrieve_with_client(
                object(), "question", top_k=2, trace=trace, reranked_output=captured
            )

        rerank.assert_called_once()
        self.assertEqual(rerank.call_args.kwargs["top_n"], 3)
        self.assertEqual(len(captured), 3)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(trace["reranked"]), 3)
        self.assertTrue(all("text" not in hit for hits in trace.values() for hit in hits))

    def test_capture_does_not_silently_fall_back_when_rerank_fails(self):
        with (
            patch.object(retriever, "embed_query", return_value=[0.0]),
            patch.object(retriever, "_vector_search", return_value=[
                {"source": "A.pdf", "page": 1, "text": "private", "score": 0.9}
            ]),
            patch.object(retriever, "rerank", side_effect=RuntimeError("rerank unavailable")),
            patch.object(retriever.config, "USE_HYBRID_SEARCH", False),
            patch.object(retriever.config, "ENABLE_RERANK", True),
        ):
            with self.assertRaisesRegex(RuntimeError, "rerank unavailable"):
                retriever._retrieve_with_client(
                    object(), "question", top_k=1, reranked_output=[]
                )


if __name__ == "__main__":
    unittest.main()
