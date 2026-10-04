"""Experimental selection of varied evidence from a reranked candidate list."""

import os

from search_tokenizer import tokenize_search_text


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def select_diverse_evidence(candidates: list[dict], top_k: int) -> list[dict]:
    """Prefer distinct source/pages, then balance rerank order and text overlap.

    This does not infer question subgoals or guarantee factual coverage. It is an
    opt-in retrieval experiment, not a replacement for the reranker score.
    """
    if top_k <= 0 or not candidates:
        return []

    token_sets = [set(tokenize_search_text(hit["text"])) for hit in candidates]
    selected_indices = []
    selected_pages = set()
    remaining = set(range(len(candidates)))

    while remaining and len(selected_indices) < top_k:
        unused_pages = {
            index for index in remaining
            if (candidates[index]["source"], candidates[index]["page"]) not in selected_pages
        }
        pool = unused_pages or remaining

        def utility(index: int) -> tuple[float, int]:
            relevance = 1 - index / len(candidates)
            overlap = max(
                (_jaccard(token_sets[index], token_sets[chosen]) for chosen in selected_indices),
                default=0.0,
            )
            return relevance - 0.25 * overlap, -index

        best = max(pool, key=utility)
        selected_indices.append(best)
        selected_pages.add((candidates[best]["source"], candidates[best]["page"]))
        remaining.remove(best)

    return [candidates[index] for index in selected_indices]


COMPARISON_MARKERS = ("分别", "各自", "两篇", "两个", "比较", "对比", "不同", "差异")
QUERY_STOPWORDS = {"the", "and", "for", "with", "iot", "pdf", "paper"}


def _required_sources(question: str, candidates: list[dict]) -> list[str]:
    """Infer paper-level evidence needs without using evaluation labels."""
    sources = list(dict.fromkeys(hit["source"] for hit in candidates))
    lowered = question.casefold()
    named = [
        source for source in sources
        if len(stem := os.path.splitext(os.path.basename(source))[0]) >= 4
        and stem.casefold() in lowered
    ]
    if named:
        return named
    if len(sources) == 2 and any(marker in question for marker in COMPARISON_MARKERS):
        return sources
    return []


def select_requirement_evidence(
    question: str, candidates: list[dict], top_k: int
) -> list[dict]:
    """Cover named/comparative papers, then choose relevant nonredundant pages.

    This is a deterministic experiment on the existing rerank output; it does
    not generate subquestions or call another model. Missing source evidence is
    never fabricated, and this selector is deliberately opt-in.
    """
    if top_k <= 0 or not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates[:]

    required = _required_sources(question, candidates)
    source_stems = {
        os.path.splitext(os.path.basename(source))[0].casefold()
        for source in required
    }
    anchors = {
        token for token in tokenize_search_text(question)
        if token.isascii() and (len(token) >= 3 or token.isdigit())
        and token not in QUERY_STOPWORDS and token not in source_stems
    }
    token_sets = [set(tokenize_search_text(hit["text"])) for hit in candidates]
    selected = []
    selected_pages = set()
    covered_anchors = set()
    remaining = set(range(len(candidates)))

    def choose(index: int) -> None:
        selected.append(index)
        remaining.remove(index)
        selected_pages.add((candidates[index]["source"], candidates[index]["page"]))
        covered_anchors.update(anchors & token_sets[index])

    # Reserve one slot per requested paper before filling by global relevance.
    for source in required[:top_k]:
        pool = [index for index in remaining if candidates[index]["source"] == source]
        if not pool:
            continue
        best = max(
            pool,
            key=lambda index: (
                1 - index / len(candidates)
                + 0.15 * len(anchors & token_sets[index]),
                -index,
            ),
        )
        choose(best)

    while remaining and len(selected) < top_k:
        unused_pages = {
            index for index in remaining
            if (candidates[index]["source"], candidates[index]["page"])
            not in selected_pages
        }
        pool = unused_pages or remaining

        def utility(index: int) -> tuple[float, int]:
            relevance = 1 - index / len(candidates)
            new_anchors = len((anchors & token_sets[index]) - covered_anchors)
            overlap = max(
                (_jaccard(token_sets[index], token_sets[chosen]) for chosen in selected),
                default=0.0,
            )
            return relevance + 0.2 * new_anchors - 0.15 * overlap, -index

        choose(max(pool, key=utility))

    return [candidates[index] for index in selected]
