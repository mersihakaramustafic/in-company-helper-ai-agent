"""Deterministic metrics. Everything is compared at page level (Notion page_id):
page IDs survive re-ingestion, while chunk IDs are regenerated whenever a page changes."""
from typing import Iterable, List, Optional, Tuple


def unique_in_order(ids: Iterable[str]) -> List[str]:
    """Collapses a ranked chunk list to a ranked page list (first occurrence wins)."""
    seen = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def recall_at_k(ranked_ids: List[str], expected_ids: List[str], k: int) -> Optional[float]:
    """Share of expected pages found in the top-k retrieved pages.
    None when nothing is expected (unanswerable questions), so they don't skew the mean."""
    if not expected_ids:
        return None
    return len(set(ranked_ids[:k]) & set(expected_ids)) / len(set(expected_ids))


def reciprocal_rank(ranked_ids: List[str], expected_ids: List[str]) -> Optional[float]:
    """1/rank of the first relevant page; 0 if none was retrieved. Averaged over items = MRR."""
    if not expected_ids:
        return None
    expected = set(expected_ids)
    for rank, page_id in enumerate(ranked_ids, start=1):
        if page_id in expected:
            return 1 / rank
    return 0.0


def citation_precision_recall(
    cited_ids: List[str], expected_ids: List[str], should_abstain: bool
) -> Tuple[float, Optional[float]]:
    """Precision: share of citations that point at an expected page (citing nothing
    is precise). Recall: share of expected pages that were cited; None for abstain
    items, where expected_sources only lists pages that are acceptable to cite."""
    cited, expected = set(cited_ids), set(expected_ids)
    precision = len(cited & expected) / len(cited) if cited else 1.0
    recall = None if should_abstain or not expected else len(cited & expected) / len(expected)
    return precision, recall
