import numpy as np
import pytest

from relevanceflow.evaluation.ranking_metrics import (
    RankingMetricError,
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_recall_at_k() -> None:
    relevance = np.array([2, 0, 1, 0, 0])

    assert recall_at_k(relevance, 3) == pytest.approx(1.0)


def test_recall_at_k_partial() -> None:
    relevance = np.array([2, 0, 0, 1, 0])

    assert recall_at_k(relevance, 2) == pytest.approx(0.5)


def test_recall_without_relevant_items() -> None:
    relevance = np.array([0, 0, 0])

    assert recall_at_k(relevance, 10) == 0.0


def test_reciprocal_rank() -> None:
    relevance = np.array([0, 0, 2, 0])

    assert reciprocal_rank_at_k(relevance, 10) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_relevant() -> None:
    relevance = np.array([0, 0, 0])

    assert reciprocal_rank_at_k(relevance, 10) == 0.0


def test_ndcg_perfect_ranking() -> None:
    relevance = np.array([2, 1, 0])

    assert ndcg_at_k(relevance, 3) == pytest.approx(1.0)


def test_ndcg_zero_relevance() -> None:
    relevance = np.array([0, 0, 0])

    assert ndcg_at_k(relevance, 3) == 0.0


def test_average_precision_perfect() -> None:
    relevance = np.array([2, 1, 0])

    assert average_precision_at_k(relevance, 3) == pytest.approx(1.0)


def test_average_precision_no_relevant() -> None:
    relevance = np.array([0, 0, 0])

    assert average_precision_at_k(relevance, 3) == 0.0


def test_invalid_k() -> None:
    with pytest.raises(RankingMetricError):
        ndcg_at_k([1, 0], 0)
