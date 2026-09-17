import pytest

from relevanceflow.evaluation.common_ranking import (
    average_precision_at_k_from_candidates,
    evaluate_query,
    evaluate_rankings,
    ndcg_at_k_from_candidates,
    recall_at_k_from_candidates,
    reciprocal_rank_at_k,
)


def test_recall_uses_complete_candidate_set():
    ranked = [2, 0, 0]
    candidates = [2, 1, 0, 0, 0]

    # 1 relevant retrieved / 2 total relevant
    assert (
        recall_at_k_from_candidates(
            ranked,
            candidates,
            k=3,
        )
        == 0.5
    )


def test_recall_zero_when_no_relevant_candidates():
    ranked = [0, 0, 0]
    candidates = [0, 0, 0]

    assert (
        recall_at_k_from_candidates(
            ranked,
            candidates,
            k=3,
        )
        == 0.0
    )


def test_mrr_first_relevant():
    ranked = [0, 0, 2, 1]

    assert reciprocal_rank_at_k(
        ranked,
        k=4,
    ) == pytest.approx(1 / 3)


def test_ndcg_uses_complete_candidate_set():
    ranked = [2, 0, 0]
    candidates = [2, 1, 0, 0]

    value = ndcg_at_k_from_candidates(
        ranked,
        candidates,
        k=3,
    )

    assert 0.0 <= value <= 1.0


def test_map_uses_complete_candidate_set():
    ranked = [2, 0, 0]
    candidates = [2, 1, 0, 0]

    value = average_precision_at_k_from_candidates(
        ranked,
        candidates,
        k=3,
    )

    # One relevant retrieved.
    # Two relevant exist in the candidate set.
    assert value == pytest.approx(0.5)


def test_evaluate_query():
    ground_truth = {
        101: 2,
        102: 1,
        103: 0,
    }

    ranked = [101, 103, 102]

    metrics = evaluate_query(
        ranked_product_ids=ranked,
        candidate_relevance=ground_truth,
        k=3,
    )

    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert metrics["ndcg@10"] > 0.0
    assert metrics["map@10"] == pytest.approx(5 / 6)


def test_evaluate_rankings():
    predictions = {
        1: [10, 11, 12],
        2: [20, 21, 22],
    }

    ground_truth = {
        1: {
            10: 2,
            11: 0,
            12: 1,
        },
        2: {
            20: 0,
            21: 2,
            22: 0,
        },
    }

    metrics = evaluate_rankings(
        predictions,
        ground_truth,
        k=10,
    )

    assert metrics["queries_evaluated"] == 2
    assert 0.0 <= metrics["recall@10"] <= 1.0
    assert 0.0 <= metrics["mrr@10"] <= 1.0
    assert 0.0 <= metrics["ndcg@10"] <= 1.0
    assert 0.0 <= metrics["map@10"] <= 1.0
