import pytest

from relevanceflow.evaluation import (
    EvaluationError,
    evaluate_retriever,
)


@pytest.fixture
def ground_truth():
    return {
        "q1": {
            "p1": 2,
            "p2": 1,
            "p3": 0,
            "p4": 0,
        },
        "q2": {
            "p5": 2,
            "p6": 0,
            "p7": 1,
        },
    }


@pytest.fixture
def predictions():
    return {
        "q1": [
            "p1",
            "p3",
            "p2",
            "p4",
        ],
        "q2": [
            "p6",
            "p7",
            "p5",
        ],
    }


def test_evaluate_retriever_returns_expected_metrics(
    predictions,
    ground_truth,
):
    results = evaluate_retriever(
        predictions=predictions,
        ground_truth=ground_truth,
        k=[1, 5, 10, 20],
    )

    assert results["queries_evaluated"] == 2

    assert "recall@1" in results
    assert "recall@5" in results
    assert "recall@10" in results
    assert "recall@20" in results

    assert "mrr@10" in results

    assert "ndcg@5" in results
    assert "ndcg@10" in results
    assert "ndcg@20" in results

    assert "map@10" in results


def test_metric_ranges(
    predictions,
    ground_truth,
):
    results = evaluate_retriever(
        predictions=predictions,
        ground_truth=ground_truth,
        k=[1, 5, 10, 20],
    )

    metric_values = [
        value for key, value in results.items() if key != "queries_evaluated"
    ]

    for value in metric_values:
        assert 0.0 <= value <= 1.0


def test_perfect_ranking():
    ground_truth = {
        "q1": {
            "p1": 2,
            "p2": 1,
            "p3": 0,
        }
    }

    predictions = {
        "q1": [
            "p1",
            "p2",
            "p3",
        ]
    }

    results = evaluate_retriever(
        predictions=predictions,
        ground_truth=ground_truth,
        k=[1, 5, 10, 20],
    )

    assert results["recall@1"] == 0.5
    assert results["recall@5"] == 1.0
    assert results["recall@10"] == 1.0
    assert results["recall@20"] == 1.0

    assert results["mrr@10"] == 1.0
    assert results["ndcg@10"] == 1.0
    assert results["map@10"] == 1.0


def test_no_relevant_documents():
    ground_truth = {
        "q1": {
            "p1": 0,
            "p2": 0,
        }
    }

    predictions = {
        "q1": [
            "p1",
            "p2",
        ]
    }

    results = evaluate_retriever(
        predictions=predictions,
        ground_truth=ground_truth,
        k=[1, 5, 10, 20],
    )

    assert results["recall@10"] == 0.0
    assert results["mrr@10"] == 0.0
    assert results["ndcg@10"] == 0.0
    assert results["map@10"] == 0.0


def test_missing_query_overlap():
    predictions = {
        "q1": ["p1"],
    }

    ground_truth = {
        "q2": {
            "p1": 2,
        }
    }

    with pytest.raises(EvaluationError):
        evaluate_retriever(
            predictions=predictions,
            ground_truth=ground_truth,
        )


def test_empty_predictions():
    with pytest.raises(EvaluationError):
        evaluate_retriever(
            predictions={},
            ground_truth={
                "q1": {
                    "p1": 2,
                }
            },
        )


def test_invalid_k():
    with pytest.raises(EvaluationError):
        evaluate_retriever(
            predictions={
                "q1": ["p1"],
            },
            ground_truth={
                "q1": {
                    "p1": 2,
                }
            },
            k=[0, 5],
        )
