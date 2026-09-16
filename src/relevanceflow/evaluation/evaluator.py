from __future__ import annotations

from collections.abc import Mapping, Sequence


class EvaluationError(ValueError):
    """Raised when retrieval evaluation input is invalid."""


def _validate_k_values(k: Sequence[int]) -> list[int]:
    """Validate and normalize requested cutoff values."""
    if not k:
        raise EvaluationError("At least one k value is required.")

    normalized = sorted({int(value) for value in k})

    if any(value <= 0 for value in normalized):
        raise EvaluationError("All k values must be greater than zero.")

    return normalized


def _validate_inputs(
    predictions: Mapping,
    ground_truth: Mapping,
) -> None:
    """Validate predictions and ground-truth structure."""
    if not isinstance(predictions, Mapping):
        raise EvaluationError(
            "predictions must be a mapping of query_id to ranked product IDs."
        )

    if not isinstance(ground_truth, Mapping):
        raise EvaluationError(
            "ground_truth must be a mapping of query_id to relevance mappings."
        )

    if not predictions:
        raise EvaluationError("predictions cannot be empty.")

    if not ground_truth:
        raise EvaluationError("ground_truth cannot be empty.")


def _relevance_at_rank(
    ranked_products: Sequence,
    relevance: Mapping,
) -> list[int]:
    """Convert ranked product IDs into relevance values."""
    return [int(relevance.get(product_id, 0)) for product_id in ranked_products]


def _recall_at_k(
    relevance_values: Sequence[int],
    total_relevant: int,
    k: int,
) -> float:
    """Binary Recall@K."""
    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(1 for value in relevance_values[:k] if value > 0)

    return retrieved_relevant / total_relevant


def _mrr_at_k(
    relevance_values: Sequence[int],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, relevance in enumerate(
        relevance_values[:k],
        start=1,
    ):
        if relevance > 0:
            return 1.0 / rank

    return 0.0


def _dcg(
    relevance_values: Sequence[int],
) -> float:
    """Compute graded DCG."""
    import math

    score = 0.0

    for rank, relevance in enumerate(
        relevance_values,
        start=1,
    ):
        gain = (2**relevance) - 1
        discount = math.log2(rank + 1)
        score += gain / discount

    return score


def _ndcg_at_k(
    relevance_values: Sequence[int],
    k: int,
) -> float:
    """Compute graded NDCG@K."""
    ranked = list(relevance_values[:k])

    if not ranked:
        return 0.0

    ideal = sorted(
        ranked,
        reverse=True,
    )

    ideal_dcg = _dcg(ideal)

    if ideal_dcg == 0.0:
        return 0.0

    return _dcg(ranked) / ideal_dcg


def _average_precision_at_k(
    relevance_values: Sequence[int],
    total_relevant: int,
    k: int,
) -> float:
    """Compute binary AP@K."""
    if total_relevant == 0:
        return 0.0

    relevant_seen = 0
    precision_sum = 0.0

    for rank, relevance in enumerate(
        relevance_values[:k],
        start=1,
    ):
        if relevance > 0:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    denominator = min(total_relevant, k)

    if denominator == 0:
        return 0.0

    return precision_sum / denominator


def evaluate_retriever(
    predictions: Mapping,
    ground_truth: Mapping,
    k: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, float | int]:
    """
    Evaluate ranked retrieval predictions against graded ground truth.

    Parameters
    ----------
    predictions:
        Mapping of query_id -> ordered sequence of product_ids.

    ground_truth:
        Mapping of query_id -> mapping of product_id -> relevance score.

    k:
        Ranking cutoffs for Recall@K and NDCG@K.

    Returns
    -------
    dict
        Aggregated macro-averaged retrieval metrics.

    Metrics
    -------
    Recall@K
    MRR@10
    NDCG@K
    MAP@10
    """
    _validate_inputs(
        predictions=predictions,
        ground_truth=ground_truth,
    )

    k_values = _validate_k_values(k)

    evaluated_queries = sorted(set(predictions).intersection(ground_truth))

    if not evaluated_queries:
        raise EvaluationError(
            "No overlapping query IDs between predictions and ground_truth."
        )

    results: dict[str, float | int] = {
        "queries_evaluated": len(evaluated_queries),
    }

    # Initialize metric accumulators.
    recall_scores = {cutoff: [] for cutoff in k_values}

    ndcg_scores = {cutoff: [] for cutoff in k_values}

    mrr_scores = []
    map_scores = []

    for query_id in evaluated_queries:
        ranked_products = predictions[query_id]
        relevance = ground_truth[query_id]

        if not isinstance(ranked_products, Sequence):
            raise EvaluationError(
                f"Predictions for query '{query_id}' must be a sequence."
            )

        if not isinstance(relevance, Mapping):
            raise EvaluationError(
                f"Ground truth for query '{query_id}' must be a mapping."
            )

        relevance_values = _relevance_at_rank(
            ranked_products=ranked_products,
            relevance=relevance,
        )

        total_relevant = sum(1 for value in relevance.values() if int(value) > 0)

        for cutoff in k_values:
            recall_scores[cutoff].append(
                _recall_at_k(
                    relevance_values=relevance_values,
                    total_relevant=total_relevant,
                    k=cutoff,
                )
            )

            ndcg_scores[cutoff].append(
                _ndcg_at_k(
                    relevance_values=relevance_values,
                    k=cutoff,
                )
            )

        mrr_scores.append(
            _mrr_at_k(
                relevance_values=relevance_values,
                k=10,
            )
        )

        map_scores.append(
            _average_precision_at_k(
                relevance_values=relevance_values,
                total_relevant=total_relevant,
                k=10,
            )
        )

    # Aggregate query-level metrics using macro averaging.
    for cutoff in k_values:
        results[f"recall@{cutoff}"] = sum(recall_scores[cutoff]) / len(
            evaluated_queries
        )

        results[f"ndcg@{cutoff}"] = sum(ndcg_scores[cutoff]) / len(evaluated_queries)

    results["mrr@10"] = sum(mrr_scores) / len(evaluated_queries)
    results["map@10"] = sum(map_scores) / len(evaluated_queries)

    return results
