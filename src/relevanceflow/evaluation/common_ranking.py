from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class CommonRankingEvaluationError(ValueError):
    """Raised when common ranking evaluation inputs are invalid."""


def _validate_relevance(
    relevance: Sequence[float],
) -> np.ndarray:
    values = np.asarray(
        relevance,
        dtype=float,
    )

    if values.ndim != 1:
        raise CommonRankingEvaluationError("Relevance must be one-dimensional.")

    return values


def recall_at_k_from_candidates(
    ranked_relevance: Sequence[float],
    candidate_relevance: Sequence[float],
    k: int = 10,
) -> float:
    """
    Calculate Recall@k using the complete candidate set.

    Numerator:
        Relevant products retrieved in top-k.

    Denominator:
        Relevant products in the complete candidate set.
    """

    if k <= 0:
        raise CommonRankingEvaluationError("k must be greater than zero.")

    ranked = _validate_relevance(ranked_relevance)

    candidates = _validate_relevance(candidate_relevance)

    total_relevant = int(np.sum(candidates > 0))

    if total_relevant == 0:
        return 0.0

    top_k = ranked[:k]

    retrieved_relevant = int(np.sum(top_k > 0))

    return float(retrieved_relevant / total_relevant)


def reciprocal_rank_at_k(
    ranked_relevance: Sequence[float],
    k: int = 10,
) -> float:
    """
    Calculate MRR contribution for one query.
    """

    if k <= 0:
        raise CommonRankingEvaluationError("k must be greater than zero.")

    ranked = _validate_relevance(ranked_relevance)

    top_k = ranked[:k]

    relevant_positions = np.flatnonzero(top_k > 0)

    if len(relevant_positions) == 0:
        return 0.0

    return float(1.0 / (relevant_positions[0] + 1))


def _dcg(
    relevance: np.ndarray,
    k: int,
) -> float:
    """Calculate DCG@k using graded relevance."""

    values = relevance[:k]

    if len(values) == 0:
        return 0.0

    gains = (2.0**values) - 1.0

    discounts = np.log2(
        np.arange(
            2,
            len(values) + 2,
        )
    )

    return float(np.sum(gains / discounts))


def ndcg_at_k_from_candidates(
    ranked_relevance: Sequence[float],
    candidate_relevance: Sequence[float],
    k: int = 10,
) -> float:
    """
    Calculate graded NDCG@k.

    The ideal ranking is constructed from the complete
    candidate set, not only the retrieved top-k results.
    """

    if k <= 0:
        raise CommonRankingEvaluationError("k must be greater than zero.")

    ranked = _validate_relevance(ranked_relevance)

    candidates = _validate_relevance(candidate_relevance)

    dcg = _dcg(
        ranked,
        k,
    )

    ideal = np.sort(candidates)[::-1]

    ideal_dcg = _dcg(
        ideal,
        k,
    )

    if ideal_dcg == 0.0:
        return 0.0

    return float(dcg / ideal_dcg)


def average_precision_at_k_from_candidates(
    ranked_relevance: Sequence[float],
    candidate_relevance: Sequence[float],
    k: int = 10,
) -> float:
    """
    Calculate MAP contribution for one query.

    Relevance is binary:
        relevance > 0 -> relevant

    The denominator uses the number of relevant products
    in the complete candidate set, capped at k.
    """

    if k <= 0:
        raise CommonRankingEvaluationError("k must be greater than zero.")

    ranked = _validate_relevance(ranked_relevance)

    candidates = _validate_relevance(candidate_relevance)

    top_k = ranked[:k]

    total_relevant = int(np.sum(candidates > 0))

    if total_relevant == 0:
        return 0.0

    relevant_seen = 0
    precision_sum = 0.0

    for index, value in enumerate(
        top_k,
        start=1,
    ):
        if value > 0:
            relevant_seen += 1

            precision_sum += relevant_seen / index

    denominator = min(
        total_relevant,
        k,
    )

    return float(precision_sum / denominator)


def evaluate_query(
    ranked_product_ids: Sequence[int],
    candidate_relevance: Mapping[int, int],
    k: int = 10,
) -> dict[str, float]:
    """
    Evaluate one ranked query against its complete
    candidate relevance mapping.
    """

    if k <= 0:
        raise CommonRankingEvaluationError("k must be greater than zero.")

    ranked_ids = [int(product_id) for product_id in ranked_product_ids]

    missing_ids = [
        product_id for product_id in ranked_ids if product_id not in candidate_relevance
    ]

    if missing_ids:
        raise CommonRankingEvaluationError(
            "Ranked products are missing from "
            "the candidate ground truth: "
            f"{missing_ids[:10]}"
        )

    ranked_relevance = [
        int(candidate_relevance[product_id]) for product_id in ranked_ids
    ]

    all_relevance = [int(value) for value in candidate_relevance.values()]

    return {
        "recall@10": (
            recall_at_k_from_candidates(
                ranked_relevance,
                all_relevance,
                k,
            )
        ),
        "mrr@10": (
            reciprocal_rank_at_k(
                ranked_relevance,
                k,
            )
        ),
        "ndcg@10": (
            ndcg_at_k_from_candidates(
                ranked_relevance,
                all_relevance,
                k,
            )
        ),
        "map@10": (
            average_precision_at_k_from_candidates(
                ranked_relevance,
                all_relevance,
                k,
            )
        ),
    }


def evaluate_rankings(
    predictions: Mapping[
        int,
        Sequence[int],
    ],
    ground_truth: Mapping[
        int,
        Mapping[int, int],
    ],
    k: int = 10,
) -> dict[str, float]:
    """
    Evaluate multiple query rankings using one common protocol.

    Parameters
    ----------
    predictions:
        query_id -> ordered product IDs

    ground_truth:
        query_id -> product_id -> relevance score

    k:
        Ranking cutoff.

    Returns
    -------
    Macro-averaged ranking metrics.
    """

    if not predictions:
        raise CommonRankingEvaluationError("Predictions cannot be empty.")

    query_ids = sorted(set(predictions) & set(ground_truth))

    if not query_ids:
        raise CommonRankingEvaluationError(
            "No overlapping query IDs " "between predictions and ground truth."
        )

    per_query = []

    for query_id in query_ids:
        metrics = evaluate_query(
            ranked_product_ids=predictions[query_id],
            candidate_relevance=ground_truth[query_id],
            k=k,
        )

        per_query.append(metrics)

    metric_names = [
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "map@10",
    ]

    results = {"queries_evaluated": len(query_ids)}

    for metric_name in metric_names:
        values = [row[metric_name] for row in per_query]

        results[metric_name] = float(np.mean(values))

    return results
