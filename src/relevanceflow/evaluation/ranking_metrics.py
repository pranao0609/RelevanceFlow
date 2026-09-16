from __future__ import annotations

import numpy as np


class RankingMetricError(ValueError):
    """Raised when ranking metric inputs are invalid."""


def _validate_inputs(
    relevance: list[float] | np.ndarray,
    k: int,
) -> np.ndarray:
    if k <= 0:
        raise RankingMetricError("k must be greater than zero.")

    values = np.asarray(relevance, dtype=float)

    if values.ndim != 1:
        raise RankingMetricError("relevance must be one-dimensional.")

    return values[:k]


def recall_at_k(
    relevance: list[float] | np.ndarray,
    k: int = 10,
) -> float:
    """
    Recall@k for a ranked list.

    Binary relevance:
        relevance > 0 → relevant
    """

    values = np.asarray(relevance, dtype=float)

    if values.ndim != 1:
        raise RankingMetricError("relevance must be one-dimensional.")

    total_relevant = np.sum(values > 0)

    if total_relevant == 0:
        return 0.0

    top_k = values[:k]

    return float(np.sum(top_k > 0) / total_relevant)


def reciprocal_rank_at_k(
    relevance: list[float] | np.ndarray,
    k: int = 10,
) -> float:
    """Calculate Reciprocal Rank@k."""

    values = _validate_inputs(relevance, k)

    relevant_positions = np.flatnonzero(values > 0)

    if len(relevant_positions) == 0:
        return 0.0

    return float(1.0 / (relevant_positions[0] + 1))


def ndcg_at_k(
    relevance: list[float] | np.ndarray,
    k: int = 10,
) -> float:
    """Calculate graded NDCG@k."""

    values = _validate_inputs(relevance, k)

    if len(values) == 0:
        return 0.0

    gains = (2.0**values) - 1.0
    discounts = np.log2(np.arange(2, len(values) + 2))

    dcg = np.sum(gains / discounts)

    ideal_values = np.sort(values)[::-1]

    ideal_gains = (2.0**ideal_values) - 1.0

    ideal_dcg = np.sum(ideal_gains / discounts)

    if ideal_dcg == 0:
        return 0.0

    return float(dcg / ideal_dcg)


def average_precision_at_k(
    relevance: list[float] | np.ndarray,
    k: int = 10,
) -> float:
    """
    Calculate Average Precision@k.

    Binary relevance:
        relevance > 0 → relevant
    """

    values = _validate_inputs(relevance, k)

    binary = values > 0

    total_relevant = np.sum(binary)

    if total_relevant == 0:
        return 0.0

    precisions = []

    relevant_seen = 0

    for index, is_relevant in enumerate(binary, start=1):
        if is_relevant:
            relevant_seen += 1
            precisions.append(relevant_seen / index)

    return float(np.sum(precisions) / min(total_relevant, k))
