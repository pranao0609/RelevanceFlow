from __future__ import annotations

import math


def safe_numeric(
    value: object,
    default: float = 0.0,
) -> float:
    """Convert a value to a finite float."""
    try:
        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def metadata_features(
    average_rating: object,
    rating_count: object,
    review_count: object,
) -> dict[str, float]:
    """Generate product metadata features."""
    return {
        "average_rating": safe_numeric(average_rating),
        "rating_count": safe_numeric(rating_count),
        "review_count": safe_numeric(review_count),
    }
