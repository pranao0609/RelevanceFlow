from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _tokens(text: object) -> list[str]:
    if text is None:
        return []

    return TOKEN_PATTERN.findall(str(text).lower())


def title_length(product_name: object) -> float:
    """Number of tokens in the product title."""
    return float(len(_tokens(product_name)))


def description_length(
    product_description: object,
) -> float:
    """Number of tokens in the product description."""
    return float(len(_tokens(product_description)))


def feature_count(
    product_features: object,
) -> float:
    """
    Count product feature items.

    WANDS product_features is treated as a text field.
    We count non-empty lines first, falling back to tokens.
    """
    if product_features is None:
        return 0.0

    text = str(product_features).strip()

    if not text:
        return 0.0

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) > 1:
        return float(len(lines))

    return float(len(_tokens(text)))


def category_depth(
    category_hierarchy: object,
) -> float:
    """
    Estimate category hierarchy depth.

    Supports common delimiters used in hierarchy strings.
    """
    if category_hierarchy is None:
        return 0.0

    text = str(category_hierarchy).strip()

    if not text:
        return 0.0

    for delimiter in (
        " > ",
        ">",
        " / ",
        "/",
        "|",
    ):
        if delimiter in text:
            parts = [part.strip() for part in text.split(delimiter) if part.strip()]

            return float(len(parts))

    return 1.0
