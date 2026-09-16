from __future__ import annotations

from .lexical import normalize_text, tokenize


def category_match(
    query: object,
    category_hierarchy: object,
) -> float:
    """
    Fraction of unique query tokens present in the category.
    """
    query_tokens = set(tokenize(query))
    category_tokens = set(tokenize(category_hierarchy))

    if not query_tokens:
        return 0.0

    return len(query_tokens & category_tokens) / len(query_tokens)


def product_class_match(
    query: object,
    product_class: object,
) -> float:
    """
    Token overlap between query and product class.
    """
    query_tokens = set(tokenize(query))
    class_tokens = set(tokenize(product_class))

    if not query_tokens or not class_tokens:
        return 0.0

    return float(bool(query_tokens & class_tokens))


def phrase_match(
    query: object,
    product_name: object,
) -> float:
    """
    Check whether the normalized query occurs as a phrase
    inside the normalized product name.
    """
    query_text = normalize_text(query)
    product_text = normalize_text(product_name)

    if not query_text or not product_text:
        return 0.0

    return float(query_text in product_text)


def compatibility_features(
    query: object,
    product_name: object,
    category_hierarchy: object,
    product_class: object,
) -> dict[str, float]:
    """Generate query-product compatibility features."""
    return {
        "category_match": category_match(
            query,
            category_hierarchy,
        ),
        "product_class_match": product_class_match(
            query,
            product_class,
        ),
        "phrase_match": phrase_match(
            query,
            product_name,
        ),
    }
