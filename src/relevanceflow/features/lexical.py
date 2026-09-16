from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def normalize_text(text: object) -> str:
    """Normalize text for lexical comparison."""
    if text is None:
        return ""

    return " ".join(str(text).lower().split())


def tokenize(text: object) -> list[str]:
    """Tokenize normalized text."""
    normalized = normalize_text(text)

    return TOKEN_PATTERN.findall(normalized)


def exact_match(
    query: object,
    product_name: object,
) -> float:
    """Return 1 when normalized query and title are identical."""
    query_text = normalize_text(query)
    product_text = normalize_text(product_name)

    if not query_text or not product_text:
        return 0.0

    return float(query_text == product_text)


def token_overlap(
    query: object,
    product_name: object,
) -> float:
    """
    Compute Jaccard token overlap.

    |Q ∩ P| / |Q ∪ P|
    """
    query_tokens = set(tokenize(query))
    product_tokens = set(tokenize(product_name))

    if not query_tokens or not product_tokens:
        return 0.0

    return len(query_tokens & product_tokens) / len(query_tokens | product_tokens)


def query_title_overlap(
    query: object,
    product_name: object,
) -> float:
    """Count unique query tokens appearing in the product title."""
    query_tokens = set(tokenize(query))
    product_tokens = set(tokenize(product_name))

    return float(len(query_tokens & product_tokens))


def character_overlap(
    query: object,
    product_name: object,
    n: int = 3,
) -> float:
    """
    Compute character n-gram Jaccard overlap.
    """
    query_text = normalize_text(query)
    product_text = normalize_text(product_name)

    if not query_text or not product_text:
        return 0.0

    if len(query_text) < n or len(product_text) < n:
        return float(query_text == product_text)

    query_ngrams = {
        query_text[index : index + n] for index in range(len(query_text) - n + 1)
    }

    product_ngrams = {
        product_text[index : index + n] for index in range(len(product_text) - n + 1)
    }

    union = query_ngrams | product_ngrams

    if not union:
        return 0.0

    return len(query_ngrams & product_ngrams) / len(union)


def lexical_features(
    query: object,
    product_name: object,
    bm25_score: float = 0.0,
    tfidf_similarity: float = 0.0,
) -> dict[str, float]:
    """Generate lexical query-product features."""
    return {
        "bm25_score": float(bm25_score),
        "tfidf_similarity": float(tfidf_similarity),
        "exact_match": exact_match(
            query,
            product_name,
        ),
        "token_overlap": token_overlap(
            query,
            product_name,
        ),
        "character_overlap": character_overlap(
            query,
            product_name,
        ),
        "query_title_overlap": query_title_overlap(
            query,
            product_name,
        ),
    }
