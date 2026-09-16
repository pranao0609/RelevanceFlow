import pandas as pd

from relevanceflow.data.clean import (
    LABEL_TO_SCORE,
    SCORE_TO_LABEL,
    clean_labels,
    clean_products,
    clean_queries,
    normalize_text,
    resolve_label_group,
)


def test_normalize_text():
    series = pd.Series(
        [
            "  office   chair  ",
            "wood\ttable",
            None,
        ]
    )

    result = normalize_text(series)

    assert result.iloc[0] == "office chair"
    assert result.iloc[1] == "wood table"
    assert result.iloc[2] == ""


def test_clean_queries_normalizes_text():
    queries = pd.DataFrame(
        {
            "query_id": [1],
            "query": ["  office   chair  "],
            "query_class": ["  Chairs  "],
        }
    )

    result = clean_queries(queries)

    assert result.loc[0, "query"] == "office chair"
    assert result.loc[0, "query_class"] == "Chairs"


def test_clean_queries_preserves_missing_query_class():
    queries = pd.DataFrame(
        {
            "query_id": [1],
            "query": ["office chair"],
            "query_class": [None],
        }
    )

    result = clean_queries(queries)

    assert pd.isna(result.loc[0, "query_class"])


def test_clean_products_normalizes_text():
    products = pd.DataFrame(
        {
            "product_id": [1],
            "product_name": ["  Office   Chair "],
            "product_class": [None],
            "category hierarchy": [" Furniture / Office "],
            "product_description": ["  Comfortable   chair "],
            "product_features": [None],
        }
    )

    result = clean_products(products)

    assert result.loc[0, "product_name"] == "Office Chair"
    assert result.loc[0, "category hierarchy"] == "Furniture / Office"
    assert result.loc[0, "product_description"] == "Comfortable chair"

    assert result.loc[0, "product_class"] == ""
    assert result.loc[0, "product_features"] == ""


def test_agreement_keeps_original_label():
    labels = pd.DataFrame(
        {
            "query_id": [1, 1],
            "product_id": [10, 10],
            "id": [1, 2],
            "label": ["Partial", "Partial"],
        }
    )

    label, method = resolve_label_group(labels)

    assert label == "Partial"
    assert method == "agreement"


def test_majority_vote():
    labels = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "product_id": [10, 10, 10],
            "id": [1, 2, 3],
            "label": ["Partial", "Partial", "Exact"],
        }
    )

    label, method = resolve_label_group(labels)

    assert label == "Partial"
    assert method == "majority_vote"


def test_tie_uses_conservative_lower_label():
    labels = pd.DataFrame(
        {
            "query_id": [1, 1],
            "product_id": [10, 10],
            "id": [1, 2],
            "label": ["Exact", "Partial"],
        }
    )

    label, method = resolve_label_group(labels)

    assert label == "Partial"
    assert method == "majority_tie_conservative"


def test_partial_irrelevant_tie_resolves_to_irrelevant():
    labels = pd.DataFrame(
        {
            "query_id": [1, 1],
            "product_id": [10, 10],
            "id": [1, 2],
            "label": ["Partial", "Irrelevant"],
        }
    )

    label, method = resolve_label_group(labels)

    assert label == "Irrelevant"
    assert method == "majority_tie_conservative"


def test_clean_labels_removes_duplicate_query_product_pairs():
    labels = pd.DataFrame(
        {
            "query_id": [1, 1, 2],
            "product_id": [10, 10, 20],
            "id": [1, 2, 3],
            "label": [
                "Partial",
                "Partial",
                "Exact",
            ],
        }
    )

    cleaned, audit = clean_labels(labels)

    assert len(cleaned) == 2
    assert len(audit) == 0

    assert set(
        zip(
            cleaned["query_id"],
            cleaned["product_id"],
        )
    ) == {(1, 10), (2, 20)}


def test_clean_labels_creates_relevance_score():
    labels = pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "product_id": [10, 20, 30],
            "id": [1, 2, 3],
            "label": [
                "Irrelevant",
                "Partial",
                "Exact",
            ],
        }
    )

    cleaned, _ = clean_labels(labels)

    scores = dict(
        zip(
            cleaned["label"],
            cleaned["relevance_score"],
        )
    )

    assert scores["Irrelevant"] == 0
    assert scores["Partial"] == 1
    assert scores["Exact"] == 2


def test_conflicting_labels_are_audited():
    labels = pd.DataFrame(
        {
            "query_id": [1, 1],
            "product_id": [10, 10],
            "id": [1, 2],
            "label": ["Exact", "Partial"],
        }
    )

    cleaned, audit = clean_labels(labels)

    assert len(cleaned) == 1
    assert len(audit) == 1

    assert audit.iloc[0]["query_id"] == 1
    assert audit.iloc[0]["product_id"] == 10
    assert audit.iloc[0]["original_labels"] == "Exact|Partial"
    assert audit.iloc[0]["resolution_method"] == "majority_tie_conservative"


def test_label_mappings_are_consistent():
    for label, score in LABEL_TO_SCORE.items():
        assert SCORE_TO_LABEL[score] == label
