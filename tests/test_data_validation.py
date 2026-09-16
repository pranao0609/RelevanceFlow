import pandas as pd
import pytest

from relevanceflow.data.validate import (
    VALID_LABELS,
    DataValidationError,
    raise_if_invalid,
    validate_labels,
    validate_products,
    validate_queries,
    validate_wands,
    validation_summary,
)


def make_valid_queries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 2],
            "query": ["office chair", "wood table"],
            "query_class": ["Office Chairs", "Tables"],
        }
    )


def make_valid_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [101, 102],
            "product_name": [
                "Ergonomic Office Chair",
                "Solid Wood Table",
            ],
            "product_class": [
                "Office Chairs",
                "Dining Tables",
            ],
            "category hierarchy": [
                "Furniture / Office / Chairs",
                "Furniture / Dining / Tables",
            ],
            "product_description": [
                "Comfortable office chair",
                "Solid wood dining table",
            ],
            "product_features": [
                "material:mesh|arms:yes",
                "material:wood|shape:rectangular",
            ],
            "rating_count": [10, 20],
            "average_rating": [4.5, 4.2],
            "review_count": [5, 10],
        }
    )


def make_valid_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "query_id": [1, 1, 2],
            "product_id": [101, 102, 102],
            "label": ["Exact", "Partial", "Irrelevant"],
        }
    )


def test_valid_queries_pass():
    result = validate_queries(make_valid_queries())

    assert result.is_valid
    assert result.errors == []


def test_valid_products_pass():
    result = validate_products(make_valid_products())

    assert result.is_valid
    assert result.errors == []


def test_valid_labels_pass():
    result = validate_labels(make_valid_labels())

    assert result.is_valid
    assert result.errors == []


def test_duplicate_query_ids_fail():
    queries = make_valid_queries()
    queries.loc[1, "query_id"] = 1

    result = validate_queries(queries)

    assert not result.is_valid
    assert any("duplicate query_id" in error for error in result.errors)


def test_duplicate_product_ids_fail():
    products = make_valid_products()
    products.loc[1, "product_id"] = 101

    result = validate_products(products)

    assert not result.is_valid
    assert any("duplicate product_id" in error for error in result.errors)


def test_empty_query_fails():
    queries = make_valid_queries()
    queries.loc[0, "query"] = "   "

    result = validate_queries(queries)

    assert not result.is_valid


def test_missing_query_class_is_warning():
    queries = make_valid_queries()
    queries.loc[0, "query_class"] = None

    result = validate_queries(queries)

    assert result.is_valid
    assert len(result.warnings) == 1
    assert "query_class" in result.warnings[0]


def test_missing_product_name_fails():
    products = make_valid_products()
    products.loc[0, "product_name"] = None

    result = validate_products(products)

    assert not result.is_valid


def test_missing_optional_product_field_is_warning():
    products = make_valid_products()
    products.loc[0, "product_description"] = None

    result = validate_products(products)

    assert result.is_valid
    assert len(result.warnings) == 1


@pytest.mark.parametrize(
    "label",
    sorted(VALID_LABELS),
)
def test_valid_labels_are_accepted(label):
    labels = make_valid_labels()
    labels.loc[0, "label"] = label

    result = validate_labels(labels)

    assert result.is_valid


@pytest.mark.parametrize(
    "invalid_label",
    ["Unknown", "BadLabel", "", "exact"],
)
def test_invalid_labels_fail(invalid_label):
    labels = make_valid_labels()
    labels.loc[0, "label"] = invalid_label

    result = validate_labels(labels)

    assert not result.is_valid
    assert any("invalid labels" in error for error in result.errors)


def test_duplicate_judgment_ids_fail():
    labels = make_valid_labels()
    labels.loc[1, "id"] = 1

    result = validate_labels(labels)

    assert not result.is_valid
    assert any("duplicate judgment IDs" in error for error in result.errors)


def test_duplicate_query_product_pair_without_conflict_is_warning():
    labels = make_valid_labels()

    duplicate_row = labels.iloc[[0]].copy()
    duplicate_row["id"] = 99

    labels = pd.concat(
        [labels, duplicate_row],
        ignore_index=True,
    )

    result = validate_labels(labels)

    assert result.is_valid

    assert any(
        "duplicate query-product pairs" in warning for warning in result.warnings
    )


def test_conflicting_query_product_pair_fails():
    labels = make_valid_labels()

    conflicting_row = labels.iloc[[0]].copy()
    conflicting_row["id"] = 99
    conflicting_row["label"] = "Irrelevant"

    labels = pd.concat(
        [labels, conflicting_row],
        ignore_index=True,
    )

    result = validate_labels(labels)

    assert not result.is_valid

    assert any("conflicting relevance labels" in error for error in result.errors)


def test_invalid_foreign_key_query_fails():
    labels = make_valid_labels()
    labels.loc[0, "query_id"] = 999

    result = validate_wands(
        queries=make_valid_queries(),
        products=make_valid_products(),
        labels=labels,
    )

    assert not result.is_valid

    assert any("query_id values" in error for error in result.errors)


def test_invalid_foreign_key_product_fails():
    labels = make_valid_labels()
    labels.loc[0, "product_id"] = 999

    result = validate_wands(
        queries=make_valid_queries(),
        products=make_valid_products(),
        labels=labels,
    )

    assert not result.is_valid

    assert any("product_id values" in error for error in result.errors)


def test_full_valid_dataset_passes():
    result = validate_wands(
        queries=make_valid_queries(),
        products=make_valid_products(),
        labels=make_valid_labels(),
    )

    assert result.is_valid


def test_validation_summary():
    result = validate_queries(make_valid_queries())

    summary = validation_summary(result)

    assert summary["valid"] is True
    assert "total_checks" in summary
    assert "passed_checks" in summary
    assert "failed_checks" in summary
    assert "warning_checks" in summary
    assert "errors" in summary
    assert "warnings" in summary
    assert "checks" in summary


def test_raise_if_invalid_does_not_raise_for_valid_result():
    result = validate_queries(make_valid_queries())

    raise_if_invalid(result)


def test_raise_if_invalid_raises_for_invalid_result():
    queries = make_valid_queries()
    queries.loc[1, "query_id"] = 1

    result = validate_queries(queries)

    with pytest.raises(DataValidationError):
        raise_if_invalid(result)
