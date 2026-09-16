import pandas as pd
import pytest

from relevanceflow.data.split import (
    SplitConfig,
    SplitError,
    assign_split,
    split_judgments,
    verify_query_disjointness,
)


@pytest.fixture
def sample_judgments() -> pd.DataFrame:
    rows = []

    for query_id in range(20):
        for product_id in range(5):
            rows.append(
                {
                    "query_id": query_id,
                    "product_id": query_id * 100 + product_id,
                    "label": "Exact" if product_id == 0 else "Irrelevant",
                    "relevance_score": 2 if product_id == 0 else 0,
                }
            )

    return pd.DataFrame(rows)


def test_split_configuration_validates() -> None:
    config = SplitConfig(
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
    )

    config.validate()


def test_invalid_split_configuration_raises() -> None:
    config = SplitConfig(
        train_size=0.60,
        validation_size=0.15,
        test_size=0.15,
    )

    with pytest.raises(SplitError):
        config.validate()


def test_query_ids_are_disjoint(sample_judgments: pd.DataFrame) -> None:
    splits = split_judgments(sample_judgments)

    train_queries = set(splits["train"]["query_id"])
    validation_queries = set(splits["validation"]["query_id"])
    test_queries = set(splits["test"]["query_id"])

    assert train_queries.isdisjoint(validation_queries)
    assert train_queries.isdisjoint(test_queries)
    assert validation_queries.isdisjoint(test_queries)


def test_all_rows_are_preserved(sample_judgments: pd.DataFrame) -> None:
    splits = split_judgments(sample_judgments)

    total_rows = sum(len(dataframe) for dataframe in splits.values())

    assert total_rows == len(sample_judgments)


def test_all_queries_are_preserved(sample_judgments: pd.DataFrame) -> None:
    splits = split_judgments(sample_judgments)

    original_queries = set(sample_judgments["query_id"])

    split_queries = (
        set(splits["train"]["query_id"])
        | set(splits["validation"]["query_id"])
        | set(splits["test"]["query_id"])
    )

    assert split_queries == original_queries


def test_same_seed_produces_same_split(
    sample_judgments: pd.DataFrame,
) -> None:
    config = SplitConfig(random_seed=42)

    first = split_judgments(
        sample_judgments,
        config=config,
    )

    second = split_judgments(
        sample_judgments,
        config=config,
    )

    for split_name in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(
            first[split_name],
            second[split_name],
        )


def test_different_seed_can_change_split(
    sample_judgments: pd.DataFrame,
) -> None:
    first = split_judgments(
        sample_judgments,
        config=SplitConfig(random_seed=42),
    )

    second = split_judgments(
        sample_judgments,
        config=SplitConfig(random_seed=123),
    )

    first_train = set(first["train"]["query_id"])
    second_train = set(second["train"]["query_id"])

    assert first_train != second_train


def test_query_never_split_across_rows(
    sample_judgments: pd.DataFrame,
) -> None:
    splits = split_judgments(sample_judgments)

    for query_id in sample_judgments["query_id"].unique():
        appearances = sum(
            query_id in set(dataframe["query_id"]) for dataframe in splits.values()
        )

        assert appearances == 1


def test_missing_query_id_raises() -> None:
    dataframe = pd.DataFrame(
        {
            "product_id": [1, 2],
            "label": ["Exact", "Irrelevant"],
        }
    )

    with pytest.raises(SplitError):
        split_judgments(dataframe)


def test_missing_query_assignment_raises() -> None:
    judgments = pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "product_id": [10, 20, 30],
        }
    )

    split_ids = {
        "train": pd.Series([1]),
        "validation": pd.Series([2]),
        "test": pd.Series([]),
    }

    with pytest.raises(SplitError):
        assign_split(judgments, split_ids)


def test_overlap_detection_raises() -> None:
    split_ids = {
        "train": pd.Series([1, 2]),
        "validation": pd.Series([2, 3]),
        "test": pd.Series([4]),
    }

    with pytest.raises(SplitError):
        verify_query_disjointness(split_ids)
