import pandas as pd
import pytest

from relevanceflow.retrieval.tfidf import (
    TfidfConfig,
    TfidfRetrievalError,
    TfidfRetriever,
)


@pytest.fixture
def products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [1, 2, 3],
            "product_name": [
                "black leather sofa",
                "red cotton shirt",
                "black leather chair",
            ],
        }
    )


def test_tfidf_fit(products: pd.DataFrame) -> None:
    retriever = TfidfRetriever()

    retriever.fit(
        products["product_id"],
        products["product_name"],
    )

    assert retriever.is_fitted
    assert retriever.product_matrix is not None
    assert retriever.product_ids is not None


def test_tfidf_ranking(products: pd.DataFrame) -> None:
    retriever = TfidfRetriever(
        TfidfConfig(
            ngram_range=(1, 2),
            max_features=None,
        )
    )

    retriever.fit(
        products["product_id"],
        products["product_name"],
    )

    ranked = retriever.rank_candidates(
        query_text="black leather sofa",
        candidate_products=products,
        top_k=3,
    )

    assert ranked.iloc[0]["product_id"] == 1
    assert list(ranked["rank"]) == [1, 2, 3]


def test_tfidf_requires_fit(
    products: pd.DataFrame,
) -> None:
    retriever = TfidfRetriever()

    with pytest.raises(TfidfRetrievalError):
        retriever.rank_candidates(
            "black sofa",
            products,
        )


def test_empty_candidates(
    products: pd.DataFrame,
) -> None:
    retriever = TfidfRetriever()

    retriever.fit(
        products["product_id"],
        products["product_name"],
    )

    empty = products.iloc[0:0]

    result = retriever.rank_candidates(
        "black sofa",
        empty,
    )

    assert result.empty


def test_missing_product_id(
    products: pd.DataFrame,
) -> None:
    retriever = TfidfRetriever()

    retriever.fit(
        products["product_id"],
        products["product_name"],
    )

    invalid = products.drop(columns=["product_id"])

    with pytest.raises(TfidfRetrievalError):
        retriever.rank_candidates(
            "black sofa",
            invalid,
        )
