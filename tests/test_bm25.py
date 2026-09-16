import pytest

from relevanceflow.retrieval.bm25 import (
    BM25Config,
    BM25RetrievalError,
    BM25Retriever,
)


@pytest.fixture
def retriever():
    documents = [
        "red wooden dining table",
        "blue office chair",
        "wooden dining chair",
        "large dining table",
    ]

    return BM25Retriever(
        BM25Config(
            k1=1.5,
            b=0.75,
        )
    ).fit(documents)


def test_bm25_fit(retriever):
    assert retriever.is_fitted_
    assert retriever.num_documents_ == 4
    assert retriever.avg_doc_length_ > 0
    assert len(retriever.vocabulary_) > 0


def test_bm25_ranking(retriever):
    candidates = [
        {
            "product_id": "1",
            "product_name": "red wooden dining table",
        },
        {
            "product_id": "2",
            "product_name": "blue office chair",
        },
        {
            "product_id": "3",
            "product_name": "wooden dining chair",
        },
    ]

    results = retriever.rank_candidates(
        query="wooden dining table",
        candidate_products=candidates,
        top_k=3,
    )

    assert len(results) == 3
    assert results[0]["product_id"] == "1"
    assert results[0]["rank"] == 1


def test_top_k(retriever):
    candidates = [
        {
            "product_id": str(index),
            "product_name": "wooden dining table",
        }
        for index in range(10)
    ]

    results = retriever.rank_candidates(
        query="wooden table",
        candidate_products=candidates,
        top_k=5,
    )

    assert len(results) == 5


def test_invalid_k1():
    with pytest.raises(BM25RetrievalError):
        BM25Retriever(BM25Config(k1=-1))


def test_invalid_b():
    with pytest.raises(BM25RetrievalError):
        BM25Retriever(BM25Config(b=1.5))


def test_empty_corpus():
    retriever = BM25Retriever()

    with pytest.raises(BM25RetrievalError):
        retriever.fit([])


def test_rank_before_fit():
    retriever = BM25Retriever()

    candidates = [
        {
            "product_id": "1",
            "product_name": "wooden table",
        }
    ]

    with pytest.raises(BM25RetrievalError):
        retriever.rank_candidates(
            query="table",
            candidate_products=candidates,
            top_k=1,
        )
