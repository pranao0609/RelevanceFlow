from relevanceflow.cache import build_ranking_cache_key

PRODUCTS = [
    {
        "product_id": 2,
        "product_name": "B",
    },
    {
        "product_id": 1,
        "product_name": "A",
    },
]


def test_cache_key_is_deterministic():
    key1 = build_ranking_cache_key(
        model_name="Model",
        model_alias="champion",
        query="headphones",
        products=PRODUCTS,
        top_k=2,
    )

    key2 = build_ranking_cache_key(
        model_name="Model",
        model_alias="champion",
        query="headphones",
        products=list(reversed(PRODUCTS)),
        top_k=2,
    )

    assert key1 == key2


def test_query_changes_key():
    key1 = build_ranking_cache_key(
        model_name="Model",
        model_alias="champion",
        query="mouse",
        products=PRODUCTS,
        top_k=2,
    )

    key2 = build_ranking_cache_key(
        model_name="Model",
        model_alias="champion",
        query="keyboard",
        products=PRODUCTS,
        top_k=2,
    )

    assert key1 != key2


def test_model_alias_changes_key():
    key1 = build_ranking_cache_key(
        model_name="Model",
        model_alias="champion",
        query="mouse",
        products=PRODUCTS,
        top_k=2,
    )

    key2 = build_ranking_cache_key(
        model_name="Model",
        model_alias="candidate",
        query="mouse",
        products=PRODUCTS,
        top_k=2,
    )

    assert key1 != key2
