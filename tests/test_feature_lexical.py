from relevanceflow.features.lexical import (
    character_overlap,
    exact_match,
    lexical_features,
    query_title_overlap,
    token_overlap,
)


def test_exact_match():
    assert (
        exact_match(
            "Red Chair",
            "red chair",
        )
        == 1.0
    )


def test_exact_match_false():
    assert (
        exact_match(
            "red chair",
            "blue chair",
        )
        == 0.0
    )


def test_token_overlap():
    value = token_overlap(
        "red wooden chair",
        "wooden chair",
    )

    assert 0.0 < value < 1.0


def test_character_overlap():
    value = character_overlap(
        "chair",
        "chair",
    )

    assert value == 1.0


def test_query_title_overlap():
    assert (
        query_title_overlap(
            "red wooden chair",
            "wooden chair",
        )
        == 2.0
    )


def test_lexical_features():
    features = lexical_features(
        query="red chair",
        product_name="red chair",
        bm25_score=3.5,
        tfidf_similarity=0.8,
    )

    assert features["bm25_score"] == 3.5
    assert features["tfidf_similarity"] == 0.8
    assert features["exact_match"] == 1.0
