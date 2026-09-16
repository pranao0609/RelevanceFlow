from relevanceflow.features.metadata import (
    metadata_features,
)


def test_metadata_features():
    features = metadata_features(
        average_rating=4.5,
        rating_count=100,
        review_count=50,
    )

    assert features["average_rating"] == 4.5
    assert features["rating_count"] == 100.0
    assert features["review_count"] == 50.0


def test_invalid_metadata_defaults_to_zero():
    features = metadata_features(
        average_rating=None,
        rating_count="invalid",
        review_count=None,
    )

    assert features["average_rating"] == 0.0
    assert features["rating_count"] == 0.0
    assert features["review_count"] == 0.0
