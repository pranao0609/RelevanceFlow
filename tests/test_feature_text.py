from relevanceflow.features.text import (
    category_depth,
    description_length,
    feature_count,
    title_length,
)


def test_title_length():
    assert title_length("Red Wooden Chair") == 3.0


def test_description_length():
    assert description_length("This is a comfortable wooden chair.") == 6.0


def test_feature_count():
    assert feature_count("wood\ncomfortable\nindoor") == 3.0


def test_category_depth():
    assert category_depth("Furniture > Chairs > Dining Chairs") == 3.0
