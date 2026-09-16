from relevanceflow.features.compatibility import (
    category_match,
    compatibility_features,
    phrase_match,
    product_class_match,
)


def test_category_match():
    value = category_match(
        "dining chair",
        "Furniture > Dining Chairs",
    )

    assert value > 0.0


def test_product_class_match():
    assert (
        product_class_match(
            "chair",
            "Chair",
        )
        == 1.0
    )


def test_phrase_match():
    assert (
        phrase_match(
            "wooden chair",
            "Modern Wooden Chair",
        )
        == 1.0
    )


def test_compatibility_features():
    features = compatibility_features(
        query="wooden chair",
        product_name="Modern Wooden Chair",
        category_hierarchy="Furniture > Chairs",
        product_class="Chair",
    )

    assert "category_match" in features
    assert "product_class_match" in features
    assert "phrase_match" in features
