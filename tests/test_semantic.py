import numpy as np

from relevanceflow.features.semantic import (
    SemanticEncoder,
    SemanticEncoderConfig,
)


def test_cosine_similarity_identical_vectors():
    vector = np.array([1.0, 2.0, 3.0])

    similarity = SemanticEncoder.cosine_similarity(
        vector,
        vector,
    )

    assert np.isclose(similarity, 1.0)


def test_cosine_similarity_orthogonal_vectors():
    vector_a = np.array([1.0, 0.0])
    vector_b = np.array([0.0, 1.0])

    similarity = SemanticEncoder.cosine_similarity(
        vector_a,
        vector_b,
    )

    assert np.isclose(similarity, 0.0)


def test_cosine_similarity_zero_vector():
    vector_a = np.array([0.0, 0.0])
    vector_b = np.array([1.0, 0.0])

    similarity = SemanticEncoder.cosine_similarity(
        vector_a,
        vector_b,
    )

    assert similarity == 0.0


def test_default_config():
    config = SemanticEncoderConfig()

    assert config.model_name == ("sentence-transformers/all-MiniLM-L6-v2")
    assert config.normalize_embeddings is True
    assert config.batch_size == 32
