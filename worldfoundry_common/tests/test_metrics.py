import numpy as np

from worldfoundry_common.metrics import cosine_similarity


def test_cosine_similarity_identical_is_one():
    a = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_2d_arrays():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0]])
    sims = cosine_similarity(a, b)
    assert abs(sims[0] - 1.0) < 1e-6
    assert abs(sims[1]) < 1e-6