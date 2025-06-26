import numpy as np
import pytest

from pyturbseq.guides import hamming_dist, hamming_dist_matrix


def test_hamming_dist_basic():
    """Distance between sequences of equal length."""
    assert hamming_dist("AAAA", "AAAT") == 0.25
    assert hamming_dist("A", "T") == 1.0


def test_hamming_dist_same_strings_zero():
    """Identical strings should have zero distance."""
    assert hamming_dist("ABCD", "ABCD") == 0.0


def test_hamming_dist_invalid_length_raises():
    """Different length strings should raise ValueError."""
    with pytest.raises(ValueError):
        hamming_dist("ABC", "AB")


def test_hamming_dist_matrix_values():
    """Matrix should contain pairwise normalized Hamming distances."""
    seqs = ["AAAA", "AAAT", "AATT"]
    matrix = hamming_dist_matrix(seqs)

    expected = np.array([
        [0.0, 0.25, 0.5],
        [0.25, 0.0, 0.25],
        [0.5, 0.25, 0.0],
    ])
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == expected.shape
    assert np.allclose(matrix, expected)


def test_hamming_dist_matrix_with_tuple_input():
    """Any iterable (e.g., tuple) should be accepted."""
    seqs = ("GGGG", "GGGA")
    matrix = hamming_dist_matrix(seqs)
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == pytest.approx(0.25)
    assert matrix[1, 0] == pytest.approx(0.25)
    assert np.allclose(np.diag(matrix), 0)
