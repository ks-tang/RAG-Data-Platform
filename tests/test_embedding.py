import numpy as np

from embedding import text_hash, top_matches


def test_empreinte_stable_et_sensible():
    assert text_hash("abc") == text_hash("abc")
    assert text_hash("abc") != text_hash("abd")
    assert len(text_hash("abc")) == 64


def test_top_matches_classe_du_plus_proche_au_moins_proche():
    matrix = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    query = np.array([0.6, 0.8], dtype=np.float32)
    result = top_matches(matrix, query, 3)
    assert [i for i, _ in result] == [2, 1, 0]
    assert result[0][1] > result[1][1] > result[2][1]
    assert abs(result[0][1] - 1.0) < 1e-6


def test_top_matches_k_plus_grand_que_le_nombre_de_vecteurs():
    matrix = np.eye(3, dtype=np.float32)
    assert len(top_matches(matrix, np.array([1, 0, 0], dtype=np.float32), 10)) == 3


def test_top_matches_k_limite_les_resultats():
    matrix = np.eye(5, dtype=np.float32)
    assert len(top_matches(matrix, np.array([0, 0, 1, 0, 0], dtype=np.float32), 2)) == 2
