import numpy as np

from search import format_pages, search


def _candidate(chunk_id, vector):
    # même structure qu'une ligne de fetch_candidates : id, fichier, page début, page fin, texte, vecteur
    return (chunk_id, "doc.pdf", 1, 1, f"texte {chunk_id}", list(vector))


def test_search_retrouve_le_bon_morceau():
    candidates = [_candidate("a", [1, 0, 0]), _candidate("b", [0, 1, 0]), _candidate("c", [0, 0, 1])]
    results = search(candidates, np.array([0, 1, 0], dtype=np.float32), top_k=2)
    assert results[0][1][0] == "b"
    assert len(results) == 2


def test_format_pages():
    assert format_pages(5, 5) == "p. 5"
    assert format_pages(5, 6) == "p. 5-6"
