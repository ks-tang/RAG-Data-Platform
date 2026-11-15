import numpy as np

from diagnose import count_word_by_document, global_ranks


def test_rangs_generaux():
    scores = np.array([0.1, 0.9, 0.5])
    assert list(global_ranks(scores)) == [3, 1, 2]


def test_rangs_generaux_un_seul_morceau():
    assert list(global_ranks(np.array([0.3]))) == [1]


def _row(filename, text):
    # même structure qu'une ligne de fetch_candidates : id, fichier, page début, page fin, texte, vecteur
    return ("id", filename, 1, 1, text, [])


def test_comptage_du_mot_par_document():
    rows = [_row("a.pdf", "L'IA agentique agit seule"), _row("a.pdf", "autre sujet"),
            _row("b.pdf", "Agentique et risques"), _row("b.pdf", "AGENTIQUE")]
    counts = count_word_by_document(rows, "agentique")
    assert counts["a.pdf"] == 1 and counts["b.pdf"] == 2
