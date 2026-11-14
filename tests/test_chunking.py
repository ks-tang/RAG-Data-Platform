import pytest

from chunking import (build_document_text, chunk_document, ends_like_a_sentence,
                   split_by_sentences, split_fixed_size, split_sentences)

PHRASES = [f"Ceci est la phrase numéro {i} du document de test." for i in range(1, 41)]
TEXTE = " ".join(PHRASES)


# ---------- taille fixe ----------
def test_fixed_respecte_la_taille_maximale():
    for start, end in split_fixed_size(TEXTE, size=200, overlap=30):
        assert end - start <= 200


def test_fixed_couvre_tout_le_texte():
    spans = split_fixed_size(TEXTE, size=200, overlap=30)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(TEXTE)
    # aucun trou entre deux morceaux consécutifs
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start <= end


def test_fixed_chevauchement_present():
    spans = split_fixed_size(TEXTE, size=200, overlap=30)
    assert all(next_start < end for (_, end), (next_start, _) in zip(spans, spans[1:]))


def test_fixed_ne_coupe_pas_les_mots():
    mots = set(TEXTE.split())
    for start, end in split_fixed_size(TEXTE, size=200, overlap=30):
        assert set(TEXTE[start:end].split()) <= mots


def test_fixed_mot_tres_long_ne_bloque_pas():
    spans = split_fixed_size("x" * 1000, size=100, overlap=10)
    assert all(end - start <= 100 for start, end in spans)
    assert spans[-1][1] == 1000


def test_fixed_parametres_invalides():
    with pytest.raises(ValueError):
        split_fixed_size("abc", size=10, overlap=10)
    with pytest.raises(ValueError):
        split_fixed_size("abc", size=0, overlap=0)


# ---------- phrases ----------
def test_decoupe_en_phrases():
    spans = split_sentences("Première phrase. Deuxième phrase ! Troisième ?\n\nNouveau paragraphe.")
    textes = ["Première phrase. Deuxième phrase ! Troisième ?\n\nNouveau paragraphe."[s:e] for s, e in spans]
    assert textes == ["Première phrase.", "Deuxième phrase !", "Troisième ?", "Nouveau paragraphe."]


def test_liste_a_puces_separee():
    texte = "Principes :\n- confidentialité\n- intégrité"
    assert [texte[s:e] for s, e in split_sentences(texte)] == ["Principes :", "- confidentialité", "- intégrité"]


def test_sentences_taille_maximale_et_fin_de_phrase():
    spans = split_by_sentences(TEXTE, size=200, overlap_sentences=1)
    assert all(end - start <= 200 for start, end in spans)
    assert all(TEXTE[start:end].endswith(".") for start, end in spans)


def test_sentences_chevauchement_d_une_phrase():
    spans = split_by_sentences(TEXTE, size=200, overlap_sentences=1)
    assert all(next_start < end for (_, end), (next_start, _) in zip(spans, spans[1:]))


def test_sentences_sans_chevauchement():
    spans = split_by_sentences(TEXTE, size=200, overlap_sentences=0)
    assert all(next_start >= end for (_, end), (next_start, _) in zip(spans, spans[1:]))


def test_sentences_aucun_morceau_en_double():
    spans = split_by_sentences(TEXTE, size=100, overlap_sentences=1)
    assert len(spans) == len(set(spans))


def test_sentences_phrase_geante_decoupee():
    geante = "mot " * 500  # aucune ponctuation : une seule « phrase » de 2000 caractères
    spans = split_by_sentences(geante, size=300)
    assert all(end - start <= 300 for start, end in spans)


# ---------- pages et API ----------
def test_assemblage_des_pages():
    texte, starts, numbers = build_document_text([(1, "Fin de phrase."), (2, ""), (3, "Suite sans fin"), (4, "ok")])
    assert numbers == [1, 3, 4]
    assert texte == "Fin de phrase.\n\nSuite sans fin ok"
    assert texte[starts[1]:].startswith("Suite")


def test_pages_retrouvees():
    pages = [(5, "Alpha beta gamma. " * 30), (6, "Delta epsilon zeta. " * 30)]
    morceaux = chunk_document(pages, "sentences", size=200)
    assert morceaux[0].page_start == 5
    assert morceaux[-1].page_end == 6
    assert all(m.page_start <= m.page_end for m in morceaux)


def test_chunk_texte_coherent_avec_positions():
    pages = [(1, TEXTE)]
    texte, _, _ = build_document_text(pages)
    for strategie in ("fixed", "sentences"):
        for m in chunk_document(pages, strategie, size=200):
            assert m.text == texte[m.start:m.end]


def test_document_vide():
    assert chunk_document([], "fixed") == []
    assert chunk_document([(1, "   ")], "sentences") == []


def test_strategie_inconnue():
    with pytest.raises(ValueError):
        chunk_document([(1, "texte")], "magique")


def test_ends_like_a_sentence():
    assert ends_like_a_sentence("Une phrase complète.")
    assert not ends_like_a_sentence("Une phrase coupée au mil")
