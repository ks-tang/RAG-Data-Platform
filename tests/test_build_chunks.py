from build_chunks import (build_rows, chunk_options, default_overlap, make_chunk_id,
                          make_config, text_fingerprint)
from chunking import chunk_document

PAGES = [(1, "Première phrase du document. Deuxième phrase du document. " * 10),
         (2, "Troisième page avec du texte. " * 10)]


def test_config_lisible():
    assert make_config("sentences", 800, 1) == "sentences-800-1"


def test_chevauchement_par_defaut_selon_la_strategie():
    assert default_overlap("fixed") > default_overlap("sentences")  # caractères vs phrases


def test_options_selon_la_strategie():
    assert chunk_options("fixed", 500, 50) == {"size": 500, "overlap": 50}
    assert chunk_options("sentences", 500, 1) == {"size": 500, "overlap_sentences": 1}


def test_identifiant_stable_et_distinct():
    a = make_chunk_id("abcdef0123456789ffff", "sentences-800-1", 3)
    assert a == make_chunk_id("abcdef0123456789ffff", "sentences-800-1", 3)
    assert a != make_chunk_id("abcdef0123456789ffff", "sentences-800-1", 4)
    assert a != make_chunk_id("abcdef0123456789ffff", "fixed-800-100", 3)
    assert a == "abcdef0123456789-sentences-800-1-0003"


def test_empreinte_change_avec_le_texte():
    assert text_fingerprint("abc") == text_fingerprint("abc")
    assert text_fingerprint("abc") != text_fingerprint("abd")


def test_lignes_pretes_a_inserer():
    chunks = chunk_document(PAGES, "sentences", size=300, overlap_sentences=1)
    rows = build_rows("d" * 64, chunks, "sentences", "sentences-300-1", "hash", "2026-01-01")
    assert len(rows) == len(chunks) > 1
    assert all(len(r) == 11 for r in rows)  # une valeur par colonne de la table
    assert len({r[0] for r in rows}) == len(rows)  # chunk_id tous différents
    assert rows[0][2] == 0 and rows[0][6] == len(rows[0][5])  # chunk_index, n_chars
