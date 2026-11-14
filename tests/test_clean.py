from clean import clean_text, remove_repeated_lines


def test_mot_coupe_en_fin_de_ligne():
    assert clean_text("La sécu-\nrité des systèmes") == "La sécurité des systèmes"


def test_tiret_legitime_conserve():
    # tiret suivi d'un chiffre ou d'une majuscule : on garde le tiret, on supprime seulement le saut de ligne
    assert clean_text("ISO-\n27001") == "ISO-27001"


def test_espaces_multiples():
    assert clean_text("un   texte \t avec   espaces") == "un texte avec espaces"


def test_saut_de_ligne_dans_phrase():
    assert clean_text("Cette phrase est\ncoupée au milieu.") == "Cette phrase est coupée au milieu."


def test_paragraphes_conserves():
    assert clean_text("Premier paragraphe.\n\n\n\nSecond paragraphe.") == "Premier paragraphe.\n\nSecond paragraphe."


def test_puces_converties_en_liste():
    out = clean_text("Liste :\n\uf06e Premier point\n■ Second point")
    assert out == "Liste :\n- Premier point\n- Second point"


def test_caracteres_parasites():
    assert clean_text("a\x00b\u00a0c\u00adc") == "ab cc"


def test_texte_vide():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_entetes_et_pieds_de_page_retires():
    sujets = ["chiffrement", "pare-feu", "journalisation", "sauvegarde", "authentification",
              "segmentation", "supervision", "gestion des correctifs", "cloisonnement", "audit"]
    pages = [f"Guide ANSSI - Version 1.0\nCette page traite de {s}.\nPage {i}" for i, s in enumerate(sujets, 1)]
    out = remove_repeated_lines(pages)
    assert all("Guide ANSSI" not in p for p in out)
    assert all(f"Page {i}" != p.splitlines()[-1] for i, p in enumerate(out, 1) if p.splitlines())
    assert "Cette page traite de journalisation." in out[2]


def test_document_court_inchange():
    pages = ["En-tête\ntexte 1", "En-tête\ntexte 2"]
    assert remove_repeated_lines(pages) == pages
