from evaluate import first_relevant_rank, is_relevant, summarize


def _result(filename, text="texte"):
    # même structure qu'un résultat de recherche : (score, (id, fichier, page_début, page_fin, texte, vecteur))
    return (0.9, ("id", filename, 1, 1, text, []))


def test_pertinent_par_document():
    assert is_relevant("anssi-fondamentaux-zero-trust-v1.0.pdf", "x", {"docs": ["zero-trust"]})
    assert not is_relevant("anssi_essentiels_wifi_1.0.pdf", "x", {"docs": ["zero-trust"]})


def test_pertinent_sans_tenir_compte_des_majuscules():
    assert is_relevant("Fiche_Durcissement.pdf", "x", {"docs": ["durcissement"]})


def test_plusieurs_documents_attendus():
    expected = {"docs": ["identites", "administration"]}
    assert is_relevant("Fiche_Administration.pdf", "x", expected)
    assert is_relevant("Fiche_Gestion-des-identites.pdf", "x", expected)


def test_mot_cle_obligatoire():
    expected = {"docs": ["zero-trust"], "keywords": ["confiance implicite"]}
    assert is_relevant("zero-trust.pdf", "réduire la Confiance Implicite", expected)
    assert not is_relevant("zero-trust.pdf", "un autre sujet", expected)


def test_rang_du_premier_resultat_pertinent():
    results = [_result("a.pdf"), _result("b.pdf"), _result("cible.pdf")]
    assert first_relevant_rank(results, {"docs": ["cible"]}) == 3
    assert first_relevant_rank(results, {"docs": ["a.pdf"]}) == 1


def test_aucun_resultat_pertinent():
    assert first_relevant_rank([_result("a.pdf")], {"docs": ["cible"]}) is None


def test_resume_hit_et_mrr():
    summary = summarize([1, 2, None, 5])  # 4 questions
    assert summary["hit@1"] == 0.25
    assert summary["hit@3"] == 0.5
    assert summary["hit@5"] == 0.75
    assert abs(summary["MRR"] - (1 + 0.5 + 0 + 0.2) / 4) < 1e-9
