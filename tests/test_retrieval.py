import numpy as np
import pytest

from retrieval import BM25Index, Retriever, reciprocal_rank_fusion, rerank, tokenize, top_from_scores


# ---------- tokenisation ----------
def test_tokenize_accents_majuscules_et_ponctuation():
    assert tokenize("Systèmes d'Information, CRITIQUES !") == ["systeme", "information", "critique"]


def test_tokenize_retire_les_mots_vides():
    assert tokenize("Quels sont les risques de l'IA agentique ?") == ["risque", "ia", "agentique"]


def test_tokenize_pluriels_et_ligatures():
    assert tokenize("mesures industriels mise en œuvre") == ["mesure", "industriel", "mise", "oeuvre"]


def test_tokenize_texte_vide():
    assert tokenize("") == []


# ---------- BM25 ----------
TEXTS = [
    "la sécurité des systèmes industriels et les mesures de protection",
    "les agents agentiques agissent de façon autonome sur le système",
    "la sauvegarde des données et la restauration",
    "la sécurité du réseau wifi et la sécurité des accès",
]


def test_bm25_retrouve_le_morceau_contenant_le_mot_rare():
    scores = BM25Index(TEXTS).scores("risques agentique")
    assert int(np.argmax(scores)) == 1


def test_bm25_mot_inconnu_donne_zero():
    assert BM25Index(TEXTS).scores("blockchain").sum() == 0


def test_bm25_mot_rare_pese_plus_que_mot_courant():
    index = BM25Index(TEXTS)
    assert index.scores("agentique")[1] > index.scores("securite")[0]


def test_bm25_sans_texte():
    assert len(BM25Index([]).scores("test")) == 0


def test_top_from_scores_ignore_les_scores_nuls():
    result = top_from_scores(np.array([0.0, 2.0, 0.0, 1.0]), 4)
    assert [i for i, _ in result] == [1, 3]


# ---------- fusion ----------
def test_fusion_favorise_ce_qui_est_dans_les_deux_listes():
    fused = reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])
    assert fused[0][0] == 2  # 2 est 2e dans les deux listes : plus de points que 1 ou 4 (1er d'une seule)


def test_fusion_classement_unique():
    assert [i for i, _ in reciprocal_rank_fusion([[7, 8, 9]])] == [7, 8, 9]


# ---------- Retriever ----------
def _candidates():
    vectors = np.eye(4, dtype=np.float32)
    return [(f"c{i}", "doc.pdf", 1, 1, TEXTS[i], list(vectors[i])) for i in range(4)]


def test_retriever_modes():
    retriever = Retriever(_candidates())
    query = np.array([0, 0, 1, 0], dtype=np.float32)  # le vecteur du morceau 2 (sauvegarde)
    assert retriever.search("agentique", query, "semantic", 1)[0][1][0] == "c2"
    assert retriever.search("agentique", query, "keyword", 1)[0][1][0] == "c1"
    hybrid_ids = [row[0] for _, row in retriever.search("agentique", query, "hybrid", 2)]
    assert set(hybrid_ids) == {"c1", "c2"}  # un morceau de chaque classement


def test_retriever_keyword_sans_vecteur():
    retriever = Retriever(_candidates())
    assert retriever.search("agentique", None, "keyword", 1)[0][1][0] == "c1"


def test_retriever_mode_inconnu():
    with pytest.raises(ValueError):
        Retriever(_candidates()).search("x", None, "magique", 1)


# ---------- reranker (simulé : un vrai modèle serait trop lourd pour un test) ----------
class FakeReranker:
    """Donne le score 1 aux textes qui contiennent le mot cherché, 0 aux autres."""

    def __init__(self, word):
        self.word = word

    def rerank(self, query, documents):
        return [1.0 if self.word in document else 0.0 for document in documents]


def test_rerank_remet_dans_le_bon_ordre():
    candidates = _candidates()
    ranked = [(0.9, candidates[0]), (0.8, candidates[2]), (0.7, candidates[1])]  # c1 est dernier
    result = rerank(FakeReranker("agentiques"), "question", ranked, 2)
    assert result[0][1][0] == "c1"
    assert len(result) == 2


def test_rerank_sans_resultat():
    assert rerank(FakeReranker("x"), "question", [], 5) == []


def test_retriever_mode_rerank():
    retriever = Retriever(_candidates(), reranker=FakeReranker("sauvegarde"))
    query = np.array([1, 0, 0, 0], dtype=np.float32)  # le vecteur le plus proche est celui du morceau 0
    result = retriever.search("agentique sauvegarde", query, "rerank", 1)
    assert result[0][1][0] == "c2"  # le reranker a le dernier mot : c'est le morceau qui parle de sauvegarde


def test_retriever_rerank_sans_reranker():
    with pytest.raises(ValueError):
        Retriever(_candidates()).search("x", np.array([1, 0, 0, 0], dtype=np.float32), "rerank", 1)


def test_rerank_par_paquets_conserve_l_ordre():
    # plus de morceaux que la taille d'un paquet : le résultat doit être le même qu'en un seul appel
    vectors = np.eye(12, dtype=np.float32)
    texts = [f"morceau {i}" + (" cible" if i == 9 else "") for i in range(12)]
    candidates = [(f"c{i}", "doc.pdf", 1, 1, texts[i], list(vectors[i])) for i in range(12)]
    ranked = [(1.0 - i / 100, candidates[i]) for i in range(12)]
    result = rerank(FakeReranker("cible"), "question", ranked, 3)
    assert result[0][1][0] == "c9"
    assert len(result) == 3
