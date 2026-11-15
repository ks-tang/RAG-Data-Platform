"""Trois façons de chercher dans les morceaux : par le sens, par les mots, ou les deux.

    "semantic" : on compare les embeddings (le SENS). Bon pour les reformulations, mais peut
                 passer à côté d'un mot rare et précis (par exemple « agentique »).
    "keyword"  : on compte les mots communs avec la question (algorithme BM25). Excellent pour
                 un terme précis, mais ne comprend pas les synonymes.
    "hybrid"   : on fusionne les deux classements (méthode « Reciprocal Rank Fusion »).
    "rerank"   : on prend les meilleurs candidats de l'hybride, puis un second modèle (un « reranker »)
                 relit chaque couple (question, morceau) et les remet dans le bon ordre.
"""
import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from embedding import top_matches

FUSION_DEPTH = 100  # en mode hybride, on fusionne les 100 premiers de chaque classement
RRF_K = 60          # constante classique de la fusion (réduit le poids des tout premiers rangs)

# Reranker : un « cross-encoder » lit la question ET le morceau ensemble, ce qui est plus précis
# (mais plus lent) que de comparer deux vecteurs calculés séparément. On ne l'applique donc qu'à
# un petit nombre de candidats, présélectionnés par la recherche hybride.
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
RERANK_DEPTH = 20  # nombre de candidats de l'hybride que le reranker réordonne
RERANK_BATCH = 4   # nombre de morceaux envoyés d'un coup au reranker (petit nombre = moins de mémoire)

# Mots très fréquents qui n'aident pas à distinguer un morceau d'un autre.
STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux", "et", "ou", "en", "dans", "par",
    "pour", "sur", "avec", "sans", "ce", "cet", "cette", "ces", "son", "sa", "ses", "leur", "leurs",
    "qui", "que", "quoi", "dont", "est", "sont", "etre", "ont", "il", "elle", "ils", "elles", "on",
    "nous", "vous", "je", "tu", "ne", "pas", "plus", "se", "comment", "quel", "quels", "quelle",
    "quelles", "quelque", "quelques", "etc", "ainsi", "afin", "cela", "ceci", "donc", "mais",
}


def tokenize(text: str):
    """Découpe un texte en mots normalisés : minuscules, sans accents, sans mots vides, au singulier.

    La normalisation sert à ce que « Systèmes », « systemes » et « système » comptent pour le même mot.
    Le passage au singulier est volontairement rudimentaire (on retire un « s » ou « x » final) :
    un vrai « stemmer » français ferait mieux, mais cela suffit pour une première version.
    """
    text = text.casefold().replace("œ", "oe").replace("æ", "ae")
    text = unicodedata.normalize("NFD", text)  # sépare les lettres de leurs accents
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # supprime les accents
    tokens = []
    for token in re.findall(r"[a-z0-9]+", text):
        if token in STOPWORDS:
            continue
        if len(token) > 3 and token[-1] in "sx":
            token = token[:-1]
        if len(token) >= 2:
            tokens.append(token)
    return tokens


class BM25Index:
    """Index de mots pour la recherche par mots-clés (algorithme BM25).

    Le score d'un morceau pour une question additionne, pour chaque mot de la question :
        - un poids élevé si le mot est RARE dans le corpus (« agentique » compte plus que « sécurité ») ;
        - qui augmente avec le nombre d'occurrences dans le morceau, mais de moins en moins vite ;
        - et qui tient compte de la longueur du morceau (un mot dans un texte court compte plus).
    """

    def __init__(self, texts, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        tokenized = [tokenize(text) for text in texts]
        self.size = len(tokenized)
        self.lengths = np.array([len(tokens) for tokens in tokenized], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if self.size else 1.0
        if self.avg_length == 0:
            self.avg_length = 1.0

        # Index inversé : pour chaque mot, la liste des morceaux qui le contiennent et son nombre d'occurrences.
        postings = defaultdict(list)
        for position, tokens in enumerate(tokenized):
            for term, frequency in Counter(tokens).items():
                postings[term].append((position, frequency))
        self.postings = {
            term: (np.array([p for p, _ in pairs]), np.array([f for _, f in pairs], dtype=np.float32))
            for term, pairs in postings.items()
        }

    def scores(self, query: str):
        """Score BM25 de chaque morceau pour la question (0 si aucun mot en commun)."""
        scores = np.zeros(self.size, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            positions, frequencies = self.postings[term]
            containing = len(positions)  # nombre de morceaux qui contiennent ce mot
            idf = math.log(1 + (self.size - containing + 0.5) / (containing + 0.5))  # rareté du mot
            length_factor = 1 - self.b + self.b * self.lengths[positions] / self.avg_length
            scores[positions] += idf * frequencies * (self.k1 + 1) / (frequencies + self.k1 * length_factor)
        return scores


def top_from_scores(scores, k: int):
    """Les k meilleurs morceaux d'après leurs scores, en ignorant ceux dont le score est nul."""
    best = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in best if scores[i] > 0]


def reciprocal_rank_fusion(rankings, k: int = RRF_K):
    """Fusionne plusieurs classements en un seul.

    Chaque classement est une liste de positions, du meilleur au moins bon. Un morceau reçoit
    1/(k + rang) dans chaque classement où il figure, et ces points s'additionnent : un morceau bien
    classé dans les DEUX listes passe devant un morceau excellent dans une seule.
    Renvoie une liste de (position, score), du meilleur au moins bon.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            fused[position] += 1 / (k + rank)
    return sorted(fused.items(), key=lambda pair: -pair[1])


def load_reranker():
    """Charge le reranker (le premier appel télécharge le modèle, environ 1,1 Go)."""
    # Import à l'intérieur de la fonction : le reste du fichier reste utilisable sans la bibliothèque.
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=RERANK_MODEL)


def rerank(reranker, question: str, ranked, top_k: int):
    """Réordonne des résultats avec le reranker.

    `ranked` : liste de (score, ligne_candidate) issue d'une première recherche.
    Renvoie les top_k meilleurs selon le reranker, avec SES scores (qui ne se comparent
    pas aux scores de la recherche initiale : ce ne sont pas les mêmes unités).
    """
    if not ranked:
        return []
    texts = [row[4] for _, row in ranked]
    # Chaque couple (question, morceau) est noté indépendamment : on peut donc les traiter par petits
    # paquets sans changer le résultat, ce qui limite la mémoire utilisée à chaque appel.
    scores = []
    for start in range(0, len(texts), RERANK_BATCH):
        scores.extend(reranker.rerank(question, texts[start:start + RERANK_BATCH]))
    scores = np.array(scores, dtype=np.float32)
    best = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), ranked[i][1]) for i in best]


class Retriever:
    """Regroupe les morceaux et leurs index, et répond aux questions selon le mode choisi."""

    MODES = ("semantic", "keyword", "hybrid", "rerank")

    def __init__(self, candidates, reranker=None):
        """`candidates` : lignes de search.fetch_candidates (texte en position 4, vecteur en position 5).
        `reranker` : nécessaire uniquement pour le mode « rerank »."""
        self.candidates = candidates
        self.reranker = reranker
        self.matrix = np.array([row[5] for row in candidates], dtype=np.float32)
        self.bm25 = BM25Index([row[4] for row in candidates])

    def search(self, question: str, query_vector, mode: str = "semantic", top_k: int = 5):
        """Renvoie [(score, ligne_candidate), ...]. `query_vector` peut être None en mode « keyword »."""
        if mode == "semantic":
            ranked = top_matches(self.matrix, query_vector, top_k)
        elif mode == "keyword":
            ranked = top_from_scores(self.bm25.scores(question), top_k)
        elif mode in ("hybrid", "rerank"):
            semantic = [i for i, _ in top_matches(self.matrix, query_vector, FUSION_DEPTH)]
            keyword = [i for i, _ in top_from_scores(self.bm25.scores(question), FUSION_DEPTH)]
            fused = reciprocal_rank_fusion([semantic, keyword])
            if mode == "hybrid":
                ranked = fused[:top_k]
            else:
                if self.reranker is None:
                    raise ValueError("Le mode « rerank » nécessite un reranker (voir load_reranker).")
                candidates = [(score, self.candidates[i]) for i, score in fused[:RERANK_DEPTH]]
                return rerank(self.reranker, question, candidates, top_k)
        else:
            raise ValueError(f"Mode inconnu : {mode!r}. Choix : {self.MODES}")
        return [(score, self.candidates[i]) for i, score in ranked]
