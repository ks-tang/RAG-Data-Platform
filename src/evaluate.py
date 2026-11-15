"""Mesure la qualité de la recherche sémantique sur une liste de questions de référence.

Pourquoi ? Pour améliorer un système, il faut d'abord pouvoir le MESURER. Avec ce script, on peut
changer un réglage (taille des morceaux, modèle...) et voir si les résultats s'améliorent
ou se dégradent, au lieu de se fier à une impression.

Les questions sont dans eval/questions.json. Pour chaque question :
    "docs"     : une partie du nom du ou des documents qui contiennent la réponse
    "keywords" : (facultatif) mots que le morceau doit contenir pour compter comme pertinent
Un résultat est « pertinent » s'il vient d'un document attendu ET (s'il y a des mots-clés) contient
au moins l'un d'eux.

Mesures affichées :
    hit@k : part des questions pour lesquelles au moins un résultat pertinent est dans les k premiers
    MRR   : moyenne de 1/rang du premier résultat pertinent (1 si c'est le 1er, 0,5 si le 2e...,
            0 si aucun dans les résultats). Plus c'est proche de 1, mieux c'est.

Par défaut, les quatre modes de recherche (semantic, keyword, hybrid, rerank) sont comparés côte à côte.
Le mode rerank télécharge un second modèle au premier lancement (environ 1,1 Go).

Usage :
    python src/evaluate.py
    python src/evaluate.py --mode hybrid
    python src/evaluate.py --config fixed-1000-100
"""
import argparse
import json
import sys

from embedding import MODEL_NAME, embed_query, load_model
from retrieval import Retriever, load_reranker
from search import DEFAULT_CONFIG, DB_PATH, fetch_candidates

DEFAULT_QUESTIONS = "eval/questions.json"
HIT_LEVELS = (1, 3, 5)   # niveaux k de hit@k
RESULTS_FETCHED = 10     # nombre de résultats examinés par question


def is_relevant(filename: str, text: str, expected: dict) -> bool:
    """Un résultat est pertinent s'il vient d'un document attendu (et contient un mot-clé, s'il y en a)."""
    name = filename.casefold()  # casefold : comparaison sans tenir compte des majuscules
    if not any(doc.casefold() in name for doc in expected["docs"]):
        return False
    keywords = expected.get("keywords")
    if not keywords:
        return True
    body = text.casefold()
    return any(keyword.casefold() in body for keyword in keywords)


def first_relevant_rank(results, expected: dict):
    """Rang (1 = premier) du premier résultat pertinent, ou None s'il n'y en a aucun.

    `results` : liste de (score, ligne) dans l'ordre du classement, comme renvoyée par la recherche.
    """
    for rank, (_, row) in enumerate(results, start=1):
        filename, text = row[1], row[4]
        if is_relevant(filename, text, expected):
            return rank
    return None


def summarize(ranks, levels=HIT_LEVELS) -> dict:
    """Calcule hit@k et MRR à partir de la liste des rangs (None = aucun résultat pertinent)."""
    n = len(ranks)
    summary = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in levels}
    summary["MRR"] = sum(1 / r for r in ranks if r is not None) / n
    return summary


def format_rank(rank) -> str:
    return str(rank) if rank else "ABSENT"


def main():
    parser = argparse.ArgumentParser(description="Évalue la recherche.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="réglages de découpage à évaluer")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="fichier JSON de questions")
    parser.add_argument("--mode", default="all",
                        help="« all » (les quatre modes), un mode, ou une liste séparée par des virgules, "
                             "par exemple semantic,keyword,hybrid")
    args = parser.parse_args()
    modes = Retriever.MODES if args.mode == "all" else tuple(args.mode.split(","))
    unknown = [mode for mode in modes if mode not in Retriever.MODES]
    if unknown:
        sys.exit(f"Mode(s) inconnu(s) : {unknown}. Choix : {Retriever.MODES} ou all")

    import duckdb  # importé ici pour pouvoir tester les fonctions ci-dessus sans base de données

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)

    con = duckdb.connect(DB_PATH, read_only=True)
    candidates = fetch_candidates(con, args.config)
    if not candidates:
        sys.exit("Aucun vecteur trouvé pour ces réglages : lancez d'abord embed_chunks.py")

    needs_model = any(mode != "keyword" for mode in modes)
    model = load_model() if needs_model else None
    reranker = load_reranker() if "rerank" in modes else None
    retriever = Retriever(candidates, reranker)  # index construits une seule fois
    print(f"Réglages : {args.config} | modèle : {MODEL_NAME} | {len(candidates)} morceaux | "
          f"{len(questions)} questions\n")

    print("      " + "  ".join(f"{mode:<9}" for mode in modes) + " (rang du premier résultat pertinent)")
    ranks = {mode: [] for mode in modes}
    for number, expected in enumerate(questions, start=1):
        query_vector = embed_query(model, expected["question"]) if needs_model else None
        cells = []
        for mode in modes:
            results = retriever.search(expected["question"], query_vector, mode, RESULTS_FETCHED)
            rank = first_relevant_rank(results, expected)
            ranks[mode].append(rank)
            cells.append(f"{format_rank(rank):<9}")
        print(f"Q{number:02d}   " + "  ".join(cells) + f" {expected['question']}")

    print("\n=== Résultats ===")
    print(f"{'mode':<10}" + "".join(f"{name:>7}" for name in ("hit@1", "hit@3", "hit@5", "MRR")))
    for mode in modes:
        summary = summarize(ranks[mode])
        print(f"{mode:<10}" + "".join(f"{value:>7.2f}" for value in summary.values()))


if __name__ == "__main__":
    main()
