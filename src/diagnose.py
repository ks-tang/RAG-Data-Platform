"""Diagnostic d'une question : où se classe le meilleur morceau d'un document précis ?

Quand la recherche ne renvoie pas le document attendu, deux questions se posent :
    1. Le bon morceau est-il juste derrière les premiers (rang 11, 12...) ou très loin (rang 500) ?
    2. Le mot clé de la question apparaît-il vraiment dans les bons morceaux ?
Ce script répond aux deux. Il ne modifie rien.

Usage :
    python src/diagnose.py "Quelles mesures de protection pour les systèmes industriels ?" --doc Mesures_detaillees
    python src/diagnose.py "Quels sont les risques de l'IA agentique ?" --doc ia_agentique --word agentique
"""
import argparse
import sys
from collections import Counter

import numpy as np

from embedding import embed_query, load_model
from search import DB_PATH, DEFAULT_CONFIG, fetch_candidates

EXCERPT_LENGTH = 400


def global_ranks(scores):
    """Rang de chaque morceau dans le classement général (1 = le plus proche de la question)."""
    order = np.argsort(-scores)                    # positions classées du meilleur au moins bon
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)   # le morceau à la position order[0] a le rang 1, etc.
    return ranks


def count_word_by_document(candidates, word: str) -> Counter:
    """Nombre de morceaux contenant `word`, document par document (sans tenir compte des majuscules)."""
    word = word.casefold()
    return Counter(row[1] for row in candidates if word in row[4].casefold())


def main():
    parser = argparse.ArgumentParser(description="Diagnostic d'une question de recherche.")
    parser.add_argument("question")
    parser.add_argument("--doc", required=True, help="partie du nom du document attendu")
    parser.add_argument("--word", help="mot à rechercher tel quel dans le texte des morceaux")
    parser.add_argument("--show", type=int, default=3, help="nombre de morceaux du document à afficher")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    import duckdb  # importé ici pour pouvoir tester les fonctions ci-dessus sans base de données

    con = duckdb.connect(DB_PATH, read_only=True)
    candidates = fetch_candidates(con, args.config)
    if not candidates:
        sys.exit("Aucun vecteur trouvé : lancez d'abord embed_chunks.py")

    matrix = np.array([row[5] for row in candidates], dtype=np.float32)
    scores = matrix @ embed_query(load_model(), args.question)
    ranks = global_ranks(scores)
    total = len(candidates)

    print(f"Question : {args.question}")
    print(f"({total} morceaux comparés)\n")

    print("--- Les 3 premiers résultats de la recherche")
    for i in np.argsort(-scores)[:3]:
        print(f"  rang {ranks[i]:>4}  score {scores[i]:.3f}  {candidates[i][1]}, p. {candidates[i][2]}")

    doc_positions = [i for i, row in enumerate(candidates) if args.doc.casefold() in row[1].casefold()]
    if not doc_positions:
        sys.exit(f"\nAucun document ne contient « {args.doc} » dans son nom.")
    doc_positions.sort(key=lambda i: ranks[i])
    doc_ranks = [int(ranks[i]) for i in doc_positions]
    print(f"\n--- Document « {args.doc} » ({len(doc_positions)} morceaux)")
    print(f"  meilleur rang : {doc_ranks[0]} sur {total} | rang médian : {int(np.median(doc_ranks))}")
    for i in doc_positions[:args.show]:
        row = candidates[i]
        excerpt = row[4].replace("\n", " ↵ ")[:EXCERPT_LENGTH]
        print(f"\n  rang {ranks[i]} | score {scores[i]:.3f} | p. {row[2]}")
        print(f"  {excerpt}")

    if args.word:
        sizes = Counter(row[1] for row in candidates)
        counts = count_word_by_document(candidates, args.word)
        print(f"\n--- Morceaux contenant le mot « {args.word} » : {sum(counts.values())} au total")
        for filename, n in counts.most_common(8):
            print(f"  {n:>4} sur {sizes[filename]:>4}  {filename}")


if __name__ == "__main__":
    main()
