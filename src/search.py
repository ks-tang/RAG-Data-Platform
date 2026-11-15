"""Recherche sémantique : trouve les morceaux les plus proches d'une question.

Usage :
    python src/search.py "Comment fonctionne le zero trust ?"
    python src/search.py "Comment sauvegarder un SI ?" --top 3
    python src/search.py "..." --config fixed-1000-100

Principe : on transforme la question en vecteur, puis on compare ce vecteur à celui de TOUS les
morceaux (recherche « exacte »). Avec quelques milliers de morceaux, c'est instantané. Au-delà
de plusieurs centaines de milliers, on utiliserait un index vectoriel (pgvector, Qdrant...).
"""
import argparse
import sys

import duckdb
import numpy as np

from embedding import MODEL_NAME, embed_query, load_model, top_matches

DB_PATH = "data/warehouse.duckdb"
DEFAULT_CONFIG = "sentences-800-1"
EXCERPT_LENGTH = 450  # nombre de caractères du morceau affichés


def fetch_candidates(con, config: str):
    """Lit en base tous les morceaux qui ont un vecteur, avec leur document et leurs pages."""
    return con.execute("""
        SELECT c.chunk_id, d.filename, c.page_start, c.page_end, c.text, e.embedding
        FROM chunk_embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE e.model = ? AND c.config = ?
        ORDER BY c.chunk_id""", [MODEL_NAME, config]).fetchall()


def search(candidates, query_vector, top_k: int):
    """Renvoie [(score, ligne_candidate), ...] des top_k morceaux les plus proches de la question."""
    matrix = np.array([row[5] for row in candidates], dtype=np.float32)  # un vecteur par ligne
    return [(score, candidates[i]) for i, score in top_matches(matrix, query_vector, top_k)]


def format_pages(page_start: int, page_end: int) -> str:
    return f"p. {page_start}" if page_start == page_end else f"p. {page_start}-{page_end}"


def main():
    parser = argparse.ArgumentParser(description="Recherche sémantique dans les morceaux.")
    parser.add_argument("question")
    parser.add_argument("--top", type=int, default=5, help="nombre de résultats")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="réglages de découpage à interroger")
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH, read_only=True)
    candidates = fetch_candidates(con, args.config)
    if not candidates:
        sys.exit("Aucun vecteur trouvé : lancez d'abord embed_chunks.py")

    model = load_model()
    results = search(candidates, embed_query(model, args.question), args.top)

    print(f"Question : {args.question}")
    print(f"({len(candidates)} morceaux comparés)\n")
    for rank, (score, row) in enumerate(results, start=1):
        _, filename, page_start, page_end, text, _ = row
        excerpt = text.replace("\n", " ↵ ")
        if len(excerpt) > EXCERPT_LENGTH:
            excerpt = excerpt[:EXCERPT_LENGTH] + "..."
        print(f"#{rank}  score {score:.3f}  {filename}, {format_pages(page_start, page_end)}")
        print(f"    {excerpt}\n")


if __name__ == "__main__":
    main()
