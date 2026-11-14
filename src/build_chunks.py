"""Étape 4 du pipeline : texte nettoyé (table pages_clean) -> morceaux (table chunks).

Chaque ligne de la table `chunks` est un morceau de document, avec de quoi le retrouver :
le document d'origine, sa position, les pages où il se trouve.

Le script est idempotent ET sait détecter qu'un document a changé :
    - on calcule une « empreinte » (hash) du texte nettoyé du document ;
    - si le document a déjà des morceaux avec la même empreinte et les mêmes réglages : on ne fait rien ;
    - si l'empreinte a changé (par exemple après une modification de clean.py) : on remplace
      les morceaux de CE document, et seulement les siens.

Plusieurs réglages peuvent coexister dans la table (colonne `config`) : c'est ce qui permettra,
en phase d'évaluation, de comparer par exemple 800 et 1200 caractères sur les mêmes documents.

Usage :
    python src/build_chunks.py                                  -> réglages par défaut
    python src/build_chunks.py --strategy fixed --size 1000     -> autre stratégie / autre taille
    python src/build_chunks.py --rebuild                        -> vide la table et refait tout
                                                                   (à utiliser si on modifie chunking.py)
"""
import argparse
import hashlib
import sys
from datetime import datetime, timezone

import duckdb

from chunking import (DEFAULT_OVERLAP, DEFAULT_OVERLAP_SENTENCES, DEFAULT_SIZE,
                      STRATEGIES, build_document_text, chunk_document)

DB_PATH = "data/warehouse.duckdb"


def default_overlap(strategy: str) -> int:
    """Le chevauchement se compte en caractères pour "fixed", en phrases pour "sentences"."""
    return DEFAULT_OVERLAP if strategy == "fixed" else DEFAULT_OVERLAP_SENTENCES


def make_config(strategy: str, size: int, overlap: int) -> str:
    """Nom court d'un jeu de réglages, par exemple « sentences-800-1 »."""
    return f"{strategy}-{size}-{overlap}"


def chunk_options(strategy: str, size: int, overlap: int) -> dict:
    """Traduit les réglages en arguments pour chunk_document (les deux stratégies n'ont pas le même nom)."""
    if strategy == "fixed":
        return {"size": size, "overlap": overlap}
    return {"size": size, "overlap_sentences": overlap}


def text_fingerprint(text: str) -> str:
    """Empreinte SHA-256 d'un texte : change dès qu'un seul caractère change."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_chunk_id(doc_id: str, config: str, index: int) -> str:
    """Identifiant stable : le même document, les mêmes réglages et le même rang donnent toujours le même identifiant."""
    return f"{doc_id[:16]}-{config}-{index:04d}"


def build_rows(doc_id, chunks, strategy, config, source_hash, created_at):
    """Transforme des objets Chunk en lignes prêtes à être insérées dans la table."""
    return [
        (make_chunk_id(doc_id, config, c.index), doc_id, c.index, config, strategy,
         c.text, len(c.text), c.page_start, c.page_end, source_hash, created_at)
        for c in chunks
    ]


def main():
    parser = argparse.ArgumentParser(description="Découpe les textes nettoyés en morceaux.")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="sentences")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help="taille maximale d'un morceau, en caractères")
    parser.add_argument("--overlap", type=int, default=None,
                        help="chevauchement : caractères pour 'fixed', phrases pour 'sentences'")
    parser.add_argument("--rebuild", action="store_true",
                        help="supprime la table chunks et recalcule tout")
    args = parser.parse_args()

    overlap = args.overlap if args.overlap is not None else default_overlap(args.strategy)
    config = make_config(args.strategy, args.size, overlap)
    options = chunk_options(args.strategy, args.size, overlap)

    con = duckdb.connect(DB_PATH)
    if args.rebuild:
        con.execute("DROP TABLE IF EXISTS chunks")
        print("Table chunks supprimée, recalcul complet.")

    # Pas de PRIMARY KEY volontairement : avec DuckDB, supprimer puis réinsérer les mêmes clés dans
    # une même transaction peut provoquer une fausse erreur de doublon. L'unicité de chunk_id est
    # donc vérifiée par check_quality.py.
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id VARCHAR,
            doc_id VARCHAR,
            chunk_index INTEGER,
            config VARCHAR,
            strategy VARCHAR,
            text VARCHAR,
            n_chars INTEGER,
            page_start INTEGER,
            page_end INTEGER,
            source_hash VARCHAR,
            created_at TIMESTAMP
        )""")

    print(f"Réglages : {config}")
    documents = con.execute("SELECT doc_id, filename FROM documents ORDER BY filename").fetchall()
    counts = {"OK": 0, "MIS À JOUR": 0, "SKIP": 0, "IGNORÉ": 0, "ERREUR": 0}

    for doc_id, filename in documents:
        pages = con.execute(
            "SELECT page_number, text_clean FROM pages_clean WHERE doc_id = ? ORDER BY page_number",
            [doc_id]).fetchall()
        if not pages:
            print(f"IGNORÉ : {filename} (pas de texte nettoyé : lancez transform.py)")
            counts["IGNORÉ"] += 1
            continue

        # Empreinte du texte exact qui va être découpé.
        source_hash = text_fingerprint(build_document_text(pages)[0])
        existing = con.execute(
            "SELECT source_hash FROM chunks WHERE doc_id = ? AND config = ? LIMIT 1",
            [doc_id, config]).fetchone()
        if existing and existing[0] == source_hash:
            print(f"SKIP (à jour) : {filename}")
            counts["SKIP"] += 1
            continue

        chunks = chunk_document(pages, args.strategy, **options)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rows = build_rows(doc_id, chunks, args.strategy, config, source_hash, now)
        try:
            con.begin()  # tout ou rien : jamais un document à moitié découpé
            con.execute("DELETE FROM chunks WHERE doc_id = ? AND config = ?", [doc_id, config])
            con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            con.commit()
        except Exception as e:
            con.rollback()
            print(f"ERREUR : {filename} -> {e}", file=sys.stderr)
            counts["ERREUR"] += 1
            continue
        status = "MIS À JOUR" if existing else "OK"
        counts[status] += 1
        print(f"{status} : {filename} ({len(rows)} morceaux)")

    print("\nBilan : " + ", ".join(f"{n} {label}" for label, n in counts.items() if n))
    if counts["ERREUR"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
