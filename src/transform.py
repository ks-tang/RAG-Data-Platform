"""Étape 2 du pipeline : texte brut (table pages) -> texte nettoyé (table pages_clean).

La donnée brute n'est jamais modifiée. Le script est idempotent : un document déjà
transformé est ignoré. Pour tout recalculer après avoir changé clean.py :
    python src/transform.py --rebuild
"""
import argparse
import sys

import duckdb

from clean import clean_text, remove_repeated_lines

DB_PATH = "data/warehouse.duckdb"


def main():
    parser = argparse.ArgumentParser(description="Nettoie le texte des pages ingérées.")
    parser.add_argument("--rebuild", action="store_true",
                        help="supprime pages_clean et recalcule tout depuis le texte brut")
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH)
    if args.rebuild:
        con.execute("DROP TABLE IF EXISTS pages_clean")
        print("pages_clean supprimée, recalcul complet.")

    con.execute("""
        CREATE TABLE IF NOT EXISTS pages_clean (
            doc_id VARCHAR,
            page_number INTEGER,
            text_clean VARCHAR,
            n_chars_clean INTEGER,
            PRIMARY KEY (doc_id, page_number)
        )""")

    todo = con.execute("""
        SELECT doc_id, filename FROM documents
        WHERE doc_id NOT IN (SELECT DISTINCT doc_id FROM pages_clean)
        ORDER BY filename""").fetchall()

    if not todo:
        print("Rien à transformer : tous les documents sont déjà nettoyés.")
        return

    for doc_id, filename in todo:
        rows = con.execute(
            "SELECT page_number, text FROM pages WHERE doc_id = ? ORDER BY page_number",
            [doc_id]).fetchall()
        numbers = [r[0] for r in rows]
        raw_texts = [r[1] or "" for r in rows]

        # 1) en-têtes / pieds de page (nécessite la structure en lignes, donc avant clean_text)
        without_headers = remove_repeated_lines(raw_texts)
        # 2) nettoyage page par page
        cleaned = [clean_text(t) for t in without_headers]

        data = [(doc_id, n, t, len(t)) for n, t in zip(numbers, cleaned)]
        try:
            con.begin()  # tout ou rien : pas de document à moitié transformé
            con.executemany("INSERT INTO pages_clean VALUES (?, ?, ?, ?)", data)
            con.commit()
            print(f"OK : {filename} ({len(data)} pages)")
        except Exception as e:
            con.rollback()
            print(f"ERREUR : {filename} -> {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
