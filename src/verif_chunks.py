"""Vérifications rapides de la table chunks (lecture seule, ne modifie rien).

Usage :
    python src/verif_chunks.py                              -> résumé par document
    python src/verif_chunks.py "mon_fichier.pdf" 12         -> morceaux qui touchent la page 12
"""
import sys

import duckdb

# Lecture seule : impossible d'abîmer la base par accident.
con = duckdb.connect("data/warehouse.duckdb", read_only=True)

if len(sys.argv) < 3:
    # Résumé : combien de morceaux par document et par réglage, et quelle taille moyenne.
    # GROUP BY regroupe les lignes par document ; COUNT, AVG... calculent un chiffre par groupe.
    print("--- Morceaux par document")
    con.sql("""
        SELECT d.filename, c.config, COUNT(*) AS nb_morceaux,
               ROUND(AVG(c.n_chars)) AS taille_moyenne
        FROM chunks c
        JOIN documents d USING (doc_id)
        GROUP BY d.filename, c.config
        ORDER BY d.filename
    """).show(max_width=200)
    print('Pour voir des morceaux : python src/verif_chunks.py "nom.pdf" NUMERO_DE_PAGE')
else:
    filename, page = sys.argv[1], int(sys.argv[2])
    # Les morceaux dont la plage de pages [page_start, page_end] contient la page demandée.
    rows = con.execute("""
        SELECT c.chunk_id, c.page_start, c.page_end, c.n_chars, c.text
        FROM chunks c
        JOIN documents d USING (doc_id)
        WHERE d.filename = ? AND c.page_start <= ? AND c.page_end >= ?
        ORDER BY c.chunk_index
    """, [filename, page, page]).fetchall()
    print(f"{len(rows)} morceau(x) touchent la page {page}")
    for chunk_id, page_start, page_end, n_chars, text in rows:
        print(f"\n[{chunk_id}] pages {page_start}-{page_end}, {n_chars} caractères")
        print(text.replace("\n", " ↵ "))
