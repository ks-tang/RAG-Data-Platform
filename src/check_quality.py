"""Contrôles qualité. Code de sortie 0 si tout est bon, 1 sinon
(ce qui permettra de bloquer un pipeline automatisé en cas de problème)."""
import sys

import duckdb

DB_PATH = "data/warehouse.duckdb"
MAX_LOSS = 0.30  # alerte si plus de 30 % du texte disparaît au nettoyage

con = duckdb.connect(DB_PATH, read_only=True)
failures = []


def show(title, sql):
    print(f"\n--- {title}")
    try:
        con.sql(sql).show()
    except Exception as e:
        print(f"(indisponible : {e})")


def check(name, sql):
    """Un contrôle réussit si la requête ne renvoie AUCUNE ligne (aucun cas anormal)."""
    try:
        rows = con.execute(sql).fetchall()
    except Exception as e:
        print(f"ECHEC  {name} : requête impossible ({e})")
        failures.append(name)
        return
    if rows:
        print(f"ECHEC  {name} : {len(rows)} cas anormal(aux), par exemple {rows[:3]}")
        failures.append(name)
    else:
        print(f"OK     {name}")


# ---- Informations (n'influencent pas le code de sortie)
show("Documents ingérés", "SELECT filename, n_pages FROM documents ORDER BY n_pages DESC")
show("Documents avec plus de 10 % de pages quasi vides (< 50 caractères)", """
    WITH stats AS (
        SELECT d.filename,
               COUNT(*) FILTER (WHERE p.n_chars < 50) AS nb_vides,
               COUNT(*) AS nb_pages
        FROM pages p JOIN documents d USING (doc_id)
        GROUP BY d.filename)
    SELECT filename, nb_vides, nb_pages, ROUND(nb_vides * 1.0 / nb_pages, 2) AS proportion
    FROM stats WHERE nb_vides * 1.0 / nb_pages > 0.1 ORDER BY proportion DESC""")

# ---- Contrôles (une ligne renvoyée = un problème)
print("\n=== Contrôles qualité ===")
check("Chaque document a ses pages nettoyées", """
    SELECT filename FROM documents
    WHERE doc_id NOT IN (SELECT DISTINCT doc_id FROM pages_clean)""")
check("Chaque page brute a sa page nettoyée", """
    SELECT d.filename FROM documents d
    WHERE (SELECT COUNT(*) FROM pages p WHERE p.doc_id = d.doc_id)
       <> (SELECT COUNT(*) FROM pages_clean c WHERE c.doc_id = d.doc_id)""")
check("Aucun text_clean NULL", "SELECT doc_id, page_number FROM pages_clean WHERE text_clean IS NULL")
check("Pas de doublons dans pages_clean", """
    SELECT doc_id, page_number, COUNT(*) FROM pages_clean
    GROUP BY doc_id, page_number HAVING COUNT(*) > 1""")
check(f"Pas plus de {int(MAX_LOSS * 100)} % du texte perdu au nettoyage", f"""
    SELECT d.filename, ROUND(SUM(c.n_chars_clean) * 1.0 / SUM(p.n_chars), 2) AS ratio
    FROM pages p
    JOIN pages_clean c USING (doc_id, page_number)
    JOIN documents d USING (doc_id)
    GROUP BY d.filename
    HAVING SUM(p.n_chars) > 0
       AND SUM(c.n_chars_clean) * 1.0 / SUM(p.n_chars) < {1 - MAX_LOSS}""")
check("Chaque document a des métadonnées", """
    SELECT filename FROM documents
    WHERE doc_id NOT IN (SELECT doc_id FROM document_metadata)""")
check("Langue renseignée et reconnue", """
    SELECT d.filename FROM documents d JOIN document_metadata m USING (doc_id)
    WHERE m.language IS NULL OR m.language = 'unknown'""")
check("Classification valide", """
    SELECT d.filename FROM documents d JOIN document_metadata m USING (doc_id)
    WHERE m.classification IS NULL
       OR m.classification NOT IN ('Public', 'Interne', 'Confidentiel')""")

print()
if failures:
    print(f"{len(failures)} contrôle(s) en échec : {failures}")
    sys.exit(1)
print("Tous les contrôles sont passés.")
