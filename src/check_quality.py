"""Contrôles qualité. Code de sortie 0 si tout est bon, 1 sinon
(ce qui permettra de bloquer un pipeline automatisé en cas de problème)."""
import sys

import duckdb

DB_PATH = "data/warehouse.duckdb"
MAX_LOSS = 0.30  # alerte si plus de 30 % du texte disparaît au nettoyage
# Réglages vérifiés pour les embeddings (à adapter si on change de modèle ou de découpage).
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSIONS = 384
CHUNK_CONFIG = "sentences-800-1"

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
check("Aucune fausse puce (I, >, o) restante en début de ligne", r"""
    SELECT d.filename, p.page_number
    FROM pages_clean p JOIN documents d USING (doc_id)
    WHERE regexp_matches(p.text_clean, '(^|\n)(I|>|o)[ ]+[A-Za-zÀ-ÿ]')""")
check("Chaque document a des morceaux (chunks)", """
    SELECT filename FROM documents
    WHERE doc_id NOT IN (SELECT DISTINCT doc_id FROM chunks)""")
check("Aucun chunk_id en double", """
    SELECT chunk_id, COUNT(*) FROM chunks GROUP BY chunk_id HAVING COUNT(*) > 1""")
check("Aucun morceau vide", """
    SELECT chunk_id FROM chunks WHERE text IS NULL OR n_chars = 0""")
check("Tous les morceaux appartiennent à un document connu", """
    SELECT DISTINCT doc_id FROM chunks WHERE doc_id NOT IN (SELECT doc_id FROM documents)""")
check("Pages de début et de fin cohérentes", """
    SELECT chunk_id FROM chunks WHERE page_start > page_end""")
check("Les pages citées par les morceaux existent", """
    SELECT c.chunk_id FROM chunks c
    WHERE NOT EXISTS (SELECT 1 FROM pages_clean p
                      WHERE p.doc_id = c.doc_id AND p.page_number = c.page_start)
       OR NOT EXISTS (SELECT 1 FROM pages_clean p
                      WHERE p.doc_id = c.doc_id AND p.page_number = c.page_end)""")
check("Chaque morceau a un embedding à jour", f"""
    SELECT c.chunk_id
    FROM chunks c
    LEFT JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id AND e.model = '{EMBEDDING_MODEL}'
    WHERE c.config = '{CHUNK_CONFIG}'
      AND (e.chunk_id IS NULL OR e.text_hash <> sha256(c.text))""")
check("Embeddings de la bonne dimension", f"""
    SELECT chunk_id FROM chunk_embeddings
    WHERE model = '{EMBEDDING_MODEL}' AND len(embedding) <> {EMBEDDING_DIMENSIONS}""")
check("Aucun embedding en double", """
    SELECT chunk_id, model, COUNT(*) FROM chunk_embeddings
    GROUP BY chunk_id, model HAVING COUNT(*) > 1""")
check("Aucun embedding orphelin (morceau supprimé)", """
    SELECT chunk_id FROM chunk_embeddings
    WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks)""")
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
