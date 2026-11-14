"""Vérifications rapides après la semaine 2 (lecture seule, ne modifie rien)."""
import duckdb

# On ouvre la base en "lecture seule" : impossible de l'abîmer par accident.
con = duckdb.connect("data/warehouse.duckdb", read_only=True)

# 1) Les métadonnées trouvées pour chaque document.
#    À vérifier à l'œil : le titre est-il lisible ? La langue est-elle "fr" ou "en" ?
#    L'année est-elle plausible ?
print("--- Métadonnées des documents")
con.sql("""
    SELECT filename, title, language, publication_year, classification
    FROM documents_enriched
    ORDER BY filename
""").show(max_width=250)

# 2) Les pages presque vides du document signalé plus haut.
#    "n_chars" est le nombre de caractères de la page : sous 50, la page est quasi vide.
#    JOIN ... USING (doc_id) relie chaque page à son document grâce à l'identifiant commun.
print("--- Pages quasi vides de Document_Analyse_de_risques_FR.pdf")
con.sql("""
    SELECT p.page_number, p.n_chars
    FROM pages p
    JOIN documents d USING (doc_id)
    WHERE d.filename = 'Document_Analyse_de_risques_FR.pdf'
      AND p.n_chars < 50
    ORDER BY p.page_number
""").show()