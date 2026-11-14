import duckdb

con = duckdb.connect("data/warehouse.duckdb")

print("Documents ingérés :")
print(con.execute("SELECT filename, n_pages FROM documents ORDER BY n_pages DESC").df())

print("\nPages quasi vides (< 50 caractères), souvent des pages scannées ou des images :")
print(con.execute("""
    SELECT d.filename, COUNT(*) AS pages_vides
    FROM pages p JOIN documents d USING (doc_id)
    WHERE p.n_chars < 50
    GROUP BY d.filename
    ORDER BY pages_vides DESC
""").df())