"""Suit UNE page à travers chaque étape du nettoyage.

Usage :
    python src/demo_clean.py                       -> liste les documents
    python src/demo_clean.py "mon_fichier.pdf" 12  -> trace la page 12 de ce document
"""
import re
import sys
import unicodedata

import clean

# (nom de l'étape, motif regex pour compter les modifications ou None, fonction)
STEPS = [
    ("1. Unicode, espaces spéciaux, fins de ligne", None,
     lambda t: re.sub(r"[\u00a0\u2007\u202f\u2009\u200a]", " ",
                      unicodedata.normalize("NFC", t).replace("\u00ad", ""))
     .replace("\r\n", "\n").replace("\r", "\n")),
    ("2. Caractères de contrôle", clean.CONTROL_CHARS,
     lambda t: clean.CONTROL_CHARS.sub("", t)),
    ("3. Puces (symboles)", clean.LINE_START_BULLET,
     lambda t: clean.STRAY_SYMBOLS.sub(" ", clean.LINE_START_BULLET.sub("- ", t))),
    ("4. Fausses puces (« I », « > », « o »)", clean.FAKE_BULLET,
     lambda t: clean.FAKE_BULLET.sub("- ", t)),
    ("5. Mots coupés en fin de ligne", clean.HYPHEN_BREAK,
     lambda t: clean.HYPHEN_BREAK.sub("", t)),
    ("6. Tirets de mots composés coupés", clean.COMPOUND_BREAK,
     lambda t: clean.COMPOUND_BREAK.sub("-", t)),
    ("7. Sauts de ligne au milieu d'une phrase", clean.SOFT_NEWLINE,
     lambda t: clean.SOFT_NEWLINE.sub(" ", t)),
    ("8. Espaces multiples et lignes vides", None,
     lambda t: re.sub(r"\n{3,}", "\n\n", re.sub(r" ?\n ?", "\n", re.sub(r"[ \t]+", " ", t))).strip()),
]


def trace(text: str):
    """Applique les étapes une à une. Renvoie la liste (nom, avant, après, nb_modifs)."""
    result = []
    for name, pattern, fn in STEPS:
        before = text
        text = fn(text)
        n = len(pattern.findall(before)) if pattern is not None else None
        result.append((name, before, text, n))
    return result


def main():
    import duckdb  # importé ici pour pouvoir tester trace() sans base
    con = duckdb.connect("data/warehouse.duckdb", read_only=True)

    if len(sys.argv) < 3:
        print("Documents disponibles :")
        con.sql("SELECT filename, n_pages FROM documents ORDER BY filename").show()
        print('Relancez avec : python src/demo_clean.py "nom.pdf" NUMERO_DE_PAGE')
        return

    filename, page = sys.argv[1], int(sys.argv[2])
    doc = con.execute("SELECT doc_id FROM documents WHERE filename = ?", [filename]).fetchone()
    if not doc:
        sys.exit(f"Document introuvable : {filename}")
    rows = con.execute("SELECT page_number, text FROM pages WHERE doc_id = ? ORDER BY page_number",
                       [doc[0]]).fetchall()
    numbers = [r[0] for r in rows]
    texts = [r[1] or "" for r in rows]
    if page not in numbers:
        sys.exit(f"Page {page} introuvable (le document a {len(numbers)} pages).")
    i = numbers.index(page)

    raw = texts[i]
    print("=" * 70)
    print(f"DOCUMENT : {filename} - PAGE {page}")
    print("=" * 70)
    print(f"\n### TEXTE BRUT ({len(raw)} caractères), tel que stocké dans `pages`")
    print(repr(raw[:800]))

    # Étape 0 : en-têtes / pieds de page (calculée sur TOUTES les pages du document)
    no_headers = clean.remove_repeated_lines(texts)[i]
    removed = [l for l in raw.splitlines() if l.strip() and l not in no_headers.splitlines()]
    print(f"\n### ÉTAPE 0 : en-têtes et pieds de page retirés ({len(removed)} ligne(s))")
    for l in removed:
        print("   supprimée :", repr(l))

    text = no_headers
    for name, before, after, n in trace(text):
        changed = before != after
        extra = f", {n} occurrence(s) trouvée(s)" if n is not None else ""
        print(f"\n### ÉTAPE {name} : {len(before)} -> {len(after)} caractères"
              f"{extra}{'' if changed else ' (aucun changement)'}")
        if changed:
            print(repr(after[:800]))

    final = after
    print("\n" + "=" * 70)
    print("RÉSULTAT FINAL (texte lisible) :")
    print("=" * 70)
    print(final[:1500])
    print("\nIdentique à clean_text() :", final == clean.clean_text(no_headers))


if __name__ == "__main__":
    main()
