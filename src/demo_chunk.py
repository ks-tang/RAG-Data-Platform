"""Compare les deux stratégies de découpage sur UN document.

Usage :
    python src/demo_chunk.py                          -> liste les documents
    python src/demo_chunk.py "mon_fichier.pdf"        -> compare sur le document entier,
                                                         et montre les morceaux d'une page du milieu
    python src/demo_chunk.py "mon_fichier.pdf" 12     -> idem, en montrant les morceaux de la page 12
"""
import sys

from chunking import STRATEGIES, chunk_document, ends_like_a_sentence

MAX_CHUNKS_SHOWN = 4  # nombre maximal de morceaux affichés en détail par stratégie


def statistics(chunks) -> dict:
    """Calcule quelques chiffres pour comparer les stratégies."""
    sizes = [len(c.text) for c in chunks]
    if not sizes:
        return {"morceaux": 0}
    clean_endings = sum(ends_like_a_sentence(c.text) for c in chunks)
    return {
        "morceaux": len(chunks),
        "taille moyenne": round(sum(sizes) / len(sizes)),
        "taille min": min(sizes),
        "taille max": max(sizes),
        "% finissant par une fin de phrase": round(100 * clean_endings / len(chunks)),
    }


def show_chunk(chunk):
    """Affiche un morceau. Les retours à la ligne sont rendus visibles avec le symbole ↵."""
    pages = (f"page {chunk.page_start}" if chunk.page_start == chunk.page_end
             else f"pages {chunk.page_start}-{chunk.page_end}")
    print(f"\n  [{chunk.index}] {pages}, {len(chunk.text)} caractères")
    print("  " + chunk.text.replace("\n", " ↵ "))


def main():
    import duckdb  # importé ici pour pouvoir tester statistics() sans base de données
    con = duckdb.connect("data/warehouse.duckdb", read_only=True)

    if len(sys.argv) < 2:
        print("Documents disponibles :")
        con.sql("SELECT filename, n_pages FROM documents ORDER BY filename").show()
        print('Relancez avec : python src/demo_chunk.py "nom.pdf" [NUMERO_DE_PAGE]')
        return

    filename = sys.argv[1]
    doc = con.execute("SELECT doc_id FROM documents WHERE filename = ?", [filename]).fetchone()
    if not doc:
        sys.exit(f"Document introuvable : {filename}")

    # On lit le texte NETTOYÉ (table pages_clean), page par page.
    pages = con.execute(
        "SELECT page_number, text_clean FROM pages_clean WHERE doc_id = ? ORDER BY page_number",
        [doc[0]]).fetchall()
    if not pages:
        sys.exit("Aucun texte nettoyé pour ce document : lancez d'abord transform.py")

    # Page à examiner : celle donnée en argument, sinon une page du milieu du document.
    page = int(sys.argv[2]) if len(sys.argv) > 2 else pages[len(pages) // 2][0]

    print("=" * 70)
    print(f"DOCUMENT : {filename} ({len(pages)} pages)")
    print("=" * 70)

    # 1) Découpe avec chaque stratégie et statistiques sur tout le document.
    results = {name: chunk_document(pages, name) for name in STRATEGIES}
    print("\n### COMPARAISON SUR TOUT LE DOCUMENT")
    for name, chunks in results.items():
        print(f"\n  Stratégie « {name} » :")
        for label, value in statistics(chunks).items():
            print(f"    {label:<36} {value}")

    # 2) Détail des morceaux qui touchent la page choisie.
    for name, chunks in results.items():
        on_page = [c for c in chunks if c.page_start <= page <= c.page_end]
        print("\n" + "=" * 70)
        print(f"STRATÉGIE « {name} » : {len(on_page)} morceau(x) touchent la page {page}")
        print("=" * 70)
        for chunk in on_page[:MAX_CHUNKS_SHOWN]:
            show_chunk(chunk)


if __name__ == "__main__":
    main()
