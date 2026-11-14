"""Étape 3 du pipeline : métadonnées des documents -> table document_metadata.

Les métadonnées sont dans une table séparée : la table `documents` (brute) n'est pas modifiée.
Une vue `documents_enriched` joint les deux pour l'usage courant.
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import duckdb
from pypdf import PdfReader

try:
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    sys.exit("Il manque la bibliothèque langdetect. Lancez : pip install langdetect")

DetectorFactory.seed = 0  # résultats reproductibles

RAW_DIR = Path("data/raw")
DB_PATH = "data/warehouse.duckdb"
SOURCE = "ANSSI"  # à modifier si vous changez de corpus


def classify(n_pages: int) -> str:
    """Classification FICTIVE (sert à tester les droits d'accès en phase 4).
    Règle arbitraire, documentée dans le README : selon la taille du document."""
    if n_pages < 20:
        return "Public"
    if n_pages <= 80:
        return "Interne"
    return "Confidentiel"


def read_pdf_metadata(path: Path):
    """Renvoie (titre, année) tirés des métadonnées du PDF, ou (None, None)."""
    try:
        meta = PdfReader(str(path)).metadata
    except Exception:
        return None, None
    if meta is None:
        return None, None
    title = (meta.title or "").strip() or None
    year = None
    try:
        if meta.creation_date:
            year = meta.creation_date.year
    except Exception:
        pass
    return title, year


def is_usable_title(title) -> bool:
    if not title or len(title) < 4:
        return False
    if title.lower().startswith("microsoft word"):
        return False
    return not re.search(r"\.(docx?|pdf|indd|pptx?)$", title, re.IGNORECASE)


def guess_title(meta_title, first_page_text, filename) -> str:
    if is_usable_title(meta_title):
        return meta_title
    for line in (first_page_text or "").splitlines():
        if len(line.strip()) >= 4:
            return line.strip()[:150]
    return Path(filename).stem


def guess_year(meta_year, first_page_text):
    current = datetime.now().year
    if meta_year and 1990 <= meta_year <= current:
        return meta_year
    years = [int(y) for y in re.findall(r"\b(19[9]\d|20[0-3]\d)\b", first_page_text or "")]
    years = [y for y in years if y <= current]
    return max(years) if years else None


def detect_language(text: str) -> str:
    if len((text or "").strip()) < 50:
        return "unknown"
    try:
        return detect(text[:5000])
    except LangDetectException:
        return "unknown"


def main():
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS document_metadata (
            doc_id VARCHAR PRIMARY KEY,
            title VARCHAR,
            language VARCHAR,
            publication_year INTEGER,
            source VARCHAR,
            classification VARCHAR
        )""")

    try:
        todo = con.execute("""
            SELECT doc_id, filename, n_pages FROM documents
            WHERE doc_id NOT IN (SELECT doc_id FROM document_metadata)
            ORDER BY filename""").fetchall()
    except duckdb.CatalogException:
        sys.exit("Table introuvable : lancez d'abord ingest.py puis transform.py")

    for doc_id, filename, n_pages in todo:
        pages = con.execute(
            "SELECT text_clean FROM pages_clean WHERE doc_id = ? ORDER BY page_number LIMIT 5",
            [doc_id]).fetchall()
        if not pages:
            print(f"IGNORÉ : {filename} (pas de texte nettoyé : lancez transform.py)")
            continue
        texts = [p[0] or "" for p in pages]
        sample = "\n".join(texts)

        meta_title, meta_year = read_pdf_metadata(RAW_DIR / filename)
        title = guess_title(meta_title, texts[0], filename)
        year = guess_year(meta_year, texts[0])
        language = detect_language(sample)

        con.execute("INSERT INTO document_metadata VALUES (?, ?, ?, ?, ?, ?)",
                    [doc_id, title, language, year, SOURCE, classify(n_pages)])
        print(f"OK : {filename} -> {language}, {year}, {classify(n_pages)}")

    con.execute("""
        CREATE OR REPLACE VIEW documents_enriched AS
        SELECT d.doc_id, d.filename, d.n_pages, d.ingested_at,
               m.title, m.language, m.publication_year, m.source, m.classification
        FROM documents d
        LEFT JOIN document_metadata m USING (doc_id)""")


if __name__ == "__main__":
    main()
