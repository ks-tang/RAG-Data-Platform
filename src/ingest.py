import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from pypdf import PdfReader

RAW_DIR = Path("data/raw")
DB_PATH = "data/warehouse.duckdb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_pages(path: Path):
    reader = PdfReader(str(path))
    return [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]


def main():
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id VARCHAR PRIMARY KEY,
            filename VARCHAR,
            ingested_at TIMESTAMP,
            n_pages INTEGER
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            doc_id VARCHAR,
            page_number INTEGER,
            text VARCHAR,
            n_chars INTEGER
        )""")

    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        doc_id = sha256(pdf)
        if con.execute("SELECT 1 FROM documents WHERE doc_id = ?", [doc_id]).fetchone():
            print(f"SKIP (déjà ingéré) : {pdf.name}")
            continue
        try:
            pages = extract_pages(pdf)
        except Exception as e:
            print(f"ERREUR : {pdf.name} -> {e}", file=sys.stderr)
            continue
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        con.execute("INSERT INTO documents VALUES (?, ?, ?, ?)",
                    [doc_id, pdf.name, now, len(pages)])
        con.executemany("INSERT INTO pages VALUES (?, ?, ?, ?)",
                        [(doc_id, n, t, len(t)) for n, t in pages])
        print(f"OK : {pdf.name} ({len(pages)} pages)")


if __name__ == "__main__":
    main()