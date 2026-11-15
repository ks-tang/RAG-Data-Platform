"""Étape 5 du pipeline : morceaux (table chunks) -> vecteurs (table chunk_embeddings).

Le script ne calcule que ce qui manque ou qui est périmé :
    - un morceau sans vecteur est calculé ;
    - un morceau dont le texte a changé depuis le dernier calcul (empreinte différente) est recalculé ;
    - les vecteurs de morceaux qui n'existent plus sont supprimés.
Il peut donc être relancé sans risque, et sans rien recalculer si tout est à jour.
Le travail est enregistré par paquets : si on l'interrompt, on reprend où on s'était arrêté.

Usage :
    python src/embed_chunks.py                       -> morceaux des réglages par défaut
    python src/embed_chunks.py --config fixed-1000-100
    python src/embed_chunks.py --rebuild             -> supprime tous les vecteurs et refait tout
"""
import argparse
import sys
import time
from datetime import datetime, timezone

import duckdb

from embedding import DIMENSIONS, MODEL_NAME, embed_passages, load_model, text_hash

DB_PATH = "data/warehouse.duckdb"
DEFAULT_CONFIG = "sentences-800-1"
CHUNKS_PER_COMMIT = 64  # on enregistre en base tous les 64 morceaux


def main():
    parser = argparse.ArgumentParser(description="Calcule les embeddings des morceaux.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="réglages de découpage à traiter (colonne config de la table chunks)")
    parser.add_argument("--rebuild", action="store_true",
                        help="supprime la table chunk_embeddings et recalcule tout")
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH)
    if args.rebuild:
        con.execute("DROP TABLE IF EXISTS chunk_embeddings")
        print("Table chunk_embeddings supprimée, recalcul complet.")

    # embedding FLOAT[] : une liste de nombres décimaux (384 par morceau).
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id VARCHAR,
            model VARCHAR,
            text_hash VARCHAR,
            embedding FLOAT[],
            created_at TIMESTAMP
        )""")

    # Pour chaque morceau : le texte, et l'empreinte du texte au moment du dernier calcul (NULL si jamais calculé).
    rows = con.execute("""
        SELECT c.chunk_id, c.text, e.text_hash
        FROM chunks c
        LEFT JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id AND e.model = ?
        WHERE c.config = ?
        ORDER BY c.chunk_id""", [MODEL_NAME, args.config]).fetchall()
    if not rows:
        sys.exit(f"Aucun morceau pour les réglages « {args.config} » : lancez d'abord build_chunks.py")

    # À calculer : jamais calculé (None), ou texte modifié depuis (empreinte différente).
    todo = [(chunk_id, text) for chunk_id, text, stored in rows if stored != text_hash(text)]

    # Ménage : vecteurs de morceaux qui n'existent plus (par exemple après un build_chunks --rebuild).
    con.execute("DELETE FROM chunk_embeddings WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks)")

    print(f"Modèle : {MODEL_NAME}")
    print(f"{len(rows)} morceaux : {len(rows) - len(todo)} à jour, {len(todo)} à calculer")
    if not todo:
        return

    model = load_model()
    start = time.time()
    for first in range(0, len(todo), CHUNKS_PER_COMMIT):
        batch = todo[first:first + CHUNKS_PER_COMMIT]
        vectors = embed_passages(model, [text for _, text in batch])
        if vectors.shape != (len(batch), DIMENSIONS):
            sys.exit(f"Forme inattendue des vecteurs : {vectors.shape}")

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        data = [(chunk_id, MODEL_NAME, text_hash(text), vector.tolist(), now)
                for (chunk_id, text), vector in zip(batch, vectors)]
        try:
            con.begin()  # tout ou rien pour ce paquet
            # On supprime l'éventuel ancien vecteur (périmé) avant d'insérer le nouveau.
            con.executemany("DELETE FROM chunk_embeddings WHERE chunk_id = ? AND model = ?",
                            [(chunk_id, MODEL_NAME) for chunk_id, _ in batch])
            con.executemany("INSERT INTO chunk_embeddings VALUES (?, ?, ?, ?, ?)", data)
            con.commit()
        except Exception as e:
            con.rollback()
            sys.exit(f"ERREUR à partir du morceau {first} : {e}")

        done = first + len(batch)
        print(f"  {done}/{len(todo)} morceaux ({time.time() - start:.0f} s)")

    print(f"Terminé : {len(todo)} vecteurs calculés en {time.time() - start:.0f} s.")


if __name__ == "__main__":
    main()
