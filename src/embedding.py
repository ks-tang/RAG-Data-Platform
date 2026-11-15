"""Tout ce qui concerne le modèle d'embedding, réuni à un seul endroit.

Un « embedding » transforme un texte en une liste de nombres (un vecteur) qui représente son sens.
Deux textes qui parlent de la même chose ont des vecteurs proches, même avec des mots différents.
La « proximité » se mesure par la similarité cosinus : proche de 1 = très proche.

Ce fichier est utilisé à la fois par embed_chunks.py (qui calcule les vecteurs des morceaux)
et par search.py (qui calcule le vecteur de la question). Les deux DOIVENT utiliser le même
modèle, sinon leurs vecteurs ne sont pas comparables.
"""
import hashlib

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
DIMENSIONS = 384

# Les modèles E5 ont été entraînés avec un préfixe devant chaque texte : « passage: » pour les
# textes dans lesquels on cherche, « query: » pour la question. Sans lui, la qualité baisse.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

BATCH_SIZE = 32  # nombre de textes traités à la fois par le modèle


def text_hash(text: str) -> str:
    """Empreinte d'un texte. Si le texte d'un morceau change, son embedding doit être recalculé."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_model():
    """Charge le modèle (le premier appel le télécharge, les suivants le lisent sur le disque)."""
    # Import à l'intérieur de la fonction : les autres fonctions de ce fichier restent utilisables
    # (et testables) sans charger la bibliothèque, qui est lourde.
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    # fastembed ne connaît pas ce modèle d'office : on le déclare. Si une version récente le
    # connaît déjà, la déclaration peut être refusée, et ce n'est pas grave.
    try:
        TextEmbedding.add_custom_model(
            model=MODEL_NAME,
            pooling=PoolingType.MEAN,   # comment combiner les mots en un seul vecteur
            normalization=True,         # vecteurs de longueur 1 : similarité cosinus = produit scalaire
            sources=ModelSource(hf=MODEL_NAME),
            dim=DIMENSIONS,
            model_file="onnx/model.onnx",
        )
    except ValueError:
        pass
    return TextEmbedding(model_name=MODEL_NAME)


def embed_passages(model, texts):
    """Vecteurs de plusieurs morceaux. Renvoie un tableau de forme (nombre_de_textes, 384)."""
    vectors = model.embed([PASSAGE_PREFIX + t for t in texts], batch_size=BATCH_SIZE)
    return np.array(list(vectors), dtype=np.float32)


def embed_query(model, question: str):
    """Vecteur d'une question. Renvoie un tableau de forme (384,)."""
    vectors = model.embed([QUERY_PREFIX + question])
    return np.array(list(vectors)[0], dtype=np.float32)


def top_matches(matrix, query_vector, k: int):
    """Les k vecteurs les plus proches de la question.

    `matrix` : un vecteur par ligne. `query_vector` : le vecteur de la question.
    Les vecteurs ayant tous une longueur de 1, le produit scalaire EST la similarité cosinus.
    Renvoie une liste de (position dans la matrice, score), du plus proche au moins proche.
    """
    scores = matrix @ query_vector
    k = min(k, len(scores))
    best = np.argsort(-scores)[:k]  # le « - » inverse l'ordre : les plus grands scores d'abord
    return [(int(i), float(scores[i])) for i in best]
