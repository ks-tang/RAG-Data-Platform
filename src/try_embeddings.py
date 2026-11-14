"""Test de l'environnement d'embeddings : 

Ce script ne touche pas à la base de données. Il compare une question à trois phrases
et affiche à quel point chacune lui ressemble (entre -1 et 1, plus c'est haut, plus c'est proche).
"""
import time

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

MODEL_NAME = "intfloat/multilingual-e5-small"
DIMENSIONS = 384

# fastembed ne connaît pas ce modèle d'office : on le déclare. Si une version récente le
# connaît déjà, la déclaration peut être refusée, et ce n'est pas grave.
try:
    TextEmbedding.add_custom_model(
        model=MODEL_NAME,
        pooling=PoolingType.MEAN,      # comment combiner les mots en un seul vecteur
        normalization=True,            # vecteurs de longueur 1 : similarité = produit scalaire
        sources=ModelSource(hf=MODEL_NAME),
        dim=DIMENSIONS,
        model_file="onnx/model.onnx",
    )
except ValueError as erreur:
    print(f"(déclaration ignorée : {erreur})")

print("Chargement du modèle (le premier lancement télécharge le modèle, ça peut prendre quelques minutes)...")
start = time.time()
model = TextEmbedding(model_name=MODEL_NAME)
print(f"Modèle prêt en {time.time() - start:.1f} s")

passages = [
    "Le zero trust impose de vérifier chaque demande d'accès, même depuis le réseau interne.",
    "La sauvegarde des systèmes d'information doit être testée régulièrement.",
    "Le durcissement consiste à réduire la surface d'attaque en désactivant les services inutiles.",
]
question = "Comment fonctionne le principe de confiance zéro ?"

# Les modèles E5 demandent un préfixe : « passage: » devant les textes à indexer,
# « query: » devant les questions.
start = time.time()
passage_vectors = np.array(list(model.embed(["passage: " + p for p in passages])))
query_vector = np.array(list(model.embed(["query: " + question])))[0]
print(f"Calcul de {len(passages) + 1} vecteurs en {time.time() - start:.2f} s")

print(f"\nForme des vecteurs : {passage_vectors.shape} (3 textes x {DIMENSIONS} dimensions attendues)")
print(f"Longueur d'un vecteur : {np.linalg.norm(query_vector):.3f} (doit être proche de 1)")

# Les vecteurs sont de longueur 1 : leur produit scalaire est leur similarité cosinus.
scores = passage_vectors @ query_vector
print(f"\nQuestion : {question}")
for score, passage in sorted(zip(scores, passages), reverse=True):
    print(f"  {score:.3f}  {passage}")