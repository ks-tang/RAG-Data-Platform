# RAG-Data-Platform

Pipeline de données qui prépare un corpus de documents techniques (guides publics, PDF) pour une architecture RAG : ingestion, nettoyage, métadonnées, contrôles qualité.

## Pipeline

```
data/raw/*.pdf --ingest.py--> documents, pages          (données brutes, jamais modifiées)
               --transform.py--> pages_clean            (texte nettoyé)
               --enrich.py--> document_metadata         (titre, langue, année, source, classification)
               --check_quality.py--> contrôles (code de sortie 1 si échec)
```

Base : DuckDB (`data/warehouse.duckdb`, recréée par le pipeline, non versionnée).

## Installation et lancement

```bash
pip install -r requirements.txt
# déposer les PDF dans data/raw/ (non versionnés, voir « Données »)
bash run_pipeline.sh
pytest            # tests unitaires du nettoyage
```

Pour recalculer le texte nettoyé après une modification de `src/clean.py` : `python src/transform.py --rebuild`.

## Données

Guides publics de l'ANSSI (cyber.gouv.fr), à télécharger dans `data/raw/`. Les PDF ne sont pas versionnés : on versionne le code, pas les données.

## Choix de conception

- **Idempotence** : chaque étape peut être relancée sans créer de doublons (identifiant = empreinte SHA-256 du fichier).
- **Donnée brute immuable** : le texte nettoyé et les métadonnées sont dans des tables séparées, ce qui permet de tout recalculer en cas de bug.
- **Robustesse** : un PDF illisible est signalé et ignoré, sans bloquer les autres ; chaque document est transformé dans une transaction (tout ou rien).
- **Traçabilité** : chaque page est reliée à son document par `doc_id`.

## Règles de nettoyage (`src/clean.py`)

- mots coupés en fin de ligne recollés (« sécu-/rité » -> « sécurité ») ; un tiret suivi d'une majuscule ou d'un chiffre est conservé (« ISO-27001 ») ;
- sauts de ligne au milieu d'une phrase remplacés par un espace ; paragraphes et listes conservés ;
- espaces multiples et caractères de contrôle supprimés ;
- puces de polices symboliques (Dingbats) converties en « - » ;
- en-têtes et pieds de page retirés : lignes courtes présentes sur au moins 30 % des pages d'un document de 5 pages ou plus (les numéros de page sont neutralisés avant comparaison).

**Limite connue** : le recollage des mots coupés ne distingue pas un mot coupé d'un mot composé écrit en minuscules (« peut-/être »).

## Classification (fictive)

Les documents du corpus sont tous publics. Pour préparer le test des droits d'accès, une classification fictive est attribuée selon la taille : moins de 20 pages = Public, 20 à 80 pages = Interne, plus de 80 pages = Confidentiel.

## Contrôles qualité

Pages nettoyées présentes pour chaque page brute, aucun texte NULL, pas de doublons, pas plus de 30 % du texte perdu au nettoyage, langue reconnue, classification valide.

## Prochaines étapes

Chunking, embeddings et recherche sémantique, filtrage par niveau de classification, CI/CD (GitHub Actions), évaluation du RAG.
