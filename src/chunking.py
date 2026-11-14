"""Découpage (« chunking ») d'un document en morceaux pour la recherche sémantique.

Pourquoi découper ?
    Un moteur de recherche sémantique ne compare pas une question à un document entier
    (trop long, trop de sujets mélangés) : il la compare à de petits morceaux de texte.
    Chaque morceau devient plus tard un « embedding » (une suite de nombres qui représente
    son sens). Un bon découpage donne des morceaux qui ont du sens à eux seuls.

Deux stratégies sont proposées :
    "fixed"     : taille fixe, avec un chevauchement entre deux morceaux consécutifs.
                  Simple et prévisible, mais peut couper au milieu d'une phrase.
    "sentences" : on regroupe des phrases entières jusqu'à une taille maximale.
                  Les morceaux se terminent presque toujours par une fin de phrase.

Dans tout ce fichier, une « span » est un couple (début, fin) de positions dans le texte :
    texte[début:fin] donne le morceau. On travaille avec des positions plutôt qu'avec
    des copies de texte pour pouvoir retrouver à quelle page se trouve chaque morceau.
"""
import re
from bisect import bisect_right
from dataclasses import dataclass

# --- Réglages par défaut (à modifier pour tester d'autres valeurs) ---------------
DEFAULT_SIZE = 800            # taille maximale d'un morceau, en caractères (environ 130 mots)
DEFAULT_OVERLAP = 100         # caractères répétés entre deux morceaux (stratégie "fixed")
DEFAULT_OVERLAP_SENTENCES = 1  # phrases répétées entre deux morceaux (stratégie "sentences")

# Une page qui se termine par un de ces signes finit « proprement » une phrase.
SENTENCE_END = (".", "!", "?", "…", ":")

# Où une phrase se termine-t-elle ? Trois cas, séparés par le signe « | » (« ou ») :
#   1. un point (ou ! ? …) suivi d'espaces puis d'une majuscule, d'un guillemet ou d'une parenthèse
#   2. une ligne vide (nouveau paragraphe)
#   3. un retour à la ligne suivi d'une puce « - » (début d'un élément de liste)
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?…])\s+(?=[A-ZÀ-Þ«\"(])"
    r"|\n{2,}"
    r"|\n(?=- )"
)


@dataclass(frozen=True)
class Chunk:
    """Un morceau de document, avec de quoi retrouver d'où il vient."""
    index: int        # numéro du morceau dans le document (0, 1, 2, ...)
    text: str         # le texte du morceau
    start: int        # position de début dans le texte complet du document
    end: int          # position de fin dans le texte complet du document
    page_start: int   # numéro de la page où le morceau commence
    page_end: int     # numéro de la page où le morceau se termine


def _trim_span(text: str, start: int, end: int):
    """Retire les espaces au début et à la fin d'une span. Renvoie None si elle est vide."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def build_document_text(pages):
    """Assemble les pages d'un document en un seul texte.

    `pages` est une liste de couples (numéro_de_page, texte).
    Renvoie (texte_complet, positions_de_début_des_pages, numéros_de_pages), ce qui permettra
    de retrouver la page d'un morceau à partir de sa position.

    Entre deux pages : une ligne vide si la page précédente finit une phrase, sinon un simple
    espace (la phrase continue sur la page suivante et ne doit pas être coupée).
    """
    parts = []
    page_starts = []
    page_numbers = []
    position = 0
    previous_text = ""
    for page_number, page_text in pages:
        page_text = (page_text or "").strip()
        if not page_text:
            continue  # page vide : rien à ajouter
        if parts:
            separator = "\n\n" if previous_text.endswith(SENTENCE_END) else " "
            parts.append(separator)
            position += len(separator)
        page_starts.append(position)
        page_numbers.append(page_number)
        parts.append(page_text)
        position += len(page_text)
        previous_text = page_text
    return "".join(parts), page_starts, page_numbers


def _page_at(offset: int, page_starts, page_numbers) -> int:
    """Numéro de la page qui contient la position `offset` (recherche dichotomique)."""
    i = bisect_right(page_starts, offset) - 1
    return page_numbers[max(i, 0)]


# ------------------------------------------------------------------------------
# Stratégie 1 : taille fixe avec chevauchement
# ------------------------------------------------------------------------------
def split_fixed_size(text: str, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP):
    """Découpe en morceaux d'au plus `size` caractères.

    - On ne coupe jamais au milieu d'un mot : on recule jusqu'au dernier espace.
    - Deux morceaux consécutifs partagent environ `overlap` caractères, pour qu'une
      information située à la frontière apparaisse en entier dans au moins un morceau.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Il faut size > 0 et 0 <= overlap < size.")

    spans = []
    length = len(text)
    start = 0
    while start < length:
        end = min(start + size, length)
        if end < length:
            # On cherche le dernier espace ou retour à la ligne dans la 2e moitié de la fenêtre.
            half = start + size // 2
            cut = max(text.rfind(" ", half, end), text.rfind("\n", half, end))
            if cut > start:
                end = cut
        span = _trim_span(text, start, end)
        if span:
            spans.append(span)
        if end >= length:
            break
        # Le morceau suivant recule de `overlap` caractères (et avance toujours d'au moins 1).
        next_start = max(end - overlap, start + 1)
        # S'il tombe au milieu d'un mot, on avance jusqu'au début du mot suivant.
        if not text[next_start - 1].isspace():
            space = re.compile(r"\s").search(text, next_start, end)
            if space:
                next_start = space.end()
        start = next_start
    return spans


# ------------------------------------------------------------------------------
# Stratégie 2 : phrases entières regroupées
# ------------------------------------------------------------------------------
def split_sentences(text: str):
    """Découpe le texte en phrases (renvoie des spans)."""
    spans = []
    position = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        span = _trim_span(text, position, match.start())
        if span:
            spans.append(span)
        position = match.end()
    span = _trim_span(text, position, len(text))
    if span:
        spans.append(span)
    return spans


def split_by_sentences(text: str, size: int = DEFAULT_SIZE,
                       overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES):
    """Regroupe des phrases entières en morceaux d'au plus `size` caractères.

    Chaque nouveau morceau reprend les `overlap_sentences` dernières phrases du précédent
    (si cela ne le fait pas dépasser `size`), pour garder un peu de contexte.
    """
    if size <= 0 or overlap_sentences < 0:
        raise ValueError("Il faut size > 0 et overlap_sentences >= 0.")

    # 1) Découpe en phrases. Une phrase plus longue que `size` (liste mal ponctuée, tableau...)
    #    est elle-même découpée en tranches de taille fixe, pour respecter la limite.
    sentences = []
    for s, e in split_sentences(text):
        if e - s <= size:
            sentences.append((s, e))
        else:
            for a, b in split_fixed_size(text[s:e], size=size, overlap=0):
                sentences.append((s + a, s + b))

    # 2) Regroupement. `new` = indice de la première phrase pas encore dans un morceau.
    spans = []
    new = 0
    while new < len(sentences):
        # Phrases reprises du morceau précédent (chevauchement)...
        first = max(new - overlap_sentences, 0)
        # ... abandonnées une à une si elles empêchent d'ajouter la phrase nouvelle.
        while first < new and sentences[new][1] - sentences[first][0] > size:
            first += 1
        # On ajoute des phrases tant que le morceau ne dépasse pas `size`.
        last = new
        while last + 1 < len(sentences) and sentences[last + 1][1] - sentences[first][0] <= size:
            last += 1
        spans.append((sentences[first][0], sentences[last][1]))
        new = last + 1
    return spans


# ------------------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------------------
STRATEGIES = {
    "fixed": split_fixed_size,
    "sentences": split_by_sentences,
}


def chunk_document(pages, strategy: str = "sentences", **options):
    """Découpe un document en objets Chunk.

    `pages` : liste de (numéro_de_page, texte_nettoyé).
    `strategy` : "fixed" ou "sentences".
    `options` : réglages propres à la stratégie, par exemple size=1000, overlap=150.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Stratégie inconnue : {strategy!r}. Choix : {sorted(STRATEGIES)}")
    text, page_starts, page_numbers = build_document_text(pages)
    spans = STRATEGIES[strategy](text, **options)
    return [
        Chunk(
            index=i,
            text=text[start:end],
            start=start,
            end=end,
            page_start=_page_at(start, page_starts, page_numbers),
            page_end=_page_at(end - 1, page_starts, page_numbers),
        )
        for i, (start, end) in enumerate(spans)
    ]


def ends_like_a_sentence(chunk_text: str) -> bool:
    """Vrai si le morceau se termine par une fin de phrase (indicateur de qualité du découpage)."""
    return chunk_text.rstrip().endswith(SENTENCE_END + (";", ")", "»", '"'))
