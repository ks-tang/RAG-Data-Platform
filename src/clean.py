"""Nettoyage du texte extrait des PDF."""
import re
import unicodedata
from collections import Counter

# Glyphes de puces fréquents + zone d'usage privé (où atterrissent les polices Dingbats/Symbol)
BULLETS = "•●▪■◦‣▶►◆❖✓✔➢➤"
PRIVATE_USE = "\uf000-\uf8ff"
LINE_START_BULLET = re.compile(rf"^[ \t]*[{re.escape(BULLETS)}{PRIVATE_USE}]+[ \t]*", re.MULTILINE)
# Dans certains PDF, la puce de liste est extraite comme un caractère ordinaire :
#   la lettre « I » (police Dingbats), le signe « > » ou la lettre « o » (puce de 2e niveau).
# Ces trois caractères ont été repérés avec survey_bullets.py : ils apparaissent en début de
# ligne, suivis d'un espace puis d'une lettre ("I Pertinence...", "> des topologies...",
# "o Connaître..."). Le « (?=...) » vérifie qu'une lettre suit, sans la consommer.
# On ne touche donc pas à "IGI 1300", "ISO" ou "oui" (pas d'espace après la lettre).
FAKE_BULLET = re.compile(r"^(?:I|>|o)[ \t]+(?=[^\W\d_])", re.MULTILINE)
STRAY_SYMBOLS = re.compile(rf"[{re.escape(BULLETS)}{PRIVATE_USE}]")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HYPHEN_BREAK = re.compile(r"(?<=[a-zà-ÿ])-[ \t]*\n[ \t]*(?=[a-zà-ÿ])")
COMPOUND_BREAK = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=[A-ZÀ-Þ0-9])")
SOFT_NEWLINE = re.compile(r"(?<!\n)\n(?!\n|- )")


def clean_text(text: str) -> str:
    """Nettoie le texte d'une page. Ne touche jamais à la donnée brute : renvoie une copie."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # espaces spéciaux (insécable, fine, césure optionnelle) -> espace normal ou rien
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\u00a0\u2007\u202f\u2009\u200a]", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS.sub("", text)
    # puces en début de ligne -> "- ", puces isolées ailleurs -> espace
    text = LINE_START_BULLET.sub("- ", text)
    text = STRAY_SYMBOLS.sub(" ", text)
    text = FAKE_BULLET.sub("- ", text)
    # mots coupés en fin de ligne : "sécu-\nrité" -> "sécurité"
    text = HYPHEN_BREAK.sub("", text)
    # tiret légitime coupé avant une majuscule/un chiffre : "ISO-\n27001" -> "ISO-27001"
    text = COMPOUND_BREAK.sub("-", text)
    # saut de ligne simple au milieu d'une phrase -> espace (on garde les paragraphes et les listes)
    text = SOFT_NEWLINE.sub(" ", text)
    # espaces multiples, espaces en fin de ligne, lignes vides multiples
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_key(line: str) -> str:
    """Clé de comparaison : minuscules, chiffres remplacés (« Page 3 » et « Page 4 » sont la même ligne)."""
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_repeated_lines(pages: list[str], min_ratio: float = 0.3,
                          min_pages: int = 5, max_len: int = 120) -> list[str]:
    """Retire les en-têtes/pieds de page : lignes courtes présentes sur au moins
    `min_ratio` des pages d'un même document. À appliquer AVANT clean_text
    (on a besoin de la structure en lignes)."""
    n = len(pages)
    if n < min_pages:
        return pages
    counts: Counter = Counter()
    for page in pages:
        keys = {_line_key(l) for l in page.splitlines() if l.strip() and len(l.strip()) <= max_len}
        counts.update(keys)
    threshold = min_ratio * n
    repeated = {k for k, c in counts.items() if c >= threshold}
    result = []
    for page in pages:
        kept = [l for l in page.splitlines()
                if not (l.strip() and len(l.strip()) <= max_len and _line_key(l) in repeated)]
        result.append("\n".join(kept))
    return result
