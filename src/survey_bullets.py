"""Cherche dans le texte BRUT les débuts de ligne qui ressemblent à des puces mal extraites.

Principe : une puce de liste est parfois extraite comme UN SEUL caractère suivi d'un espace
(par exemple « I Pertinence des attributs »). On repère tous les débuts de ligne de ce type.

Ce script lit la base en lecture seule : il ne modifie rien.

Deux modes d'utilisation :
    python src/survey_bullets.py
        -> tableau récapitulatif : les caractères les plus fréquents en début de ligne
    python src/survey_bullets.py I U+003E o
        -> détail pour certains caractères (écrits tels quels, ou par leur code « U+XXXX »).
           Pour le signe « > », utilisez son code U+003E : tapé dans le terminal, il
           serait pris pour une redirection.
"""
import re
import sys
from collections import Counter, defaultdict

import duckdb

DB_PATH = "data/warehouse.duckdb"
TOP = 15              # nombre de caractères affichés dans le tableau récapitulatif
EXAMPLES_PER_KIND = 3  # exemples affichés par catégorie dans le mode détail

# ^           début de ligne (grâce à re.MULTILINE, chaque ligne du texte compte)
# (\S)        UN seul caractère qui n'est pas un espace : c'est lui qu'on étudie
# [ ]         suivi d'un espace
# (\S.{0,60}) puis le début de la suite de la ligne, pour avoir un exemple lisible
PATTERN = re.compile(r"^(\S) (\S.{0,60})", re.MULTILINE)


def kind_of_next_char(rest: str) -> str:
    """Décrit le caractère qui suit la puce : majuscule, minuscule, chiffre ou autre."""
    first = rest[0]
    if first.isupper():
        return "majuscule"
    if first.islower():
        return "minuscule"
    if first.isdigit():
        return "chiffre"
    return "autre"


def collect(rows):
    """Parcourt les pages et renvoie la liste (caractère, fichier, début de la suite de la ligne)."""
    found = []
    for filename, text in rows:
        for match in PATTERN.finditer(text or ""):
            found.append((match.group(1), filename, match.group(2)))
    return found


def parse_argument(arg: str) -> str:
    """Accepte « I » ou « U+0049 »."""
    if arg.upper().startswith("U+"):
        return chr(int(arg[2:], 16))
    return arg


def print_summary(found):
    counts = Counter(char for char, _, _ in found)
    documents = defaultdict(set)
    examples = {}
    for char, filename, rest in found:
        documents[char].add(filename)
        examples.setdefault(char, rest)
    print(f"{'caractère':<10}{'code':<9}{'lignes':>7}{'docs':>6}   exemple")
    for char, n in counts.most_common(TOP):
        print(f"{char!r:<10}U+{ord(char):04X}  {n:>7}{len(documents[char]):>6}   {examples[char][:40]!r}")


def print_detail(found, chars):
    for char in chars:
        selection = [(f, r) for c, f, r in found if c == char]
        print("\n" + "=" * 70)
        print(f"CARACTÈRE {char!r} (U+{ord(char):04X}) : {len(selection)} lignes")
        print("=" * 70)
        by_kind = defaultdict(list)
        for filename, rest in selection:
            by_kind[kind_of_next_char(rest)].append((filename, rest))
        for kind, items in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
            print(f"\n  Suivi d'une {kind} : {len(items)} lignes, {len({f for f, _ in items})} document(s)")
            seen_docs = set()
            shown = 0
            for filename, rest in items:  # un exemple par document, pour varier
                if filename in seen_docs:
                    continue
                seen_docs.add(filename)
                print(f"    {filename[:38]:<38} {rest!r}")
                shown += 1
                if shown >= EXAMPLES_PER_KIND:
                    break


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    rows = con.execute("""
        SELECT d.filename, p.text
        FROM pages p JOIN documents d USING (doc_id)""").fetchall()
    found = collect(rows)
    if len(sys.argv) > 1:
        print_detail(found, [parse_argument(a) for a in sys.argv[1:]])
    else:
        print_summary(found)


if __name__ == "__main__":
    main()
