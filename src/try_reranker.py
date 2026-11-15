"""Test isolé du reranker : se charge-t-il, et quelle mémoire utilise-t-il ?

Ce script n'utilise PAS la base de données. Il affiche la mémoire de la machine, puis celle du
programme à chaque étape, pour repérer l'étape qui pose problème si le programme est arrêté.
Usage : python src/try_reranker.py
"""
import time

from retrieval import RERANK_MODEL, load_reranker, rerank


def machine_memory() -> str:
    """Mémoire totale et disponible de la machine (lue dans le fichier système /proc/meminfo)."""
    values = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, rest = line.split(":", 1)
            values[key] = int(rest.split()[0]) / 1024  # converti de kilooctets en mégaoctets
    return f"total {values['MemTotal']:.0f} Mo, disponible {values['MemAvailable']:.0f} Mo"


def program_memory() -> str:
    """Mémoire utilisée par CE programme : actuelle (VmRSS) et maximale atteinte (VmHWM)."""
    values = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM")):
                key, rest = line.split(":", 1)
                values[key] = int(rest.split()[0]) / 1024
    return f"actuelle {values['VmRSS']:.0f} Mo, maximale {values['VmHWM']:.0f} Mo"


def step(label: str):
    print(f"[{label}] programme : {program_memory()} | machine : {machine_memory()}", flush=True)


print(f"Modèle : {RERANK_MODEL}", flush=True)
step("départ")

start = time.time()
reranker = load_reranker()
print(f"Chargement : {time.time() - start:.1f} s", flush=True)
step("modèle chargé")

question = "Quels sont les risques spécifiques de l'IA agentique ?"
passages = [
    "Les systèmes agentiques agissent de façon autonome et enchaînent des actions via des outils et des connecteurs.",
    "La sauvegarde des systèmes d'information doit être testée régulièrement.",
    "L'IA générative expose à des fuites d'informations sensibles et à des attaques par manipulation.",
]
start = time.time()
scores = list(reranker.rerank(question, passages))
print(f"3 textes courts notés en {time.time() - start:.1f} s : {[round(float(s), 2) for s in scores]}", flush=True)
step("après 3 textes courts")

# Cas réaliste : 20 morceaux d'environ 800 caractères, comme dans la recherche.
long_text = (passages[0] + " ") * 8
ranked = [(0.0, (f"c{i}", "doc.pdf", 1, 1, long_text[:800] + f" {i}", [])) for i in range(20)]
start = time.time()
rerank(reranker, question, ranked, 5)
print(f"20 morceaux de 800 caractères notés en {time.time() - start:.1f} s", flush=True)
step("après 20 morceaux")
