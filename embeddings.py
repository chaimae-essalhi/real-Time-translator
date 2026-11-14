"""
Module de generation d'embeddings vectoriels via Mistral AI.
- get_embedding: retourne le vecteur d'un texte (1024 dimensions)
- get_embeddings_batch: vectorise plusieurs textes en un seul appel
- cosine_similarity: calcule la similarite entre 2 vecteurs

Inclut un retry automatique sur les erreurs 429 (service surcharge),
frequentes avec le free tier de mistral-embed.
"""

import time
import numpy as np
from config import get_mistral

EMBEDDING_MODEL = "mistral-embed"
EMBEDDING_DIM = 1024

_client = get_mistral()


def _call_with_retry(fn, max_retries: int = 6, base_delay: float = 2.0):
    """
    Execute une fonction d'appel API avec retry automatique en cas d'erreur 429.
    Attend de plus en plus longtemps a chaque echec (backoff exponentiel).
    Delais : 2s, 4s, 8s, 16s, 32s, puis abandon.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            is_rate_limit = (
                "429" in msg
                or "capacity_exceeded" in msg
                or "rate" in msg.lower()
                or "service_tier_capacity" in msg
            )
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  [WARN] Mistral surcharge, nouvel essai dans {delay:.0f}s... ({attempt+1}/{max_retries})")
                time.sleep(delay)
                continue
            raise


def get_embedding(text: str) -> list:
    """
    Genere le vecteur embedding d'un texte (1024 floats).
    Avec retry automatique en cas de surcharge du service.
    """
    def _do():
        return _client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=[text],
        )
    response = _call_with_retry(_do)
    return response.data[0].embedding


def get_embeddings_batch(texts: list) -> list:
    """
    Genere les embeddings de plusieurs textes en un seul appel API.
    Retourne une liste de vecteurs dans le meme ordre que les textes.
    Avec retry automatique en cas de surcharge.
    """
    def _do():
        return _client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=texts,
        )
    response = _call_with_retry(_do)
    return [item.embedding for item in response.data]


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calcule la similarite cosinus entre 2 vecteurs.
    Retourne une valeur entre -1 et 1 (1 = identiques, 0 = sans rapport).
    """
    a = np.array(vec1)
    b = np.array(vec2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------- TEST ----------

if __name__ == "__main__":
    print("=== Test embeddings.py ===\n")

    # 1. Embedding simple
    print("--- Embedding d'un texte ---")
    text = "Le chat dort sur le canape."
    emb = get_embedding(text)
    print(f"  Texte : '{text}'")
    print(f"  Vecteur de dimension {len(emb)}")
    print(f"  5 premiers floats : {[round(x, 4) for x in emb[:5]]}\n")

    # 2. Batch embeddings
    print("--- Batch embeddings ---")
    sentences = [
        "Un chat dort sur le canape.",
        "Le felin se repose sur le sofa.",
        "Le president a tenu un discours important.",
    ]
    embs = get_embeddings_batch(sentences)
    for s, e in zip(sentences, embs):
        print(f"  '{s}' -> vecteur dim {len(e)}")
    print()

    # 3. Test de similarite (deux phrases proches vs une eloignee)
    print("--- Similarite cosinus ---")
    sim_proches = cosine_similarity(embs[0], embs[1])
    sim_eloignees = cosine_similarity(embs[0], embs[2])
    print(f"  Phrase 1 vs Phrase 2 (proches)    : {sim_proches:.4f}")
    print(f"  Phrase 1 vs Phrase 3 (eloignees)  : {sim_eloignees:.4f}")
    print()

    print("=== Tests OK ===")
