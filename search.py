"""
Gestion de l'index Elasticsearch.
- Cree un index avec support pour la recherche full-text ET vectorielle (kNN).
- Indexe, recherche, supprime des documents.

Structure d'un doc indexe :
{
    "doc_id": "id_mongo",
    "text": "Bonjour le monde",
    "lang": "fr",
    "embedding": [1024 floats]   # pour la recherche semantique (Mistral)
}
"""

from config import get_elasticsearch

INDEX_NAME = "translations"
EMBEDDING_DIM = 1024  # dimension des embeddings Mistral

es = get_elasticsearch()


# ---------- INDEX MANAGEMENT ----------

def create_index(force: bool = False):
    """
    Cree l'index Elasticsearch s'il n'existe pas.
    Si force=True, supprime l'index existant puis le recree.
    """
    if es.indices.exists(index=INDEX_NAME):
        if force:
            es.indices.delete(index=INDEX_NAME)
            print(f"[INFO] Index '{INDEX_NAME}' supprime.")
        else:
            print(f"[INFO] Index '{INDEX_NAME}' existe deja.")
            return

    mapping = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": "standard",
                },
                "lang": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"[OK] Index '{INDEX_NAME}' cree (dim={EMBEDDING_DIM}).")


def delete_index():
    """Supprime completement l'index (pour reset)."""
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"[OK] Index '{INDEX_NAME}' supprime.")


# ---------- INDEX DOCUMENTS ----------

def index_document(doc_id: str, text: str, lang: str, embedding: list = None):
    """Ajoute (ou met a jour) un document dans l'index."""
    body = {"doc_id": doc_id, "text": text, "lang": lang}
    if embedding is not None:
        body["embedding"] = embedding
    es.index(index=INDEX_NAME, id=doc_id, document=body, refresh=True)


# ---------- SEARCH ----------

def search_text(query: str, top_k: int = 10) -> list:
    """Recherche full-text classique (basee sur les mots)."""
    response = es.search(
        index=INDEX_NAME,
        query={"match": {"text": query}},
        size=top_k,
    )
    results = []
    for hit in response["hits"]["hits"]:
        item = dict(hit["_source"])
        item["score"] = hit["_score"]
        results.append(item)
    return results


def search_semantic(query_embedding: list, top_k: int = 10) -> list:
    """Recherche semantique (kNN) basee sur les embeddings."""
    response = es.search(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 10,
        },
        size=top_k,
    )
    results = []
    for hit in response["hits"]["hits"]:
        item = dict(hit["_source"])
        item["score"] = hit["_score"]
        results.append(item)
    return results


def delete_document(doc_id: str):
    """Supprime un document de l'index."""
    if es.exists(index=INDEX_NAME, id=doc_id):
        es.delete(index=INDEX_NAME, id=doc_id, refresh=True)


# ---------- TEST ----------

if __name__ == "__main__":
    print("=== Test search.py ===\n")

    # 1. Creation de l'index
    create_index(force=True)

    # 2. Indexation de quelques documents (sans embedding pour ce test)
    docs = [
        ("doc1", "Le chat dort sur le canape", "fr"),
        ("doc2", "The cat sleeps on the couch", "en"),
        ("doc3", "Les oiseaux chantent dans les arbres", "fr"),
        ("doc4", "Birds sing in the trees", "en"),
    ]
    for doc_id, text, lang in docs:
        index_document(doc_id, text, lang)
    print(f"[OK] {len(docs)} documents indexes\n")

    # 3. Recherche full-text
    print("--- Recherche : 'chat' ---")
    results = search_text("chat")
    for r in results:
        print(f"  [{r['score']:.2f}] ({r['lang']}) {r['text']}")
    print()

    print("--- Recherche : 'birds trees' ---")
    results = search_text("birds trees")
    for r in results:
        print(f"  [{r['score']:.2f}] ({r['lang']}) {r['text']}")
    print()

    print("=== Tests OK ===")
