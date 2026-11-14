"""
Gestion de la persistance MongoDB.
Stocke les documents originaux et leurs traductions.

Structure d'un document dans la collection 'documents' :
{
    "_id": ObjectId,
    "text": "Bonjour le monde",
    "lang_source": "fr",
    "translations": {
        "en": "Hello world",
        "es": "Hola mundo"
    },
    "embedding": [0.123, -0.456, ...],   # ajoute plus tard
    "created_at": datetime,
    "metadata": { ... }                   # libre
}
"""

from datetime import datetime
from bson import ObjectId
from config import get_mongo


# Connexion unique a la base
_db = get_mongo()
_documents = _db["documents"]


# ---------- CREATE ----------

def save_document(text: str, lang_source: str = "auto", metadata: dict = None) -> str:
    """
    Sauvegarde un nouveau document.
    Retourne son ID MongoDB (sous forme de string).
    """
    doc = {
        "text": text,
        "lang_source": lang_source,
        "translations": {},
        "embedding": None,
        "created_at": datetime.utcnow(),
        "metadata": metadata or {},
    }
    result = _documents.insert_one(doc)
    return str(result.inserted_id)


# ---------- READ ----------

def get_document(doc_id: str):
    """Recupere un document par son ID."""
    doc = _documents.find_one({"_id": ObjectId(doc_id)})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def list_documents(limit: int = 100) -> list:
    """Liste les documents recents (du plus recent au plus ancien)."""
    cursor = _documents.find().sort("created_at", -1).limit(limit)
    docs = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        docs.append(doc)
    return docs


# ---------- UPDATE ----------

def add_translation(doc_id: str, target_lang: str, translated_text: str) -> bool:
    """Ajoute une traduction a un document existant."""
    result = _documents.update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": {f"translations.{target_lang}": translated_text}}
    )
    return result.modified_count > 0


def set_embedding(doc_id: str, embedding: list) -> bool:
    """Stocke le vecteur embedding d'un document."""
    result = _documents.update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": {"embedding": embedding}}
    )
    return result.modified_count > 0


# ---------- DELETE ----------

def delete_document(doc_id: str) -> bool:
    """Supprime un document."""
    result = _documents.delete_one({"_id": ObjectId(doc_id)})
    return result.deleted_count > 0


# ---------- TEST ----------

if __name__ == "__main__":
    print("=== Test storage.py ===\n")

    # 1. Creation
    new_id = save_document(
        text="Bonjour le monde",
        lang_source="fr",
        metadata={"source": "test"}
    )
    print(f"[OK] Document cree : {new_id}")

    # 2. Lecture
    doc = get_document(new_id)
    print(f"[OK] Document lu : {doc}\n")

    # 3. Ajout d'une traduction
    add_translation(new_id, "en", "Hello world")
    add_translation(new_id, "es", "Hola mundo")
    doc = get_document(new_id)
    print(f"[OK] Apres traductions : {doc['translations']}\n")

    # 4. Liste
    all_docs = list_documents(limit=5)
    print(f"[OK] {len(all_docs)} document(s) dans la base")

    # 5. Suppression (nettoyage)
    delete_document(new_id)
    print(f"[OK] Document supprime\n")

    print("=== Tous les tests OK ===")
