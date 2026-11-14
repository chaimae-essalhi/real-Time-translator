"""
Module de detection de plagiat.
Combine les embeddings Mistral et la recherche vectorielle Elasticsearch
pour detecter si un texte ressemble fortement a des documents deja indexes.

Strategie :
1. On genere l'embedding du texte a verifier.
2. On cherche dans Elasticsearch les K documents les plus proches (kNN cosinus).
3. On considere comme "plagiat" tout document depassant un seuil de similarite.

Note sur les seuils :
Mistral-embed donne des scores eleves meme pour des textes sans rapport (0.70-0.85).
Le vrai plagiat se distingue a partir de 0.92.
"""

from embeddings import get_embedding, cosine_similarity
from search import search_semantic, index_document
from storage import save_document, set_embedding

# Seuil par defaut : 0.92 = un texte est considere comme plagiat
# si sa similarite avec un document existant depasse 92%.
DEFAULT_THRESHOLD = 0.92


def check_plagiarism(text: str, threshold: float = DEFAULT_THRESHOLD, top_k: int = 5) -> dict:
    """
    Verifie si un texte ressemble fortement a des documents deja indexes.

    Retourne un dict :
    {
        "is_plagiarism": bool,
        "max_score": float,                # similarite la plus haute trouvee
        "threshold": float,                # seuil utilise
        "matches": [                       # tous les documents similaires
            {"doc_id", "text", "lang", "score", "is_match"},
            ...
        ]
    }
    """
    # 1. Embedding du texte d'entree
    query_embedding = get_embedding(text)

    # 2. Recherche des documents les plus proches dans Elasticsearch
    results = search_semantic(query_embedding, top_k=top_k)

    if not results:
        return {
            "is_plagiarism": False,
            "max_score": 0.0,
            "threshold": threshold,
            "matches": [],
        }

    # 3. Marquer ceux qui depassent le seuil
    for r in results:
        r["is_match"] = r["score"] >= threshold

    max_score = max(r["score"] for r in results)
    is_plagiarism = max_score >= threshold

    return {
        "is_plagiarism": is_plagiarism,
        "max_score": max_score,
        "threshold": threshold,
        "matches": results,
    }


def add_reference_document(text: str, lang: str = "auto", metadata: dict = None) -> str:
    """
    Ajoute un texte de reference a la base de plagiat :
    - Sauvegarde dans MongoDB
    - Genere son embedding
    - Indexe dans Elasticsearch

    Retourne l'ID Mongo du document.
    """
    # 1. Sauvegarde dans MongoDB
    doc_id = save_document(text=text, lang_source=lang, metadata=metadata or {})

    # 2. Generation de l'embedding
    embedding = get_embedding(text)
    set_embedding(doc_id, embedding)

    # 3. Indexation dans Elasticsearch
    index_document(doc_id=doc_id, text=text, lang=lang, embedding=embedding)

    return doc_id


# ---------- TEST ----------

if __name__ == "__main__":
    from search import create_index, delete_index
    from storage import delete_document as mongo_delete

    print("=== Test plagiarism.py ===\n")

    # On part d'une base propre
    delete_index()
    create_index()

    # 1. Ajout de documents de reference
    print("--- Ajout de documents de reference ---")
    references = [
        ("L'intelligence artificielle transforme notre societe moderne.", "fr"),
        ("Le rechauffement climatique est un defi mondial majeur.", "fr"),
        ("La cuisine italienne est appreciee dans le monde entier.", "fr"),
        ("Machine learning algorithms can recognize patterns in data.", "en"),
    ]
    added_ids = []
    for text, lang in references:
        doc_id = add_reference_document(text, lang)
        added_ids.append(doc_id)
        print(f"  + {text[:50]}... (id={doc_id[:8]}...)")
    print()

    # 2. Test : texte tres similaire (devrait etre detecte comme plagiat)
    print("--- Test 1 : texte tres similaire (devrait etre PLAGIAT) ---")
    suspect_1 = "L'IA est en train de transformer la societe moderne."
    print(f"  Texte verifie : '{suspect_1}'")
    result = check_plagiarism(suspect_1)
    print(f"  Plagiat detecte : {result['is_plagiarism']}")
    print(f"  Score max : {result['max_score']:.4f} (seuil : {result['threshold']})")
    print(f"  Meilleur match : '{result['matches'][0]['text']}'\n")

    # 3. Test : texte different (ne devrait PAS etre du plagiat)
    print("--- Test 2 : texte different (ne devrait PAS etre du plagiat) ---")
    suspect_2 = "J'aime jouer de la guitare le week-end."
    print(f"  Texte verifie : '{suspect_2}'")
    result = check_plagiarism(suspect_2)
    print(f"  Plagiat detecte : {result['is_plagiarism']}")
    print(f"  Score max : {result['max_score']:.4f} (seuil : {result['threshold']})")
    print(f"  Meilleur match : '{result['matches'][0]['text']}'\n")

    # 4. Test : meme phrase en anglais (cross-lingual, embeddings multilingues)
    print("--- Test 3 : meme idee en autre langue (devrait etre PLAGIAT cross-lingual) ---")
    suspect_3 = "AI is transforming modern society."
    print(f"  Texte verifie : '{suspect_3}'")
    result = check_plagiarism(suspect_3)
    print(f"  Plagiat detecte : {result['is_plagiarism']}")
    print(f"  Score max : {result['max_score']:.4f} (seuil : {result['threshold']})")
    print(f"  Meilleur match : '{result['matches'][0]['text']}'\n")

    # 5. Nettoyage MongoDB (les docs de test)
    for doc_id in added_ids:
        mongo_delete(doc_id)

    print("=== Tests OK ===")
