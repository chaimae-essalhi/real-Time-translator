"""
Configuration centrale du projet.
Charge les secrets depuis .env et expose les clients prets a l'emploi.
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from elasticsearch import Elasticsearch

# Charger les variables du fichier .env
load_dotenv()

# ---------- Variables d'environnement ----------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "translator")


# ---------- Clients reutilisables ----------

def get_mongo():
    """Retourne la base MongoDB du projet."""
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


def get_elasticsearch():
    """Retourne un client Elasticsearch en HTTP simple."""
    return Elasticsearch(
        "http://localhost:9200",
        verify_certs=False,
        ssl_show_warn=False,
    )


def get_mistral():
    """Retourne un client Mistral AI authentifie."""
    from mistralai import Mistral
    return Mistral(api_key=MISTRAL_API_KEY)
