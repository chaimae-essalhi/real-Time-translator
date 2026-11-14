"""Verifie que MongoDB, Elasticsearch et Mistral AI sont accessibles."""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("TEST DE L'ENVIRONNEMENT - Real Time Translator")
print("=" * 60)

# 1. MongoDB
print("\n[1/3] Test MongoDB...")
try:
    from pymongo import MongoClient
    client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    print(f"    OK - bases : {client.list_database_names()}")
except Exception as e:
    print(f"    ERREUR : {e}")

# 2. Elasticsearch
print("\n[2/3] Test Elasticsearch...")
try:
    from elasticsearch import Elasticsearch
    es = Elasticsearch(
        os.getenv("ES_HOST"),
        basic_auth=(os.getenv("ES_USER"), os.getenv("ES_PASSWORD")),
        verify_certs=False,
        ssl_show_warn=False
    )
    info = es.info()
    print(f"    OK - version {info['version']['number']}")
except Exception as e:
    print(f"    ERREUR : {e}")

# 3. Mistral AI
print("\n[3/3] Test Mistral AI...")
try:
    from mistralai import Mistral
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "Traduis en anglais : Bonjour le monde"}]
    )
    print(f"    OK - reponse : {response.choices[0].message.content.strip()}")
except Exception as e:
    print(f"    ERREUR : {e}")

print("\n" + "=" * 60)
