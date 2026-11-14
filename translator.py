"""
Module de traduction multilingue via Mistral AI.
- detect_language: detecte la langue d'un texte
- translate: traduit un texte vers une langue cible
- translate_to_many: traduit vers plusieurs langues d'un coup
"""

from config import get_mistral

MODEL = "mistral-small-latest"

# Langues supportees (codes ISO 639-1)
SUPPORTED_LANGUAGES = {
    "fr": "francais",
    "en": "anglais",
    "es": "espagnol",
    "de": "allemand",
    "it": "italien",
    "pt": "portugais",
    "ar": "arabe",
    "ru": "russe",
    "zh": "chinois",
    "ja": "japonais",
}


# Client unique
_client = get_mistral()


def detect_language(text: str) -> str:
    """
    Detecte la langue d'un texte (code ISO 639-1).
    Retourne 'en', 'fr', 'es', etc., ou 'unknown' si non detecte.
    """
    prompt = f"""Detecte la langue du texte suivant et reponds UNIQUEMENT avec le code ISO 639-1 sur 2 lettres (fr, en, es, de, it, pt, ar, ru, zh, ja...).
Ne donne aucune explication, juste les 2 lettres en minuscules.

Texte : "{text}"

Code langue :"""

    response = _client.chat.complete(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    code = response.choices[0].message.content.strip().lower()[:2]
    return code if code.isalpha() else "unknown"


def translate(text: str, target_lang: str, source_lang: str = None) -> str:
    """
    Traduit un texte vers une langue cible.
    target_lang et source_lang sont des codes ISO 639-1 (fr, en, es, ...).
    Si source_lang est None, la detection est automatique.
    """
    if target_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Langue cible non supportee : {target_lang}")

    target_name = SUPPORTED_LANGUAGES[target_lang]

    if source_lang and source_lang in SUPPORTED_LANGUAGES:
        source_name = SUPPORTED_LANGUAGES[source_lang]
        prompt = f"""Traduis le texte suivant du {source_name} vers le {target_name}.
Reponds UNIQUEMENT avec la traduction, sans aucune explication, ni guillemets, ni prefixe.

Texte original :
{text}

Traduction en {target_name} :"""
    else:
        prompt = f"""Traduis le texte suivant en {target_name}.
Reponds UNIQUEMENT avec la traduction, sans aucune explication, ni guillemets, ni prefixe.

Texte original :
{text}

Traduction en {target_name} :"""

    response = _client.chat.complete(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def translate_to_many(text: str, target_langs: list, source_lang: str = None) -> dict:
    """
    Traduit un texte vers plusieurs langues cibles.
    Retourne un dict {lang_code: traduction}.
    """
    return {
        lang: translate(text, lang, source_lang)
        for lang in target_langs
    }


# ---------- TEST ----------

if __name__ == "__main__":
    print("=== Test translator.py ===\n")

    # 1. Detection de langue
    print("--- Detection de langue ---")
    samples = [
        "Bonjour le monde",
        "Hello world",
        "Hola mundo",
        "Guten Tag",
    ]
    for s in samples:
        lang = detect_language(s)
        print(f"  '{s}' -> {lang}")
    print()

    # 2. Traduction simple
    print("--- Traduction simple ---")
    text = "L'intelligence artificielle change le monde."
    en = translate(text, "en", source_lang="fr")
    print(f"  FR: {text}")
    print(f"  EN: {en}\n")

    # 3. Traduction multilingue
    print("--- Traduction vers plusieurs langues ---")
    text = "Bienvenue dans notre application."
    translations = translate_to_many(text, ["en", "es", "de", "it"], source_lang="fr")
    print(f"  FR (source): {text}")
    for lang, trans in translations.items():
        print(f"  {lang.upper()}: {trans}")
    print()

    print("=== Tests OK ===")
