"""
Module OCR (Optical Character Recognition) avec Tesseract.
Extrait le texte d'une image pour ensuite le rechercher ou le traduire.

Necessite l'installation de Tesseract-OCR :
https://github.com/UB-Mannheim/tesseract/wiki

Par defaut, on cherche Tesseract a 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'.
Si installe ailleurs, modifier la variable TESSERACT_PATH.
"""

import os
import pytesseract
from PIL import Image

# Chemin par defaut sur Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# Langues supportees par Tesseract (codes Tesseract != codes ISO)
TESSERACT_LANGS = {
    "fr": "fra",     # Francais
    "en": "eng",     # Anglais
    "es": "spa",     # Espagnol
    "de": "deu",     # Allemand
    "it": "ita",     # Italien
    "pt": "por",     # Portugais
    "ar": "ara",     # Arabe
    "ru": "rus",     # Russe
}


def extract_text_from_image(image_path_or_pil, langs: list = None) -> str:
    """
    Extrait le texte d'une image.

    Parametres :
    - image_path_or_pil : chemin vers une image OU objet PIL.Image
    - langs : liste de codes langue ISO (ex: ['fr', 'en']) ou None pour tout

    Retourne le texte extrait, nettoye des espaces superflus.
    """
    # Convertir la liste de langues ISO en codes Tesseract
    if langs:
        tess_codes = [TESSERACT_LANGS.get(lg, lg) for lg in langs]
        lang_string = "+".join(tess_codes)
    else:
        # Par defaut : francais + anglais
        lang_string = "fra+eng"

    # Charger l'image
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil)
    else:
        image = image_path_or_pil

    # Extraction
    text = pytesseract.image_to_string(image, lang=lang_string)

    # Nettoyage
    return " ".join(text.split())


def list_installed_languages() -> list:
    """Retourne la liste des langues installees dans Tesseract."""
    try:
        return pytesseract.get_languages(config="")
    except Exception:
        return []


# ---------- TEST ----------

if __name__ == "__main__":
    print("=== Test ocr.py ===\n")

    # 1. Verifier l'installation
    print("--- Verification de Tesseract ---")
    if not os.path.exists(TESSERACT_PATH):
        print(f"  [ERREUR] Tesseract non trouve a : {TESSERACT_PATH}")
        print(f"  Installe-le depuis : https://github.com/UB-Mannheim/tesseract/wiki")
        exit(1)
    print(f"  [OK] Tesseract trouve : {TESSERACT_PATH}")

    # 2. Lister les langues disponibles
    langs = list_installed_languages()
    print(f"  Langues installees : {langs}\n")

    # 3. Test sur une image (creer une image avec du texte)
    print("--- Test : creation d'une image avec texte ---")
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", (600, 100), color="white")
    draw = ImageDraw.Draw(img)
    text_to_render = "Bonjour le monde ! Hello world !"
    draw.text((20, 30), text_to_render, fill="black")

    test_image_path = "ocr_test.png"
    img.save(test_image_path)
    print(f"  Image creee : {test_image_path}")
    print(f"  Texte d'origine : '{text_to_render}'\n")

    # 4. OCR sur cette image
    print("--- OCR ---")
    extracted = extract_text_from_image(test_image_path, langs=["fr", "en"])
    print(f"  Texte extrait : '{extracted}'\n")

    # Nettoyage
    os.remove(test_image_path)

    print("=== Test OK ===")
