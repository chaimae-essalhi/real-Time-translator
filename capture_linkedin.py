"""
Refait les 6 captures LinkedIn avec TES vraies donnees (MongoDB, Elasticsearch, Mistral).

1. Lance l'application dans un premier terminal :   streamlit run app.py
2. Dans un second terminal (meme venv) :
       pip install playwright
       playwright install chromium
       python capture_linkedin.py
3. Les images (2880 x 1800, format PC 1440 x 900 en haute definition)
   arrivent dans le dossier captures_linkedin/.

Attention : le script clique vraiment sur "Traduire" et "Sauvegarder la traduction"
est coche par defaut, il ajoute donc un document a ta base.
"""
import asyncio, pathlib
from playwright.async_api import async_playwright

async def idle(pg, ms=1500):
    await pg.wait_for_timeout(600)
    try: await pg.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=8000)
    except Exception: pass
    await pg.evaluate("document.fonts.ready")
    await pg.wait_for_timeout(ms)

async def main():
    out = pathlib.Path("captures_linkedin"); out.mkdir(exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width":1440,"height":900}, device_scale_factor=2, locale="fr-FR")
        pg = await ctx.new_page()
        await pg.goto("http://localhost:8501"); await idle(pg, 3000)
        await pg.screenshot(path=out/"01-tableau-de-bord.png", full_page=True)
        async def nav(k): await pg.click(f".st-key-nav_{k} button"); await idle(pg)
        await nav("translate")
        await pg.locator("textarea").first.fill("Bienvenue sur notre plateforme de traduction basée sur l’IA. Cette solution vous permet de traduire vos documents et vos textes dans plusieurs langues, en quelques secondes.")
        await pg.keyboard.press("Control+Enter"); await idle(pg, 500)
        await pg.get_by_role("button", name="Traduire").click(); await idle(pg, 6500)
        await pg.mouse.move(5, 890)
        await pg.screenshot(path=out/"02-traduction.png", full_page=True)
        await nav("similarity")
        ta = pg.locator("textarea")
        await ta.nth(0).fill("L’intelligence artificielle transforme profondément la manière dont les entreprises analysent leurs données.")
        await ta.nth(1).fill("Grâce à l’IA, les entreprises analysent désormais leurs données autrement : leurs méthodes changent en profondeur.")
        await pg.keyboard.press("Control+Enter"); await idle(pg, 500)
        await pg.get_by_role("button", name="Comparer les textes").click(); await idle(pg, 2000)
        await pg.mouse.move(5, 890)
        await pg.screenshot(path=out/"03-similarite.png", full_page=True)
        await nav("search")
        await pg.locator("input[type=text]").first.fill("intelligence artificielle éducation")
        await pg.keyboard.press("Enter"); await idle(pg, 1500)
        await pg.click(".ph h1"); await pg.wait_for_timeout(600)
        await pg.mouse.move(5, 890)
        await pg.screenshot(path=out/"04-recherche.png", full_page=True)
        await nav("plagiarism")
        await pg.locator("textarea").first.fill("L’IA transforme en profondeur notre société moderne et la façon dont nous travaillons au quotidien.")
        await pg.keyboard.press("Control+Enter"); await idle(pg, 500)
        await pg.get_by_role("button", name="Vérifier le plagiat").click(); await idle(pg, 2000)
        await pg.mouse.move(5, 890)
        await pg.screenshot(path=out/"05-plagiat.png", full_page=True)
        await nav("stats"); await pg.mouse.move(5, 890); await idle(pg, 1500)
        await pg.screenshot(path=out/"06-statistiques.png", full_page=True)
        await b.close()
asyncio.run(main())
