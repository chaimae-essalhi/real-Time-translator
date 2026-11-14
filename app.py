"""
Application Streamlit - Real Time Translator
Interface "admin dashboard" : sidebar violette, cartes blanches, graphiques SVG.

Lancer avec :
    streamlit run app.py
"""

import html as _html
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import streamlit as st

from translator import translate_to_many, detect_language, SUPPORTED_LANGUAGES
from embeddings import get_embedding, cosine_similarity
from search import create_index, search_text, search_semantic, index_document, INDEX_NAME, es
from storage import save_document, add_translation, set_embedding
from plagiarism import check_plagiarism, add_reference_document, DEFAULT_THRESHOLD

try:
    import requests  # installe avec streamlit
except ImportError:  # pragma: no cover
    requests = None


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="RT Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

KIBANA_URL = "http://localhost:5601"
ES_URL = "http://localhost:9200"
MISTRAL_URL = "https://console.mistral.ai"

# Palette (utilisee dans les SVG generes en Python)
C_PRIMARY = "#6C47FF"
C_PRIMARY_SOFT = "#A78BFA"
C_GREEN = "#22C55E"
C_ORANGE = "#F59E0B"
C_BLUE = "#3B82F6"
C_RED = "#E5484D"
C_GRID = "#ECE8F7"
C_MUTED = "#8C89A8"

LANG_NAMES = {
    "fr": "Français", "en": "Anglais", "es": "Espagnol", "de": "Allemand",
    "it": "Italien", "pt": "Portugais", "ar": "Arabe", "ru": "Russe",
    "zh": "Chinois", "ja": "Japonais",
}
LANG_FLAGS = {  # code langue -> code pays (flagcdn)
    "fr": "fr", "en": "gb", "es": "es", "de": "de", "it": "it",
    "pt": "pt", "ar": "sa", "ru": "ru", "zh": "cn", "ja": "jp",
}
RTL_LANGS = {"ar"}
MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
        "juil.", "août", "sept.", "oct.", "nov.", "déc."]


# ============================================================
# PETITS UTILITAIRES
# ============================================================

def esc(text) -> str:
    return _html.escape(str(text if text is not None else ""))


def md(markup: str):
    """Affiche du HTML. Supprime l'indentation et les lignes vides
    (sinon le moteur Markdown transforme le HTML en bloc de code)."""
    lines = [ln.strip() for ln in markup.splitlines()]
    st.markdown("".join(ln for ln in lines if ln), unsafe_allow_html=True)


def lang_name(code: str) -> str:
    code = (code or "").lower()
    return LANG_NAMES.get(code, code.upper() if code else "Inconnue")


def flag(code: str) -> str:
    cc = LANG_FLAGS.get((code or "").lower())
    if not cc:
        return f'<span class="flag flag-txt">{esc((code or "?").upper())}</span>'
    return f'<img class="flag" src="https://flagcdn.com/w40/{cc}.png" alt="{esc(code.upper())}">'


def fnum(n) -> str:
    return f"{int(n):,}".replace(",", "\u202f")


def fpct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}".replace(".", ",") + "\u202f%"


def fdate(d) -> str:
    if not isinstance(d, datetime):
        return "—"
    return f"{d.day} {MOIS[d.month - 1]} {d.year}"


def fdatetime(d) -> str:
    if not isinstance(d, datetime):
        return "—"
    return f"{d.day:02d}/{d.month:02d}/{d.year} {d.hour:02d}:{d.minute:02d}"


def short_title(text: str, n: int = 64) -> str:
    text = " ".join((text or "").split())
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    first = first if len(first) <= n else first[:n].rsplit(" ", 1)[0] + "…"
    return first


def truncate(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def alert(kind: str, text: str):
    ico = {"info": "info", "success": "check", "warning": "alert-triangle", "error": "alert-triangle"}[kind]
    md(f'<div class="alert alert-{kind}">{icon(ico, 18)}<span>{text}</span></div>')


def friendly_error(e: Exception) -> str:
    msg = str(e)
    if "429" in msg or "capacity" in msg or "rate" in msg.lower():
        return "Mistral AI est saturé (limite du forfait gratuit). Réessayez dans quelques secondes."
    if "Connection" in msg or "refused" in msg.lower() or "timed out" in msg.lower():
        return "Un service ne répond pas. Vérifiez que MongoDB et Elasticsearch sont démarrés."
    return f"Erreur : {esc(truncate(msg, 220))}"


# ============================================================
# ICONES SVG (style Lucide)
# ============================================================

ICON_PATHS = {
    "home": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "languages": '<path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="m22 22-5-10-5 10"/><path d="M14 18h6"/>',
    "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
    "zap": '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "file-text": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><line x1="10" x2="8" y1="9" y2="9"/>',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',
    "external-link": '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
    "circle": '<circle cx="12" cy="12" r="10"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "calendar": '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "trend-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "trend-down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "list": '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    "pie": '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
    "server": '<rect width="20" height="8" x="2" y="2" rx="2"/><rect width="20" height="8" x="2" y="14" rx="2"/><path d="M6 6h.01"/><path d="M6 18h.01"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "split": '<path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3"/><path d="m15 9 6-6"/>',
}


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    path = ICON_PATHS.get(name, ICON_PATHS["circle"])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{path}</svg>')


# ============================================================
# CSS
# ============================================================

md("""
<style>
:root {
  --primary: #6C47FF;
  --primary-600: #5A33E6;
  --primary-50: #F3F0FF;
  --primary-100: #E8E2FF;
  --bg: #F5F4FB;
  --card: #FFFFFF;
  --border: #ECE8F7;
  --text: #1E1B3A;
  --text-2: #4A4766;
  --muted: #8C89A8;
  --green: #16A34A;  --green-bg: #E9F8EF;
  --red: #E5484D;    --red-bg: #FDEDED;
  --orange: #D97706; --orange-bg: #FEF3E2;
  --blue: #2563EB;   --blue-bg: #EAF1FE;
  --shadow: 0 1px 2px rgba(30,27,58,.04), 0 10px 28px -16px rgba(76,48,200,.22);
  --grad: linear-gradient(135deg, #7B5BFF 0%, #5A33E6 100%);
}
.stApp { background: var(--bg); }
.block-container { padding-top: 1.6rem !important; padding-bottom: 3rem !important; max-width: 1440px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"], footer, #MainMenu { display: none !important; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--bg) !important; border-right: none !important; }
section[data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg, #6E45FA 0%, #5228DC 55%, #3E1CB4 100%);
  margin: 12px 0 12px 12px; border-radius: 22px;
  height: calc(100% - 24px); width: calc(100% - 12px);
  box-shadow: 0 18px 40px -18px rgba(62,28,180,.55);
}
[data-testid="stSidebarHeader"] { padding: 1rem 1rem 0 1.2rem; min-height: 0; }
[data-testid="stSidebarUserContent"] { padding: 0 .9rem 1.5rem .9rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .2rem; }
[data-testid="stSidebarResizeHandle"] { opacity: 0; }
.brand { display: flex; align-items: center; gap: 12px; padding: .2rem .4rem 1rem .4rem; }
.brand-icon { width: 38px; height: 38px; border-radius: 12px; background: rgba(255,255,255,.18);
  display: flex; align-items: center; justify-content: center; }
.brand-name { font-size: 1.12rem; font-weight: 700; color: #fff; letter-spacing: -.01em; }
.sb-section { color: rgba(255,255,255,.55) !important; font-size: .68rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .1em; margin: 1.1rem 0 .45rem .7rem; }
section[data-testid="stSidebar"] .stButton button {
  justify-content: flex-start; background: transparent; border: none; box-shadow: none;
  color: rgba(255,255,255,.80); padding: .5rem .8rem; min-height: 2.6rem;
  border-radius: 12px; font-weight: 500; font-size: .92rem;
}
section[data-testid="stSidebar"] .stButton button > div { justify-content: flex-start; }
section[data-testid="stSidebar"] .stButton button p { font-size: .92rem; margin: 0; }
section[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,.10); color: #fff; }
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
  background: rgba(255,255,255,.17) !important; color: #fff !important; font-weight: 600;
  box-shadow: inset 3px 0 0 #fff !important;
}
.sb-status { margin-top: 1.2rem; background: rgba(255,255,255,.10); border-radius: 16px; padding: .85rem 1rem; }
.sb-status-title { color: #fff; font-weight: 600; font-size: .85rem; margin-bottom: .45rem; }
.sb-svc { display: flex; align-items: center; gap: 8px; color: rgba(255,255,255,.85);
  font-size: .82rem; padding: .18rem 0; }
.sb-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.sb-dot.on { background: #4ADE80; box-shadow: 0 0 0 3px rgba(74,222,128,.22); }
.sb-dot.off { background: #FB7185; box-shadow: 0 0 0 3px rgba(251,113,133,.22); }

/* ---------- Cartes ---------- */
div[class*="st-key-card"], .kcard {
  background: var(--card); border: 1px solid var(--border); border-radius: 18px;
  padding: 1.1rem 1.25rem 1.2rem 1.25rem; box-shadow: var(--shadow);
}
div[class*="st-key-card"] { gap: .7rem; }
.card-title { display: flex; align-items: center; gap: 9px; margin: 0; font-weight: 700;
  font-size: 1rem; color: var(--text); }
.card-title svg { color: var(--primary); flex: none; }
.card-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.card-sub { color: var(--muted); font-size: .82rem; margin: 0; }
.link { color: var(--primary) !important; font-weight: 600; font-size: .82rem; text-decoration: none; }

/* ---------- En-tete de page ---------- */
.ph { display: flex; align-items: center; gap: 16px; margin-bottom: .3rem; }
.ph-icon { width: 54px; height: 54px; border-radius: 16px; background: var(--primary-100);
  display: flex; align-items: center; justify-content: center; color: var(--primary); flex: none; }
.ph h1 { font-size: 1.65rem; font-weight: 800; margin: 0; padding: 0; color: var(--text);
  letter-spacing: -.02em; line-height: 1.2; }
.ph .sub { margin: .15rem 0 0 0; color: var(--muted); font-size: .9rem; }
.chip { display: inline-flex; align-items: center; gap: 8px; background: #fff;
  border: 1px solid var(--border); border-radius: 12px; padding: .5rem .85rem;
  font-size: .82rem; color: var(--text-2); font-weight: 500; white-space: nowrap; }
.chip svg { color: var(--primary); }
.chip-wrap { display: flex; justify-content: flex-end; }

/* ---------- KPI ---------- */
.kpi { display: flex; flex-direction: column; gap: .55rem; height: 100%; }
.kpi-top { display: flex; align-items: flex-start; gap: 12px; }
.kpi-ico { width: 44px; height: 44px; border-radius: 13px; display: flex; align-items: center;
  justify-content: center; flex: none; }
.kpi-label { color: var(--text-2); font-size: .82rem; font-weight: 600; margin: 0; }
.kpi-val { font-size: 1.55rem; font-weight: 800; color: var(--text); margin: 0; letter-spacing: -.02em; line-height: 1.2; }
.kpi-spark { margin-left: auto; align-self: center; flex: 0 1 110px; min-width: 56px; }
.kpi-spark svg { width: 100%; height: auto; display: block; }
.kpi-foot { font-size: .76rem; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.up { color: var(--green); font-weight: 700; display: inline-flex; align-items: center; gap: 3px; }
.down { color: var(--red); font-weight: 700; display: inline-flex; align-items: center; gap: 3px; }
.t-purple { background: var(--primary-100); color: var(--primary); }
.t-green { background: var(--green-bg); color: var(--green); }
.t-blue { background: var(--blue-bg); color: var(--blue); }
.t-orange { background: var(--orange-bg); color: var(--orange); }
.t-red { background: var(--red-bg); color: var(--red); }

/* ---------- Pastilles ---------- */
.pill { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px;
  padding: .28rem .7rem; font-size: .76rem; font-weight: 600; white-space: nowrap; }
.flag { width: 20px; height: 14px; border-radius: 3px; object-fit: cover; flex: none;
  box-shadow: 0 0 0 1px rgba(0,0,0,.06); }
.flag-txt { display: inline-flex; align-items: center; justify-content: center; width: auto;
  padding: 0 4px; font-size: .62rem; font-weight: 700; background: var(--primary-50); color: var(--primary); }
.lang-head { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: .9rem; color: var(--text); }
.badge-letter { width: 26px; height: 26px; border-radius: 8px; background: var(--primary-100);
  color: var(--primary); font-weight: 800; display: inline-flex; align-items: center;
  justify-content: center; font-size: .85rem; }

/* ---------- Widgets ---------- */
[data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="select"] > div {
  border-radius: 12px !important; background: #FBFAFF !important; border-color: var(--border) !important;
}
[data-baseweb="textarea"] textarea { font-size: .95rem; line-height: 1.6; color: var(--text); }
.stButton button, .stFormSubmitButton button { border-radius: 12px; font-weight: 600; min-height: 2.75rem; }
.block-container [data-testid="stBaseButton-primary"],
.block-container [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--grad); border: none; box-shadow: 0 10px 22px -10px rgba(90,51,230,.65);
}
.block-container [data-testid="stBaseButton-primary"]:hover,
.block-container [data-testid="stBaseButton-primaryFormSubmit"]:hover { filter: brightness(1.06); }
.block-container [data-testid="stBaseButton-secondary"] {
  background: var(--primary-50); color: var(--primary); border: 1px solid var(--primary-100);
}
.arrow-btn { display: flex; justify-content: center; }
.arrow-btn span { width: 42px; height: 42px; border-radius: 50%; background: var(--primary-100);
  color: var(--primary); display: flex; align-items: center; justify-content: center; }

/* Onglets en pastilles (anciennes et nouvelles versions de Streamlit) */
[data-testid="stTabs"] [role="tablist"] { gap: 8px; background: transparent; border: none !important;
  box-shadow: none !important; padding-bottom: .3rem; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], .react-aria-SelectionIndicator { display: none !important; }
[data-testid="stTabs"] [role="tab"] { background: #fff; border: 1px solid var(--border); border-radius: 12px;
  padding: .55rem 1.1rem !important; height: auto; }
[data-testid="stTabs"] [role="tab"] p { font-weight: 600; color: var(--text-2); margin: 0; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: var(--grad); border-color: transparent;
  box-shadow: 0 10px 22px -12px rgba(90,51,230,.7); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
[data-testid="stTabs"] [role="tab"][aria-selected="true"] span { color: #fff !important; }

/* Tuiles de traduction copiables (st.code restyle) */
div[class*="st-key-tile_"] { border: 1px solid var(--border); border-radius: 14px; background: #FBFAFF;
  padding: .75rem .8rem .5rem .85rem; gap: .35rem; height: 100%; }
div[class*="st-key-tile_"] .lang-head { margin-bottom: .8rem; }
div[class*="st-key-tile_"] [data-testid="stCode"] pre,
div[class*="st-key-tile_"] [data-testid="stCode"] code {
  font-family: inherit !important; font-size: .9rem !important; line-height: 1.6 !important;
  white-space: pre-wrap !important; overflow-wrap: break-word; word-break: normal;
  color: var(--text) !important; background: transparent !important;
}
div[class*="st-key-tile_"] [data-testid="stCode"] pre { border: none; padding: 0 1.9rem 0 0 !important; }
div[class*="st-key-tile_res_ar"] [data-testid="stCode"] pre,
div[class*="st-key-tile_main_ar"] [data-testid="stCode"] pre { direction: rtl; text-align: right;
  padding: 0 0 0 1.9rem !important; }

/* ---------- Alertes ---------- */
.alert { display: flex; align-items: center; gap: 10px; padding: .75rem 1rem; border-radius: 12px;
  font-size: .88rem; margin: .2rem 0; }
.alert svg { flex: none; }
.alert-info { background: var(--blue-bg); color: #1E40AF; }
.alert-success { background: var(--green-bg); color: #166534; }
.alert-warning { background: var(--orange-bg); color: #92400E; }
.alert-error { background: var(--red-bg); color: #9F1239; }
.placeholder { border: 1.5px dashed var(--primary-100); background: #FBFAFF; border-radius: 12px;
  padding: 1.6rem 1rem; text-align: center; color: var(--muted); font-size: .88rem; }
.ready { display: flex; justify-content: flex-end; align-items: center; gap: 6px; color: var(--green);
  font-size: .78rem; font-weight: 600; }

/* ---------- Jauge / resultats ---------- */
.gauge-wrap { display: flex; flex-direction: column; align-items: center; gap: .4rem; }
.gauge-wrap svg { width: 100%; max-width: 260px; height: auto; }
.mini-stat { border: 1px solid var(--border); border-radius: 14px; padding: .8rem .9rem; }
.mini-stat-top { display: flex; align-items: center; gap: 10px; }
.mini-stat-label { color: var(--text-2); font-size: .8rem; font-weight: 600; }
.mini-stat-val { font-size: 1.3rem; font-weight: 800; color: var(--text); }
.bar { height: 7px; background: var(--primary-50); border-radius: 99px; overflow: hidden; margin-top: .5rem; }
.bar > span { display: block; height: 100%; border-radius: 99px; }
.lead { font-size: .98rem; color: var(--text-2); margin: 0; line-height: 1.6; }
.lead b.hl { color: var(--primary); font-size: 1.15rem; }
.diff-box { background: #FBFAFF; border: 1px solid var(--border); border-radius: 12px; padding: .8rem .9rem;
  font-size: .9rem; line-height: 1.75; color: var(--text); }
.diff-del { background: #FDE2E4; color: #B4232C; border-radius: 4px; padding: 0 2px; }
.diff-add { background: #DCF7E5; color: #11753A; border-radius: 4px; padding: 0 2px; }
.verdict { display: flex; gap: 14px; align-items: flex-start; }
.verdict h4 { margin: 0 0 .3rem 0; font-size: 1.05rem; font-weight: 700; color: var(--text); padding: 0; }
.verdict .sub { margin: 0; color: var(--text-2); font-size: .86rem; line-height: 1.55; }
.vsep { border-left: 1px solid var(--border); height: 100%; min-height: 160px; margin: 0 auto; width: 1px; }

/* ---------- Listes de documents ---------- */
.doc-row { display: flex; align-items: center; gap: 12px; padding: .65rem .2rem;
  border-bottom: 1px solid var(--border); }
.doc-row:last-child { border-bottom: none; }
.doc-ico { width: 38px; height: 38px; border-radius: 11px; background: var(--primary-50);
  color: var(--primary); display: flex; align-items: center; justify-content: center; flex: none; }
.doc-main { flex: 1; min-width: 0; }
.doc-title { font-weight: 600; font-size: .88rem; color: var(--text); margin: 0;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-meta { color: var(--muted); font-size: .76rem; margin: .1rem 0 0 0; display: flex; gap: 12px; flex-wrap: wrap; }
.doc-meta span { display: inline-flex; align-items: center; gap: 4px; }

.sr { display: flex; gap: 16px; align-items: center; background: #fff; border: 1px solid var(--border);
  border-radius: 18px; padding: 1rem 1.2rem; margin-bottom: .75rem; box-shadow: var(--shadow); }
.sr-ico { width: 56px; height: 56px; border-radius: 16px; background: var(--primary-50); color: var(--primary);
  display: flex; align-items: center; justify-content: center; flex: none; }
.sr-main { flex: 1; min-width: 0; }
.sr-title { font-weight: 700; font-size: 1rem; color: var(--text); margin: 0 0 .25rem 0; }
.sr-snip { color: var(--text-2); font-size: .88rem; line-height: 1.6; margin: 0 0 .45rem 0; }
.sr-snip mark { background: #FDEFA8; color: var(--text); border-radius: 4px; padding: 0 3px; font-weight: 600; }
.sr-score { flex: none; text-align: center; background: var(--primary-50); border-radius: 14px;
  padding: .55rem .8rem; min-width: 92px; }
.sr-score b { display: flex; align-items: center; justify-content: center; gap: 5px;
  color: var(--primary); font-size: 1.05rem; }
.sr-score small { color: var(--muted); font-size: .7rem; }
.count-title { display: flex; align-items: center; gap: 8px; font-weight: 700; color: var(--text); font-size: 1rem; }
.count-title svg { color: var(--primary); }

/* ---------- Tableau ---------- */
.tbl-wrap { overflow-x: auto; }
[data-testid="stMarkdownContainer"] table.tbl { width: 100%; border-collapse: collapse; font-size: .83rem;
  border: none; margin: 0; display: table; }
[data-testid="stMarkdownContainer"] table.tbl th { text-align: left; color: var(--muted); font-weight: 600;
  font-size: .75rem; padding: .55rem .5rem; background: #FAF9FE; border: none; }
[data-testid="stMarkdownContainer"] table.tbl td { padding: .55rem .5rem; border: none;
  border-top: 1px solid var(--border); color: var(--text-2); vertical-align: middle; }
[data-testid="stMarkdownContainer"] table.tbl tr { background: transparent !important; }
[data-testid="stMarkdownContainer"] table.tbl td.id { color: var(--text); font-weight: 600; white-space: nowrap; }
[data-testid="stMarkdownContainer"] table.tbl .ov { max-width: 230px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ---------- Legende / services ---------- */
.legend { display: flex; flex-direction: column; gap: .7rem; }
.legend-row { display: flex; align-items: center; gap: 10px; font-size: .86rem; color: var(--text-2); }
.legend-row b { margin-left: auto; color: var(--text); }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.svc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.svc { border: 1px solid var(--border); border-radius: 14px; padding: .9rem; display: flex;
  flex-direction: column; align-items: center; gap: .35rem; text-align: center; }
.svc-ico { width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; }
.svc-name { font-weight: 700; color: var(--text); font-size: .88rem; }
.svc-state { font-size: .75rem; display: flex; align-items: center; gap: 6px; color: var(--text-2); }
.svc-link { margin-top: .3rem; font-size: .76rem; font-weight: 600; color: var(--primary) !important;
  border: 1px solid var(--primary-100); border-radius: 9px; padding: .3rem .7rem; text-decoration: none !important; }
.svc-addr { margin-top: .3rem; font-size: .72rem; color: var(--muted); }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot.on { background: #22C55E; } .dot.off { background: #E5484D; }
.chart-wrap svg { width: 100%; height: auto; display: block; }
.big-num { font-size: 1.9rem; font-weight: 800; color: var(--text); letter-spacing: -.02em; margin: 0; line-height: 1.1; }
.big-lbl { font-size: .8rem; color: var(--muted); display: flex; align-items: center; gap: 6px; margin: .15rem 0 1rem 0; }
.hello { color: var(--text-2); font-size: .95rem; margin: 0; }

div[class*="st-key-quick"] .stButton button { background: #fff; border: 1px solid var(--border);
  color: var(--text); min-height: 3.4rem; box-shadow: var(--shadow); justify-content: flex-start; padding: 0 1rem; }
div[class*="st-key-quick"] .stButton button:hover { border-color: var(--primary); color: var(--primary); }
div[class*="st-key-quick"] .stButton button [data-testid="stIconMaterial"] { color: var(--primary); }

@media (max-width: 900px) { .vsep { display: none; } .sr { flex-wrap: wrap; } }
</style>
""")


# ============================================================
# DONNEES (avec cache)
# ============================================================

@st.cache_resource
def _init():
    try:
        create_index()
    except Exception:
        pass
    return True


_init()


@st.cache_data(ttl=20, show_spinner=False)
def fetch_docs(limit: int = 2000) -> list:
    """Documents MongoDB recents, sans le vecteur d'embedding (lourd)."""
    try:
        from storage import _documents
        docs = []
        for d in _documents.find({}, {"embedding": 0}).sort("created_at", -1).limit(limit):
            d["_id"] = str(d["_id"])
            docs.append(d)
        return docs
    except Exception:
        return []


@st.cache_data(ttl=20, show_spinner=False)
def es_stats() -> dict:
    out = {"count": 0, "buckets": []}
    try:
        out["count"] = es.count(index=INDEX_NAME)["count"]
        agg = es.search(index=INDEX_NAME, size=0,
                        aggregations={"by_lang": {"terms": {"field": "lang", "size": 20}}})
        out["buckets"] = [(b["key"], b["doc_count"]) for b in agg["aggregations"]["by_lang"]["buckets"]]
    except Exception:
        pass
    return out


@st.cache_data(ttl=20, show_spinner=False)
def services_status() -> dict:
    s = {"mongo": False, "elastic": False, "mistral": False, "kibana": False, "latency_ms": None}
    try:
        t0 = time.perf_counter()
        es.info()
        s["latency_ms"] = (time.perf_counter() - t0) * 1000
        s["elastic"] = True
    except Exception:
        pass
    try:
        from storage import _db
        _db.client.admin.command("ping")
        s["mongo"] = True
    except Exception:
        pass
    try:
        from config import MISTRAL_API_KEY
        s["mistral"] = bool(MISTRAL_API_KEY)
    except Exception:
        pass
    if requests is not None:
        try:
            s["kibana"] = requests.get(f"{KIBANA_URL}/api/status", timeout=0.8).status_code < 500
        except Exception:
            pass
    return s


def refresh_data():
    fetch_docs.clear()
    es_stats.clear()


def _utcnow() -> datetime:
    """Heure UTC sans fuseau, comme les dates stockees par MongoDB."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def daily_series(docs: list, days: int, value=lambda d: 1, end: datetime = None):
    """Compte par jour sur les `days` derniers jours (dates UTC de MongoDB)."""
    end = (end or _utcnow()).date()
    start = end - timedelta(days=days - 1)
    buckets = Counter()
    for d in docs:
        c = d.get("created_at")
        if isinstance(c, datetime) and start <= c.date() <= end:
            buckets[c.date()] += value(d)
    dates = [start + timedelta(days=i) for i in range(days)]
    return dates, [buckets[x] for x in dates]


def period_totals(docs: list, days: int, value=lambda d: 1):
    now = _utcnow()
    cur_start, prev_start = now - timedelta(days=days), now - timedelta(days=2 * days)
    cur = prev = 0
    for d in docs:
        c = d.get("created_at")
        if not isinstance(c, datetime):
            continue
        if c >= cur_start:
            cur += value(d)
        elif c >= prev_start:
            prev += value(d)
    return cur, prev


def n_translations(d) -> int:
    return len(d.get("translations") or {})


# ============================================================
# GRAPHIQUES SVG
# ============================================================

_uid_counter = [0]


def _uid(prefix: str) -> str:
    _uid_counter[0] += 1
    return f"{prefix}{_uid_counter[0]}"


def _smooth_path(pts, y_min, y_max) -> str:
    if not pts:
        return ""
    if len(pts) == 1:
        return f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    clamp = lambda y: max(y_min, min(y_max, y))
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, clamp(p1[1] + (p2[1] - p0[1]) / 6))
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, clamp(p2[1] - (p3[1] - p1[1]) / 6))
        d += f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    return d


def sparkline(values, color=C_PRIMARY, w=110, h=42) -> str:
    if not values:
        values = [0, 0]
    pad = 4
    vmax = max(values) or 1
    n = len(values)
    pts = [(pad + i * (w - 2 * pad) / max(n - 1, 1), h - pad - (v / vmax) * (h - 2 * pad)) for i, v in enumerate(values)]
    line = _smooth_path(pts, pad, h - pad)
    gid = _uid("sg")
    area = f"{line} L{pts[-1][0]:.1f},{h} L{pts[0][0]:.1f},{h} Z"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".22"/><stop offset="1" stop-color="{color}" stop-opacity="0"/>'
            f'</linearGradient></defs><path d="{area}" fill="url(#{gid})"/>'
            f'<path d="{line}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round"/></svg>')


def _nice_max(v: float) -> int:
    if v <= 4:
        return 4
    for step in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000):
        if v <= step * 4:
            return step * 4
    return int(v * 1.1)


def area_chart(series, labels, height=240) -> str:
    """series = [(nom, valeurs, couleur)], labels = textes de l'axe X."""
    W, H = 600, height
    L, R, T, B = 40, 12, 12, 30
    vmax = _nice_max(max((max(v) if v else 0) for _, v, _ in series))
    n = len(labels)
    x = lambda i: L + i * (W - L - R) / max(n - 1, 1)
    y = lambda v: T + (1 - v / vmax) * (H - T - B)
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    for k in range(5):
        val = vmax * k / 4
        yy = y(val)
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{C_GRID}" stroke-dasharray="{"0" if k == 0 else "4 4"}"/>')
        parts.append(f'<text x="{L - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="13" fill="{C_MUTED}">{int(val)}</text>')
    step = max(1, round(n / 6))
    for i in range(0, n, step):
        parts.append(f'<text x="{x(i):.1f}" y="{H - 6}" text-anchor="middle" font-size="13" fill="{C_MUTED}">{esc(labels[i])}</text>')
    for idx, (name, vals, color) in enumerate(series):
        pts = [(x(i), y(v)) for i, v in enumerate(vals)]
        line = _smooth_path(pts, T, H - B)
        gid = _uid("ag")
        op = ".26" if idx == 0 else ".10"
        parts.append(f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
                     f'<stop offset="0" stop-color="{color}" stop-opacity="{op}"/>'
                     f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>')
        parts.append(f'<path d="{line} L{pts[-1][0]:.1f},{H - B} L{pts[0][0]:.1f},{H - B} Z" fill="url(#{gid})"/>')
        parts.append(f'<path d="{line}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round"/>')
        if n <= 31 and idx == 0:
            for px, py in pts:
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="#fff" stroke="{color}" stroke-width="2"/>')
        else:
            parts.append(f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4" fill="#fff" stroke="{color}" stroke-width="2.4"/>')
    parts.append("</svg>")
    return "".join(parts)


def donut(parts, center_value: str, center_label: str = "Total", size=200) -> str:
    total = sum(v for _, v, _ in parts) or 1
    r, sw = 72, 26
    c = 2 * 3.14159265 * r
    cx = cy = size / 2
    out = [f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">',
           f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{C_GRID}" stroke-width="{sw}"/>']
    offset = 0
    for _, v, color in parts:
        seg = v / total * c
        gap = 2.5 if len(parts) > 1 and seg > 6 else 0
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{sw}" '
                   f'stroke-dasharray="{max(seg - gap, 0):.2f} {c:.2f}" stroke-dashoffset="{-offset:.2f}" '
                   f'transform="rotate(-90 {cx} {cy})"/>')
        offset += seg
    out.append(f'<text x="{cx}" y="{cy + 2}" text-anchor="middle" font-size="24" font-weight="800" fill="#1E1B3A">{esc(center_value)}</text>')
    out.append(f'<text x="{cx}" y="{cy + 22}" text-anchor="middle" font-size="12" fill="{C_MUTED}">{esc(center_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def gauge(value: float, caption: str) -> str:
    v = max(0.0, min(1.0, value))
    gid = _uid("gg")
    pct = f"{v * 100:.1f}".replace(".", ",")
    return (f'<svg viewBox="0 0 240 150" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0" stop-color="#9B7CFF"/><stop offset="1" stop-color="#5A33E6"/></linearGradient></defs>'
            f'<path d="M 24 128 A 96 96 0 0 1 216 128" fill="none" stroke="{C_GRID}" stroke-width="20" stroke-linecap="round"/>'
            f'<path d="M 24 128 A 96 96 0 0 1 216 128" fill="none" stroke="url(#{gid})" stroke-width="20" '
            f'stroke-linecap="round" pathLength="100" stroke-dasharray="{v * 100:.2f} 100"/>'
            f'<text x="120" y="112" text-anchor="middle" font-size="38" font-weight="800" fill="#1E1B3A">{pct}'
            f'<tspan font-size="20" dx="2">%</tspan></text>'
            f'<text x="120" y="138" text-anchor="middle" font-size="13" fill="{C_MUTED}">{esc(caption)}</text></svg>')


def mini_bars(values, labels, color=C_PRIMARY, h=96) -> str:
    W = 260
    n = max(len(values), 1)
    vmax = max(values) if values and max(values) > 0 else 1
    bw = (W - 10) / n * 0.42
    out = [f'<svg viewBox="0 0 {W} {h + 18}" xmlns="http://www.w3.org/2000/svg">']
    for i, v in enumerate(values):
        cx = 5 + (i + 0.5) * (W - 10) / n
        bh = max(4, v / vmax * h)
        out.append(f'<rect x="{cx - bw / 2:.1f}" y="{h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="{bw / 2:.1f}" '
                   f'fill="{color}" opacity="{1 if v == max(values) else .45}"/>')
        out.append(f'<text x="{cx:.1f}" y="{h + 14}" text-anchor="middle" font-size="13" fill="{C_MUTED}">{esc(labels[i])}</text>')
    out.append("</svg>")
    return "".join(out)


# ============================================================
# COMPOSANTS D'INTERFACE
# ============================================================

def page_header(ico: str, title: str, subtitle: str, right=None):
    """En-tete de page. `right` : HTML (chip) ou fonction qui dessine un widget."""
    if right is None:
        md(f'<div class="ph"><div class="ph-icon">{icon(ico, 26)}</div>'
           f'<div><h1>{esc(title)}</h1><div class="sub">{esc(subtitle)}</div></div></div>')
        return
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        md(f'<div class="ph"><div class="ph-icon">{icon(ico, 26)}</div>'
           f'<div><h1>{esc(title)}</h1><div class="sub">{esc(subtitle)}</div></div></div>')
    with c2:
        if callable(right):
            right()
        else:
            md(f'<div class="chip-wrap">{right}</div>')


def today_chip() -> str:
    return f'<span class="chip">{icon("calendar", 16)} Aujourd’hui, {fdate(datetime.now())}</span>'


def card_title(ico: str, title: str, right_html: str = ""):
    md(f'<div class="card-row"><div class="card-title">{icon(ico, 18)} {esc(title)}</div>{right_html}</div>')


def delta_html(cur, prev, label="vs période précédente") -> str:
    if prev == 0 and cur == 0:
        return f'<span class="kpi-foot">Aucune activité sur la période</span>'
    if prev == 0:
        return f'<span class="kpi-foot"><span class="up">{icon("trend-up", 14)} +{fnum(cur)}</span> nouveaux sur la période</span>'
    ch = (cur - prev) / prev
    cls, ico = ("up", "trend-up") if ch >= 0 else ("down", "trend-down")
    sign = "+" if ch >= 0 else ""
    return f'<span class="kpi-foot"><span class="{cls}">{icon(ico, 14)} {sign}{fpct(ch)}</span> {label}</span>'


def kpi_card(ico: str, tone: str, label: str, value: str, foot_html: str, spark: str = ""):
    md(f'<div class="kcard kpi"><div class="kpi-top">'
       f'<div class="kpi-ico t-{tone}">{icon(ico, 22)}</div>'
       f'<div><div class="kpi-label">{esc(label)}</div><div class="kpi-val">{value}</div></div>'
       f'<div class="kpi-spark">{spark}</div></div>{foot_html}</div>')


def doc_row(title: str, meta: list, right_html: str = "", ico: str = "file-text") -> str:
    meta_html = "".join(f"<span>{m}</span>" for m in meta)
    return (f'<div class="doc-row"><div class="doc-ico">{icon(ico, 18)}</div>'
            f'<div class="doc-main"><div class="doc-title">{esc(title)}</div><div class="doc-meta">{meta_html}</div></div>'
            f'{right_html}</div>')


def score_pill(score: float, hot: bool) -> str:
    cls = "t-red" if hot else "t-purple"
    return f'<span class="pill {cls}">{fpct(score)}</span>'


def docs_by_id() -> dict:
    return {d["_id"]: d for d in fetch_docs()}


# ============================================================
# NAVIGATION (sidebar)
# ============================================================

PAGES = [
    ("Vue d'ensemble", [("home", "Tableau de bord", ":material/space_dashboard:")]),
    ("Outils", [
        ("translate", "Traduction", ":material/translate:"),
        ("similarity", "Similarité", ":material/compare_arrows:"),
        ("search", "Recherche", ":material/search:"),
        ("plagiarism", "Détection plagiat", ":material/verified_user:"),
    ]),
    ("Analyse", [("stats", "Statistiques", ":material/bar_chart:")]),
]

if "page" not in st.session_state:
    st.session_state.page = "home"


def go(page_key: str):
    st.session_state.page = page_key


with st.sidebar:
    md(f'<div class="brand"><div class="brand-icon">{icon("languages", 20, "white")}</div>'
       f'<div class="brand-name">RT Translator</div></div>')
    for section, items in PAGES:
        md(f'<div class="sb-section">{esc(section)}</div>')
        for key, label, ico in items:
            st.button(label, icon=ico, key=f"nav_{key}", width="stretch",
                      type="primary" if st.session_state.page == key else "tertiary",
                      on_click=go, args=(key,))
    svc = services_status()
    rows = "".join(
        f'<div class="sb-svc"><span class="sb-dot {"on" if ok else "off"}"></span>{name}</div>'
        for name, ok in [("MongoDB", svc["mongo"]), ("Elasticsearch", svc["elastic"]),
                         ("Mistral AI", svc["mistral"]), ("Kibana", svc["kibana"])]
    )
    online = sum([svc["mongo"], svc["elastic"], svc["mistral"], svc["kibana"]])
    md(f'<div class="sb-status"><div class="sb-status-title">Services : {online}/4 en ligne</div>{rows}</div>')

page = st.session_state.page


# ============================================================
# PAGE : TABLEAU DE BORD
# ============================================================

if page == "home":
    docs = fetch_docs()
    stats = es_stats()
    days = 30
    end = datetime.now()
    start = end - timedelta(days=days - 1)
    range_chip = (f'<span class="chip">{icon("calendar", 16)} '
                  f'{start.day} {MOIS[start.month - 1]} – {end.day} {MOIS[end.month - 1]} {end.year}</span>')

    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        md('<div class="hello">Bonjour 👋</div>'
           '<div class="ph" style="margin-top:.1rem"><div><h1>Bienvenue sur RT Translator</h1>'
           '<div class="sub">Traduction, recherche et détection de plagiat sur une seule plateforme.</div></div></div>')
    with c2:
        md(f'<div class="chip-wrap">{range_chip}</div>')

    cur_docs, prev_docs = period_totals(docs, days)
    cur_tr, prev_tr = period_totals(docs, days, n_translations)
    langs_used = {d.get("lang_source") for d in docs if d.get("lang_source") not in (None, "auto", "unknown")}
    for d in docs:
        langs_used.update((d.get("translations") or {}).keys())
    _, spark_docs = daily_series(docs, days)
    _, spark_tr = daily_series(docs, days, n_translations)

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("database", "purple", "Documents enregistrés", fnum(len(docs)),
                 delta_html(cur_docs, prev_docs), sparkline(spark_docs, C_PRIMARY))
    with k2:
        kpi_card("languages", "green", "Traductions effectuées", fnum(sum(map(n_translations, docs))),
                 delta_html(cur_tr, prev_tr), sparkline(spark_tr, C_GREEN))
    with k3:
        kpi_card("globe", "orange", "Langues utilisées", f"{len(langs_used)}<span style='font-size:1rem;color:var(--muted)'> / {len(SUPPORTED_LANGUAGES)}</span>",
                 f'<span class="kpi-foot">Sur {len(SUPPORTED_LANGUAGES)} langues disponibles</span>')

    left, right = st.columns([2.1, 1], gap="medium")
    with left:
        with st.container(key="card_home_activity"):
            card_title("activity", "Activité sur 30 jours")
            dates, vals_docs = daily_series(docs, days)
            _, vals_tr = daily_series(docs, days, n_translations)
            a, b = st.columns([1.15, 2.85], vertical_alignment="center")
            with a:
                if prev_docs:
                    ch = (cur_docs - prev_docs) / prev_docs
                    trend = f'<div class="card-sub" style="margin-bottom:1rem"><span class="{"up" if ch >= 0 else "down"}">{"+" if ch >= 0 else ""}{fpct(ch)}</span> de documents par rapport aux 30 jours précédents.</div>'
                else:
                    trend = '<div class="card-sub" style="margin-bottom:1rem">Documents et traductions ajoutés chaque jour.</div>'
                md(f'{trend}<div class="big-num">{fnum(cur_docs)}</div>'
                   f'<div class="big-lbl"><span class="legend-dot" style="background:{C_PRIMARY}"></span>Documents ajoutés</div>'
                   f'<div class="big-num">{fnum(cur_tr)}</div>'
                   f'<div class="big-lbl"><span class="legend-dot" style="background:{C_ORANGE}"></span>Traductions</div>')
            with b:
                labels = [f"{d.day} {MOIS[d.month - 1]}" for d in dates]
                md(f'<div class="chart-wrap">{area_chart([("Documents", vals_docs, C_PRIMARY), ("Traductions", vals_tr, C_ORANGE)], labels, 250)}</div>')

    with right:
        with st.container(key="card_home_week"):
            d7, v7 = daily_series(docs, 7, n_translations)
            week = sum(v7)
            card_title("zap", "Traductions sur 7 jours")
            jours = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
            md(f'<div class="big-num" style="margin-top:.2rem">{fnum(week)}</div>'
               f'<div class="chart-wrap">{mini_bars(v7, [jours[d.weekday()] for d in d7], C_PRIMARY, 80)}</div>')
        with st.container(key="card_home_lang"):
            card_title("globe", "Langues cibles les plus demandées")
            targets = Counter()
            for d in docs:
                targets.update((d.get("translations") or {}).keys())
            total_t = sum(targets.values())
            if not total_t:
                md('<div class="card-sub">Aucune traduction pour le moment.</div>')
            else:
                rows = ""
                for code, cnt in targets.most_common(3):
                    share = cnt / total_t
                    rows += (f'<div style="margin-top:.55rem"><div class="legend-row">{flag(code)} {esc(lang_name(code))}<b>{fpct(share)}</b></div>'
                             f'<div class="bar"><span style="width:{share * 100:.1f}%;background:var(--grad)"></span></div></div>')
                md(rows)

    md('<div class="card-title" style="margin:.6rem 0 .1rem 0">Accès rapide</div>')
    with st.container(key="quick"):
        qcols = st.columns(5)
        quick = [("translate", "Traduire un texte", ":material/translate:"),
                 ("similarity", "Comparer deux textes", ":material/compare_arrows:"),
                 ("search", "Rechercher", ":material/search:"),
                 ("plagiarism", "Vérifier le plagiat", ":material/verified_user:"),
                 ("stats", "Voir les statistiques", ":material/bar_chart:")]
        for col, (key, label, ico) in zip(qcols, quick):
            with col:
                st.button(label, icon=ico, key=f"q_{key}", width="stretch", on_click=go, args=(key,))


# ============================================================
# PAGE : TRADUCTION
# ============================================================

elif page == "translate":
    page_header("languages", "Traduction", "Traduisez vos textes vers plusieurs langues en une fois",
                f'<span class="chip">{icon("zap", 16)} Traduction multilingue</span>')

    lang_codes = list(SUPPORTED_LANGUAGES.keys())
    left, mid, right = st.columns([1, 0.07, 1], vertical_alignment="center")

    with left:
        with st.container(key="card_tr_src"):
            card_title("globe", "Langue source")
            source_choice = st.selectbox(
                "Langue source", ["auto"] + lang_codes, key="tr_src", label_visibility="collapsed",
                format_func=lambda x: "Détection automatique" if x == "auto" else lang_name(x),
            )
            card_title("file-text", "Texte source")
            text = st.text_area("Texte source", height=190, max_chars=5000, key="translate_input",
                                placeholder="Saisissez ou collez votre texte ici…",
                                label_visibility="collapsed")

    with mid:
        md(f'<div class="arrow-btn"><span>{icon("arrow-right", 20)}</span></div>')

    with right:
        with st.container(key="card_tr_dst"):
            card_title("globe", "Langues cibles")
            target_langs = st.multiselect("Langues cibles", lang_codes, default=["en", "es", "ar", "de"],
                                          key="tr_targets", format_func=lang_name,
                                          label_visibility="collapsed",
                                          placeholder="Choisissez une ou plusieurs langues")
            card_title("sparkles", "Résultat traduit")
            out_slot = st.container()

    b1, b2, _ = st.columns([1, 1.2, 2.2], vertical_alignment="center")
    with b1:
        do_translate = st.button("Traduire", type="primary", icon=":material/auto_awesome:", width="stretch")
    with b2:
        save_to_db = st.checkbox("Sauvegarder la traduction", value=True)

    if do_translate:
        if not text.strip():
            alert("warning", "Saisissez un texte à traduire.")
        elif not target_langs:
            alert("warning", "Choisissez au moins une langue cible.")
        else:
            try:
                with st.spinner("Traduction en cours…"):
                    src = detect_language(text) if source_choice == "auto" else source_choice
                    results = translate_to_many(text, target_langs, source_lang=src)
                st.session_state.tr = {"src": src, "auto": source_choice == "auto",
                                       "text": text, "results": results}
                if save_to_db:
                    with st.spinner("Sauvegarde…"):
                        doc_id = save_document(text=text, lang_source=src, metadata={"target_langs": target_langs})
                        for lg, tr in results.items():
                            add_translation(doc_id, lg, tr)
                        emb = get_embedding(text)
                        set_embedding(doc_id, emb)
                        index_document(doc_id=doc_id, text=text, lang=src, embedding=emb)
                    refresh_data()
                    st.toast("Traduction sauvegardée", icon=":material/check_circle:")
            except Exception as e:
                alert("error", friendly_error(e))

    tr = st.session_state.get("tr")
    with out_slot:
        if tr and tr["results"]:
            main = next((lg for lg in target_langs if lg in tr["results"]), next(iter(tr["results"])))
            with st.container(key=f"tile_main_{main}"):
                md(f'<div class="lang-head">{flag(main)} {esc(lang_name(main))}</div>')
                st.code(tr["results"][main], language=None, wrap_lines=True)
            detected = f"Langue détectée : {esc(lang_name(tr['src']))}" if tr["auto"] else f"Depuis : {esc(lang_name(tr['src']))}"
            md(f'<div class="ready">{icon("check", 15)} Traduction prête · {detected}</div>')
        else:
            md('<div class="placeholder">La traduction s’affichera ici.</div>')

    if tr and tr["results"]:
        with st.container(key="card_tr_all"):
            card_title("list", "Résultats multilingues",
                       f'<span class="card-sub">{len(tr["results"])} langue(s)</span>')
            items = list(tr["results"].items())
            for row_start in range(0, len(items), 4):
                cols = st.columns(4)
                for col, (lg, trans) in zip(cols, items[row_start:row_start + 4]):
                    with col:
                        with st.container(key=f"tile_res_{lg}"):
                            md(f'<div class="lang-head">{flag(lg)} {esc(lang_name(lg))}</div>')
                            st.code(trans, language=None, wrap_lines=True)


# ============================================================
# PAGE : SIMILARITE
# ============================================================

elif page == "similarity":
    page_header("scale", "Comparer deux textes", "Mesurez la ressemblance de sens entre deux textes",
                today_chip())

    def _swap():
        a, b = st.session_state.get("sim_text1", ""), st.session_state.get("sim_text2", "")
        st.session_state.sim_text1, st.session_state.sim_text2 = b, a

    ca, cm, cb = st.columns([1, 0.08, 1], vertical_alignment="center")
    with ca:
        with st.container(key="card_sim_a"):
            md('<div class="lang-head"><span class="badge-letter">A</span> Texte A</div>')
            text1 = st.text_area("Texte A", height=140, max_chars=10000, key="sim_text1",
                                 placeholder="Premier texte…", label_visibility="collapsed")
    with cm:
        st.button("", icon=":material/swap_horiz:", key="sim_swap", on_click=_swap, help="Inverser les textes")
    with cb:
        with st.container(key="card_sim_b"):
            md('<div class="lang-head"><span class="badge-letter">B</span> Texte B</div>')
            text2 = st.text_area("Texte B", height=140, max_chars=10000, key="sim_text2",
                                 placeholder="Deuxième texte…", label_visibility="collapsed")

    _, bc, _ = st.columns([1.4, 1, 1.4])
    with bc:
        run = st.button("Comparer les textes", type="primary", icon=":material/balance:", width="stretch")

    if run:
        if not text1.strip() or not text2.strip():
            alert("warning", "Renseignez les deux textes.")
        else:
            try:
                with st.spinner("Calcul des embeddings…"):
                    score = cosine_similarity(get_embedding(text1), get_embedding(text2))
                st.session_state.sim = {"score": score, "a": text1, "b": text2}
            except Exception as e:
                alert("error", friendly_error(e))

    sim = st.session_state.get("sim")
    if sim:
        score = sim["score"]
        words = lambda t: re.findall(r"\w+", t.lower())
        set_a, set_b = set(words(sim["a"])), set(words(sim["b"]))
        union = set_a | set_b
        common = len(set_a & set_b) / len(union) if union else 0.0

        if score >= 0.92:
            label, tone = "Quasi identiques", "t-red"
        elif score >= 0.80:
            label, tone = "Très proches", "t-purple"
        elif score >= 0.65:
            label, tone = "Thème commun", "t-blue"
        else:
            label, tone = "Différents", "t-green"

        def diff_html(text: str, other: set, cls: str) -> str:
            out = []
            for i, part in enumerate(re.split(r"(\w+)", text)):
                if i % 2 == 1 and part.lower() not in other:
                    out.append(f'<span class="{cls}">{esc(part)}</span>')
                else:
                    out.append(esc(part))
            return "".join(out).replace("\n", "<br>")

        g, o = st.columns([1, 1.7], gap="medium")
        with g:
            with st.container(key="card_sim_gauge"):
                md(f'<div class="gauge-wrap">{gauge(score, "Score de similarité")}'
                   f'<span class="pill {tone}">{icon("check", 13)} {label}</span></div>')
        with o:
            with st.container(key="card_sim_overview"):
                card_title("chart", "Aperçu du résultat")
                adj = {"Quasi identiques": "quasi identiques", "Très proches": "très proches",
                       "Thème commun": "sur un thème commun", "Différents": "différents"}[label]
                md(f'<div class="lead">Les deux textes sont <b>{adj}</b>, avec un score sémantique de '
                   f'<b class="hl">{fpct(score)}</b>.</div>')
                m1, m2 = st.columns(2)
                with m1:
                    md(f'<div class="mini-stat"><div class="mini-stat-top"><div class="kpi-ico t-purple" style="width:34px;height:34px">{icon("file-text", 17)}</div>'
                       f'<div><div class="mini-stat-label">Mots en commun</div><div class="mini-stat-val">{fpct(common)}</div></div></div>'
                       f'<div class="bar"><span style="width:{common * 100:.1f}%;background:var(--grad)"></span></div></div>')
                with m2:
                    md(f'<div class="mini-stat"><div class="mini-stat-top"><div class="kpi-ico t-purple" style="width:34px;height:34px">{icon("split", 17)}</div>'
                       f'<div><div class="mini-stat-label">Mots différents</div><div class="mini-stat-val">{fpct(1 - common)}</div></div></div>'
                       f'<div class="bar"><span style="width:{(1 - common) * 100:.1f}%;background:{C_PRIMARY_SOFT}"></span></div></div>')

        d, s = st.columns([1.7, 1], gap="medium")
        with d:
            with st.container(key="card_sim_details"):
                card_title("sparkles", "Détail de la comparaison",
                           '<span class="card-sub"><span class="diff-del">absent de B</span> &nbsp; <span class="diff-add">absent de A</span></span>')
                x1, x2 = st.columns(2)
                with x1:
                    md(f'<div class="lang-head" style="margin-bottom:.4rem"><span class="badge-letter">A</span> Texte A</div>'
                       f'<div class="diff-box">{diff_html(sim["a"], set_b, "diff-del")}</div>')
                with x2:
                    md(f'<div class="lang-head" style="margin-bottom:.4rem"><span class="badge-letter">B</span> Texte B</div>'
                       f'<div class="diff-box">{diff_html(sim["b"], set_a, "diff-add")}</div>')
        with s:
            with st.container(key="card_sim_scale"):
                card_title("pie", "Lecture du score")
                scale = [("≥ 92 %", "Quasi identiques, plagiat probable", C_RED),
                         ("80 – 92 %", "Paraphrase ou traduction", C_PRIMARY),
                         ("65 – 80 %", "Thème commun", C_BLUE),
                         ("< 65 %", "Textes différents", C_GREEN)]
                md('<div class="legend">' + "".join(
                    f'<div class="legend-row"><span class="legend-dot" style="background:{c}"></span>{esc(t)}<b>{esc(r)}</b></div>'
                    for r, t, c in scale) + '</div>'
                   '<div class="card-sub" style="margin-top:.9rem">Le score vient des embeddings Mistral : '
                   'il mesure le sens, pas les mots. Deux traductions d’un même texte obtiennent donc un score élevé.</div>')


# ============================================================
# PAGE : RECHERCHE
# ============================================================

elif page == "search":
    page_header("search", "Recherche intelligente", "Trouvez des documents par mots-clés ou par sens",
                today_chip())

    with st.container(key="card_search_bar"):
        with st.form("search_form", border=False):
            q1, q2 = st.columns([5, 1], vertical_alignment="bottom")
            with q1:
                query = st.text_input("Requête", placeholder="Rechercher un mot-clé, une phrase ou une idée…",
                                      label_visibility="collapsed", icon=":material/search:")
            with q2:
                submitted = st.form_submit_button("Rechercher", type="primary", icon=":material/auto_awesome:",
                                                  width="stretch")
            m1, _, m3 = st.columns([1.6, 0.6, 2], vertical_alignment="center")
            with m1:
                mode = st.segmented_control("Mode", ["Full-text", "Sémantique"], default="Full-text",
                                            key="search_mode", label_visibility="collapsed")
            with m3:
                top_k = st.slider("Nombre de résultats", 1, 20, 10)

    if submitted:
        if not query.strip():
            alert("warning", "Saisissez une requête.")
        else:
            try:
                with st.spinner("Recherche…"):
                    if (mode or "Full-text") == "Full-text":
                        results = search_text(query, top_k=top_k)
                    else:
                        results = search_semantic(get_embedding(query), top_k=top_k)
                st.session_state.search = {"query": query, "mode": mode or "Full-text", "results": results}
            except Exception as e:
                alert("error", friendly_error(e))

    srch = st.session_state.get("search")
    if srch:
        results = list(srch["results"])
        if not results:
            alert("info", "Aucun résultat. Indexez d’abord des documents depuis la page Traduction ou Plagiat.")
        else:
            h1, h2 = st.columns([3, 1], vertical_alignment="center")
            with h1:
                md(f'<div class="count-title">{icon("chart", 18)} Résultats trouvés ({len(results)})'
                   f'<span class="card-sub" style="font-weight:500">pour « {esc(srch["query"])} » en mode {esc(srch["mode"].lower())}</span></div>')
            with h2:
                sort = st.selectbox("Trier", ["Pertinence", "Plus récents", "Langue"],
                                    label_visibility="collapsed", key="search_sort")

            semantic = srch["mode"] != "Full-text"
            max_score = max(r["score"] for r in results) or 1
            by_id = docs_by_id()
            for r in results:
                r["_rel"] = r["score"] if semantic else r["score"] / max_score
                r["_date"] = (by_id.get(r.get("doc_id")) or {}).get("created_at")
            if sort == "Plus récents":
                results.sort(key=lambda r: r["_date"] or datetime.min, reverse=True)
            elif sort == "Langue":
                results.sort(key=lambda r: lang_name(r.get("lang")))

            terms = sorted({w for w in re.findall(r"\w+", srch["query"].lower()) if len(w) > 1}, key=len, reverse=True)
            pattern = re.compile(r"\b(" + "|".join(map(re.escape, terms)) + r")\w*", re.IGNORECASE) if terms else None

            for r in results:
                snippet = esc(truncate(r.get("text", ""), 240))
                if pattern:
                    snippet = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet)
                meta = [f'{icon("globe", 13)} {esc(lang_name(r.get("lang")))}',
                        f'{icon("file-text", 13)} Document #{esc(str(r.get("doc_id", ""))[-6:].upper())}']
                if r["_date"]:
                    meta.append(f'{icon("calendar", 13)} Ajouté le {fdate(r["_date"])}')
                md(f'<div class="sr"><div class="sr-ico">{icon("file-text", 26)}</div>'
                   f'<div class="sr-main"><div class="sr-title">{esc(short_title(r.get("text", "")))}</div>'
                   f'<div class="sr-snip">{snippet}</div><div class="doc-meta">{"".join(f"<span>{m}</span>" for m in meta)}</div></div>'
                   f'<div class="sr-score"><b>{icon("star", 15)} {fpct(r["_rel"])}</b><small>Pertinence</small></div></div>')


# ============================================================
# PAGE : PLAGIAT
# ============================================================

elif page == "plagiarism":
    page_header("shield", "Détection de plagiat", "Comparez un texte aux documents de référence",
                today_chip())

    tab1, tab2 = st.tabs([":material/edit_note: Vérifier un texte", ":material/database: Ajouter une référence"])

    with tab1:
        with st.container(key="card_plag_input"):
            card_title("file-text", "Votre texte")
            text = st.text_area("Texte à analyser", height=150, max_chars=10000, key="plag_check",
                                placeholder="Collez ou saisissez votre texte ici…",
                                label_visibility="collapsed")
            t1, t2, t3 = st.columns([1.1, 3, 1.3], vertical_alignment="center")
            with t1:
                thr_pct = st.session_state.get("plag_thr", int(round(DEFAULT_THRESHOLD * 100)))
                md(f'<div class="card-title" style="font-size:.9rem">{icon("gauge", 17)} Seuil : '
                   f'<span style="color:var(--primary)">{thr_pct}&nbsp;%</span></div>')
            with t2:
                thr_pct = st.slider("Seuil de détection", 70, 100, int(round(DEFAULT_THRESHOLD * 100)),
                                    key="plag_thr", label_visibility="collapsed", format="%d %%")
            with t3:
                run = st.button("Vérifier le plagiat", type="primary", icon=":material/search:", width="stretch")
        threshold = thr_pct / 100

        if run:
            if not text.strip():
                alert("warning", "Saisissez un texte à vérifier.")
            else:
                try:
                    with st.spinner("Analyse…"):
                        st.session_state.plag = check_plagiarism(text, threshold=threshold, top_k=5)
                except Exception as e:
                    alert("error", friendly_error(e))

        res = st.session_state.get("plag")
        if res:
            by_id = docs_by_id()
            best = res["matches"][0] if res["matches"] else None
            c_left, c_right = st.columns([1.35, 1], gap="medium")
            with c_left:
                with st.container(key="card_plag_result"):
                    card_title("chart", "Résultat de l’analyse")
                    gcol, scol, vcol = st.columns([1.15, 0.06, 1], vertical_alignment="center")
                    with gcol:
                        verdict_pill = (f'<span class="pill t-red">{icon("alert-triangle", 14)} Plagiat détecté</span>'
                                        if res["is_plagiarism"] else
                                        f'<span class="pill t-green">{icon("check", 14)} Aucun plagiat détecté</span>')
                        md(f'<div class="gauge-wrap">{gauge(res["max_score"], "de similarité")}{verdict_pill}</div>')
                    with scol:
                        md('<div class="vsep"></div>')
                    with vcol:
                        if res["is_plagiarism"]:
                            head = "Contenu très similaire"
                            body = (f"Ce texte atteint {fpct(res['max_score'])} de similarité avec "
                                    f"« {esc(short_title(best['text'], 60))} », au-dessus du seuil de {fpct(res['threshold'], 0)}.")
                            vt = "t-red"
                        else:
                            head = "Contenu original"
                            body = (f"Aucun document de la base ne dépasse le seuil de {fpct(res['threshold'], 0)}. "
                                    f"Le plus proche atteint {fpct(res['max_score'])}." if best else
                                    "La base de référence est vide : ajoutez des documents dans l’onglet voisin.")
                            vt = "t-green"
                        md(f'<div class="verdict"><div class="kpi-ico {vt}">{icon("scale", 22)}</div>'
                           f'<div><h4>{head}</h4><div class="sub">{body}</div></div></div>')
            with c_right:
                with st.container(key="card_plag_matches"):
                    card_title("file-text", "Documents similaires",
                               f'<span class="card-sub">{len(res["matches"])} trouvé(s)</span>')
                    if not res["matches"]:
                        md('<div class="card-sub">Aucun document de référence.</div>')
                    else:
                        rows = ""
                        for m in res["matches"]:
                            dt = (by_id.get(m.get("doc_id")) or {}).get("created_at")
                            meta = [esc(lang_name(m.get("lang")))]
                            if dt:
                                meta.append(f"Ajouté le {fdate(dt)}")
                            rows += doc_row(short_title(m["text"], 48), meta, score_pill(m["score"], m["is_match"]))
                        md(rows)

    with tab2:
        with st.container(key="card_plag_add"):
            card_title("database", "Ajouter une référence")
            md('<div class="card-sub">Ce document servira de base de comparaison pour les prochaines vérifications.</div>')
            ref_text = st.text_area("Référence", height=170, max_chars=10000, key="plag_add",
                                    placeholder="Texte qui servira de référence…", label_visibility="collapsed")
            r1, r2 = st.columns([2, 1], vertical_alignment="bottom")
            with r1:
                ref_lang = st.selectbox("Langue du document", list(SUPPORTED_LANGUAGES.keys()),
                                        format_func=lang_name, key="plag_lang")
            with r2:
                add = st.button("Ajouter à la base", type="primary", icon=":material/add:", width="stretch")
            if add:
                if not ref_text.strip():
                    alert("warning", "Saisissez un texte.")
                else:
                    try:
                        with st.spinner("Indexation…"):
                            doc_id = add_reference_document(ref_text, lang=ref_lang)
                        refresh_data()
                        alert("success", f"<b>Référence ajoutée.</b> Document #{esc(doc_id[-6:].upper())}")
                    except Exception as e:
                        alert("error", friendly_error(e))


# ============================================================
# PAGE : STATISTIQUES
# ============================================================

elif page == "stats":
    def _period_picker():
        st.selectbox("Période", [7, 30, 90], index=1, key="stats_days", label_visibility="collapsed",
                     format_func=lambda d: f"Derniers {d} jours")

    page_header("chart", "Statistiques", "Vue d’ensemble des données et des services", _period_picker)
    days = st.session_state.get("stats_days", 30)

    docs = fetch_docs()
    stats = es_stats()
    svc = services_status()
    cur_docs, prev_docs = period_totals(docs, days)
    cur_tr, prev_tr = period_totals(docs, days, n_translations)
    dates, s_docs = daily_series(docs, days)
    _, s_tr = daily_series(docs, days, n_translations)
    label = f"vs {days} jours précédents"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("file-text", "purple", "Documents (MongoDB)", fnum(len(docs)),
                 delta_html(cur_docs, prev_docs, label), sparkline(s_docs, C_PRIMARY))
    with k2:
        kpi_card("languages", "green", "Traductions", fnum(sum(map(n_translations, docs))),
                 delta_html(cur_tr, prev_tr, label), sparkline(s_tr, C_GREEN))
    with k3:
        sync = stats["count"] / len(docs) if docs else 0
        kpi_card("search", "blue", "Index Elasticsearch", fnum(stats["count"]),
                 f'<span class="kpi-foot">{fpct(min(sync, 1), 0)} des documents MongoDB indexés</span>')
    with k4:
        lat = svc["latency_ms"]
        kpi_card("clock", "orange", "Temps de réponse",
                 f"{lat:.0f}<span style='font-size:1rem'> ms</span>" if lat is not None else "—",
                 '<span class="kpi-foot">Elasticsearch, mesuré à l’instant</span>')

    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1:
        with st.container(key="card_stats_donut"):
            card_title("globe", "Répartition des documents par langue")
            buckets = stats["buckets"]
            total = sum(c for _, c in buckets)
            if not total:
                md('<div class="placeholder">Aucun document indexé pour le moment.</div>')
            else:
                palette = [C_PRIMARY, C_BLUE, C_GREEN, C_ORANGE]
                parts = [(lang_name(k), c, palette[i]) for i, (k, c) in enumerate(buckets[:3])]
                rest = sum(c for _, c in buckets[3:])
                if rest:
                    parts.append(("Autres", rest, palette[3]))
                dcol, lcol = st.columns([1, 1], vertical_alignment="center")
                with dcol:
                    md(f'<div style="display:flex;justify-content:center">{donut(parts, fnum(total))}</div>')
                with lcol:
                    md('<div class="legend">' + "".join(
                        f'<div class="legend-row"><span class="legend-dot" style="background:{c}"></span>{esc(n)}<b>{fpct(v / total)}</b></div>'
                        for n, v, c in parts) + '</div>')
    with c2:
        with st.container(key="card_stats_last"):
            card_title("list", "Derniers documents", f'<span class="card-sub">{min(10, len(docs))} plus récents</span>')
            if not docs:
                md('<div class="placeholder">Aucun document. Utilisez la page Traduction pour en ajouter.</div>')
            else:
                rows = "".join(
                    f'<tr><td class="id">#{esc(d["_id"][-6:].upper())}</td>'
                    f'<td><div class="ov">{esc(truncate(d.get("text", ""), 60))}</div></td>'
                    f'<td><span class="pill t-purple">{esc((d.get("lang_source") or "?").upper())}</span></td>'
                    f'<td style="white-space:nowrap">{fdate(d.get("created_at"))}</td>'
                    f'<td><span class="pill t-blue">{n_translations(d)}</span></td></tr>'
                    for d in docs[:10])
                md(f'<div class="tbl-wrap"><table class="tbl"><thead><tr><th>ID</th><th>Aperçu</th><th>Langue</th>'
                   f'<th>Ajouté le</th><th>Trad.</th></tr></thead><tbody>{rows}</tbody></table></div>')

    with c1:
        with st.container(key="card_stats_services"):
            card_title("server", "Services et infrastructure")
            tiles = [
                ("Mistral AI", svc["mistral"], "sparkles", "t-orange",
                 f'<a class="svc-link" href="{MISTRAL_URL}" target="_blank">Ouvrir la console</a>'),
                ("MongoDB", svc["mongo"], "leaf", "t-green", '<span class="svc-addr">localhost:27017</span>'),
                ("Elasticsearch", svc["elastic"], "search", "t-blue",
                 f'<a class="svc-link" href="{ES_URL}" target="_blank">Ouvrir</a>'),
                ("Kibana", svc["kibana"], "chart", "t-purple",
                 f'<a class="svc-link" href="{KIBANA_URL}" target="_blank">Ouvrir</a>'),
            ]
            md('<div class="svc-grid">' + "".join(
                f'<div class="svc"><div class="svc-ico {tone}">{icon(ico, 22)}</div>'
                f'<div class="svc-name">{name}</div>'
                f'<div class="svc-state"><span class="dot {"on" if ok else "off"}"></span>{"En ligne" if ok else "Hors ligne"}</div>'
                f'{link}</div>'
                for name, ok, ico, tone, link in tiles) + '</div>')
    with c2:
        with st.container(key="card_stats_activity"):
            card_title("activity", f"Activité sur {days} jours",
                       f'<span class="card-sub"><span class="legend-dot" style="display:inline-block;background:{C_PRIMARY}"></span> Documents &nbsp; '
                       f'<span class="legend-dot" style="display:inline-block;background:{C_ORANGE}"></span> Traductions</span>')
            labels = [f"{d.day} {MOIS[d.month - 1]}" for d in dates]
            md(f'<div class="chart-wrap">{area_chart([("Documents", s_docs, C_PRIMARY), ("Traductions", s_tr, C_ORANGE)], labels, 230)}</div>')
