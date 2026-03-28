# ============================================================
# 🌿 GREEN LENS — Synapse 3.0
# ============================================================
# INSTALL:
#   pip install streamlit transformers pandas torch requests beautifulsoup4 rapidfuzz scikit-learn matplotlib seaborn
# RUN:
#   streamlit run app.py
# ============================================================

import io
import re
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from transformers import pipeline

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(page_title="Green Lens", page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
.stApp { background: #000000; font-family: 'Space Grotesk', sans-serif; }
.header-box { background: linear-gradient(90deg, #1a472a, #2d6a4f); border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; border-left: 5px solid #52b788; }
.buzzword-badge { background: #ff6b35; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; margin: 2px; display: inline-block; }
.verify-pass { background: rgba(82,183,136,0.15); border: 1px solid #52b788; border-radius: 8px; padding: 0.7rem 1rem; margin: 0.3rem 0; }
.verify-fail { background: rgba(220,53,69,0.15); border: 1px solid #dc3545; border-radius: 8px; padding: 0.7rem 1rem; margin: 0.3rem 0; }
.verify-warn { background: rgba(255,193,7,0.15); border: 1px solid #ffc107; border-radius: 8px; padding: 0.7rem 1rem; margin: 0.3rem 0; }
.rec-card { background: rgba(31,111,235,0.12); border: 1px solid #1f6feb; border-radius: 10px; padding: 1rem 1.2rem; margin: 0.4rem 0; }
.stButton>button { background: linear-gradient(90deg, #2d6a4f, #52b788); color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem; font-weight: 600; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1 style="color:#52b788; margin:0;">🌿 Green Lens</h1>
    <p style="color:#b7e4c7; margin:0.5rem 0 0 0;">AI-powered greenwashing detector — because "eco-friendly" means nothing without proof.</p>
</div>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────
BUZZWORDS = [
    "eco-friendly", "eco friendly", "natural", "green", "sustainable",
    "environmentally friendly", "clean", "pure", "earth-friendly",
    "eco-conscious", "planet-friendly", "responsible", "ethical",
    "organic", "bio", "non-toxic", "chemical-free", "zero waste",
    "carbon neutral", "net zero", "biodegradable", "recycled",
    "renewable", "conscious", "mindful", "good for the planet"
]

CERTIFICATIONS = [
    "GOTS", "USDA Organic", "B-Corp", "B Corp", "Fair Trade",
    "NPOP", "ISO 14001", "Rainforest Alliance", "FSC", "OEKO-TEX"
]

CARBON_PATTERNS = [
    r"(\d+)\s*%\s*(recycled|renewable|organic|sustainable)",
    r"carbon\s*(neutral|negative|zero|offset)",
    r"net[\s-]zero",
    r"(\d+)\s*(kg|ton|tonne)s?\s*(of\s*)?(CO2|carbon)",
    r"scope\s*[123]",
    r"science[\s-]based\s*targets",
    r"verified\s*carbon\s*standard",
]

MATERIAL_CLAIMS = {
    "organic cotton":     "Should be GOTS or USDA Organic certified",
    "recycled polyester": "Should reference GRS (Global Recycled Standard)",
    "sustainable wood":   "Should be FSC certified",
    "natural rubber":     "Should reference FSC or Rainforest Alliance",
    "fair trade":         "Should reference Fair Trade USA or Fairtrade International",
}

CATEGORY_KEYWORDS = {
    "Food & Beverage":           ["food", "drink", "wine", "whisky", "beer", "coffee", "tea", "nutrition", "snack", "meal", "ingredient", "chocolate", "organic food", "beverage", "edible"],
    "Fashion & Apparel":         ["fashion", "apparel", "cloth", "wear", "textile", "garment", "fabric", "cotton", "shoe", "footwear", "bag", "luggage", "dress", "shirt", "denim", "stitch"],
    "Beauty & Personal Care":    ["skincare", "cosmetic", "beauty", "soap", "shampoo", "deodorant", "lotion", "cream", "hair", "makeup", "fragrance", "serum", "beeswax"],
    "Technology & Software":     ["software", "tech", "app", "digital", "saas", "cyber", "cloud", "data", "ai", "platform", "developer", "microsoft", "salesforce", "crm"],
    "Finance & Wealth":          ["finance", "wealth", "invest", "fund", "bank", "insurance", "capital", "financial", "accounting", "tax"],
    "Education & Media":         ["school", "education", "learn", "literacy", "training", "media", "news", "film", "video", "content", "journal", "magazine", "communication"],
    "Environmental Services":    ["environment", "waste", "recycle", "carbon", "emission", "climate", "solar", "renewable", "plastic", "pollution", "ecology", "conservation"],
    "Health & Wellness":         ["health", "medical", "hospital", "pharma", "biotech", "patient", "fitness", "sport", "wellbeing", "mental health"],
    "Manufacturing & Materials": ["manufactur", "material", "packaging", "factory", "industrial", "supply chain", "engineer", "hardware", "equipment", "construction", "furniture"],
    "Travel & Hospitality":      ["travel", "tourism", "hotel", "restaurant", "catering", "event", "venue", "adventure", "destination", "hospitality"],
    "Agriculture & Farming":     ["farm", "agricultur", "crop", "harvest", "seed", "soil", "plant", "grow", "livestock"],
    "Architecture & Design":     ["architect", "design", "interior", "landscape", "studio", "creative", "brand", "marketing", "graphic"],
    "Consulting & Services":     ["consult", "strateg", "advisory", "impact", "community", "inclusion", "recruit", "hr", "workforce"],
}

VALID_LABELS = [
    "vague environmental claim",
    "specific measurable claim",
    "certified sustainable product",
]

NLI_LABELS = [
    "vague environmental claim without evidence or certification",
    "specific measurable environmental claim supported by data or numbers",
    "certified sustainable product verified by a third-party certification body",
]

NLI_TO_SHORT = {
    "vague environmental claim without evidence or certification":               "vague environmental claim",
    "specific measurable environmental claim supported by data or numbers":     "specific measurable claim",
    "certified sustainable product verified by a third-party certification body": "certified sustainable product",
}

HYPOTHESIS_TEMPLATE = "This text makes a {}."

CERT_RULE_KEYWORDS = [
    'fsc', 'oeko-tex', 'oeko tex', 'b corp', 'b-corp', 'bcorp',
    'leed', 'rainforest alliance', 'usda organic', 'usda', 'iso ',
    'gots', 'fair trade', 'fairtrade', 'certified', 'certification',
    'standard 100', 'organic certified', 'ecocert', 'bluesign',
    'cradle to cradle', 'sa8000', 'leaping bunny', 'vegan society',
    'climate neutral', 'carbon trust', 'demeter', 'soil association',
    'marine stewardship', 'forest stewardship', 'global recycled standard',
    'npop', 'rspo', 'verified by',
]

def rule_based_classify(text: str):
    t = text.lower()
    if any(k in t for k in CERT_RULE_KEYWORDS):
        return "certified sustainable product", 0.95
    has_number = bool(re.search(
        r'\d+\s*%|\d+\s*(kg|ton|tonne|litre|liter|kwh|mwh|mw|gj|co2|scope)', t))
    if has_number:
        return "specific measurable claim", 0.93
    vague_signals = [
        'green', 'eco', 'sustain', 'natural', 'clean', 'responsible',
        'ethical', 'planet', 'earth', 'environment', 'conscious', 'mindful',
        'organic', 'renewable', 'biodegradable', 'recycled', 'zero waste',
    ]
    if any(v in t for v in vague_signals):
        return "vague environmental claim", 0.89
    return None, None

EVAL_CONFIDENCE_THRESH = 0.6
EVAL_TEST_SIZE         = 0.2
EVAL_RANDOM_STATE      = 42

# ── DATA LOADERS ─────────────────────────────────────────────
@st.cache_data
def load_brands():
    try:
        df = pd.read_csv("trusted_brands.csv", quotechar='"', engine="python", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"brand", "certification", "category"}
        if required - set(df.columns):
            raise ValueError("Missing columns")
        return df
    except Exception:
        return pd.DataFrame({
            "brand":         ["Patagonia", "Allbirds", "The Body Shop", "Wholsum Foods", "SOFT OPTIONS"],
            "certification": ["B-Corp",    "B-Corp",   "B-Corp",        "B-Corp",        "GOTS"],
            "category":      ["Fashion & Apparel", "Fashion & Apparel", "Beauty & Personal Care", "Food & Beverage", "Fashion & Apparel"],
            "notes":         ["", "", "", "", ""],
            "detail_url":    ["", "", "", "", ""],
        })

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("dataset.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.dropna(subset=["text", "label"])
        df["label"] = df["label"].str.strip()
        invalid = df[~df["label"].isin(VALID_LABELS)]["label"].unique()
        if len(invalid) > 0:
            st.sidebar.warning(f"Unrecognised labels: {list(invalid)}\nExpected: {VALID_LABELS}")
        return df
    except FileNotFoundError:
        return pd.DataFrame({"text": [], "label": []})
    except Exception as e:
        st.sidebar.error(f"Could not load dataset.csv: {e}")
        return pd.DataFrame({"text": [], "label": []})

CERTIFIED_BRANDS_DB = load_brands()

# ── NLP MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-large")

with st.spinner("🔄 Loading NLP model... (first time ~60 seconds)"):
    classifier = load_model()

# ── CORE FUNCTIONS ────────────────────────────────────────────
def detect_buzzwords(text):
    return [w for w in BUZZWORDS if w in text.lower()]

def check_certifications(text):
    return [c for c in CERTIFICATIONS if c.lower() in text.lower()]

def extract_keywords(text, top_n=10):
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform([text])
        feature_array = vectorizer.get_feature_names_out()
        tfidf_scores = X.toarray()[0]
        keywords = sorted(zip(feature_array, tfidf_scores), key=lambda x: x[1], reverse=True)
        return [word for word, score in keywords[:top_n]]
    except Exception:
        return []

def detect_category(text):
    text_lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
    return max(scores, key=scores.get) if scores else "General"

def recommend_brands(category, n=3):
    if "category" not in CERTIFIED_BRANDS_DB.columns:
        return []
    matches = CERTIFIED_BRANDS_DB[CERTIFIED_BRANDS_DB["category"].str.lower() == category.lower()]
    if matches.empty:
        matches = CERTIFIED_BRANDS_DB
    return matches.head(n).to_dict("records")

def check_brand(text):
    text_normalized = text.lower()
    matches = []
    for _, row in CERTIFIED_BRANDS_DB.iterrows():
        brand_name = str(row.get("brand", "")).lower()
        if not brand_name or len(brand_name) < 4:
            continue
        score = fuzz.partial_ratio(brand_name, text_normalized)
        if score >= 85:
            matches.append({"brand": row.get("brand", ""), "certification": row.get("certification", "Certified"),
                            "category": row.get("category", ""), "score": score})
    return matches

def classify_text(text: str):
    text = text[:2000]
    rule_label, rule_conf = rule_based_classify(text)
    if rule_label is not None and rule_conf >= 0.95:
        return rule_label, rule_conf
    result = classifier(text, NLI_LABELS, hypothesis_template=HYPOTHESIS_TEMPLATE)
    raw_label = result["labels"][0]
    return NLI_TO_SHORT.get(raw_label, raw_label), result["scores"][0]

# ── VERIFICATION ENGINE ───────────────────────────────────────
def get_cert_source(cert):
    return {"GOTS": "global-standard.org/public-database", "B-Corp": "bcorporation.net/find-a-b-corp",
            "USDA Organic": "ams.usda.gov/organic", "Fair Trade": "fairtradecertified.org",
            "FSC": "fsc.org/en/find-a-certificate", "OEKO-TEX": "oeko-tex.com/en/our-standards"
            }.get(cert, "the certification body's official website")

def verify_certification_claims(text, certs_mentioned):
    results = []
    brand_certs = [m["certification"].lower() for m in check_brand(text)]
    for cert in certs_mentioned:
        if any(cert.lower() in bc for bc in brand_certs):
            results.append({"claim": f'Claims "{cert}" certification', "status": "VERIFIED",
                "note": "Brand found in our certified database.", "css": "verify-pass", "icon": "✅"})
        else:
            results.append({"claim": f'Claims "{cert}" certification', "status": "UNVERIFIED",
                "note": f'"{cert}" mentioned but brand not in DB. Verify at {get_cert_source(cert)}.',
                "css": "verify-warn", "icon": "⚠️"})
    return results

def verify_carbon_claims(text):
    results = []
    text_lower = text.lower()
    for pattern in CARBON_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            claim_match = re.search(pattern, text, re.IGNORECASE)
            claim_str = claim_match.group(0) if claim_match else pattern
            has_verifier = any(v in text_lower for v in ["verified", "third-party", "independent", "audit", "science-based targets", "gold standard", "verra"])
            has_number = bool(re.search(r'\d', claim_str))
            if has_verifier and has_number:
                results.append({"claim": claim_str, "status": "CREDIBLE", "note": "Quantified claim with verifier.", "css": "verify-pass", "icon": "✅"})
            elif has_number:
                results.append({"claim": claim_str, "status": "UNVERIFIED", "note": "Has numbers but no third-party verifier.", "css": "verify-warn", "icon": "⚠️"})
            else:
                results.append({"claim": claim_str, "status": "VAGUE", "note": "No numbers or verifier — likely marketing.", "css": "verify-fail", "icon": "🚩"})
    return results

def verify_material_claims(text):
    results = []
    text_lower = text.lower()
    for material, required_cert in MATERIAL_CLAIMS.items():
        if material in text_lower:
            has_cert = any(c.lower() in text_lower for c in CERTIFICATIONS)
            if has_cert:
                results.append({"claim": f'Material claim: "{material}"', "status": "CREDIBLE", "note": "Certification mentioned.", "css": "verify-pass", "icon": "✅"})
            else:
                results.append({"claim": f'Material claim: "{material}"', "status": "UNVERIFIED", "note": f"{required_cert} — not found.", "css": "verify-fail", "icon": "🚩"})
    return results

def generate_explanation(label, buzzwords, certs, brands, verification_results):
    all_issues  = [v["claim"] + " — " + v["note"] for v in verification_results if v["status"] in ["UNVERIFIED", "VAGUE"]]
    verified_ok = [v["claim"] for v in verification_results if v["status"] in ["VERIFIED", "CREDIBLE"]]
    try:
        resp = requests.post("https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 250,
                "messages": [{"role": "user", "content":
                    f"You are a greenwashing auditor. NLP Classification: '{label}'. "
                    f"Vague buzzwords: {buzzwords[:4]}. Certifications mentioned: {certs}. "
                    f"Verified claims: {verified_ok}. Unverified claims: {all_issues}. "
                    f"Certified brands matched: {[b['brand'] for b in brands]}. "
                    f"Write 3 sentences: (1) overall verdict, (2) biggest red flag, (3) what would make it credible. Be direct."}]
            }, timeout=10)
        return resp.json()["content"][0]["text"]
    except Exception:
        if all_issues:
            return f"Key issue: {all_issues[0]}. {len(all_issues)} unverified claim(s) detected. Add third-party certification references."
        return f"{len(verified_ok)} claim(s) appear credible. Low greenwashing risk detected."

def calculate_score(label, buzzwords, certs, verification_results):
    score = 0
    if buzzwords: score += 25
    if not certs: score += 25
    if label == "vague environmental claim": score += 25
    unverified = sum(1 for v in verification_results if v["status"] in ["UNVERIFIED", "VAGUE"])
    verified   = sum(1 for v in verification_results if v["status"] in ["VERIFIED", "CREDIBLE"])
    score += min(unverified * 8, 25)
    score -= min(verified * 5, 15)
    return max(0, min(score, 100))

# ── FIXED URL SCRAPER ─────────────────────────────────────────
def scrape_url(url: str):
    if not url.startswith("http"):
        url = "https://" + url

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
    ]

    for ua in user_agents:
        try:
            session = requests.Session()
            session.headers.update({
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            })
            resp = session.get(url, timeout=12, allow_redirects=True)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "svg"]):
                tag.decompose()

            # Strategy 1: paragraphs
            text = " ".join(p.get_text(separator=" ", strip=True) for p in soup.find_all("p"))

            # Strategy 2: broader tags if paragraphs insufficient
            if len(text.strip()) < 150:
                text = " ".join(t.get_text(separator=" ", strip=True)
                                for t in soup.find_all(["p", "li", "span", "div", "article", "section", "h1", "h2", "h3"]))

            # Strategy 3: full page text
            if len(text.strip()) < 150:
                text = soup.get_text(separator=" ", strip=True)

            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 100:
                return text[:3000]

        except requests.exceptions.SSLError:
            try:
                resp = requests.get(url, timeout=12, verify=False, headers={"User-Agent": ua})
                soup = BeautifulSoup(resp.content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = re.sub(r'\s+', ' ', soup.get_text(separator=" ", strip=True)).strip()
                if len(text) > 100:
                    return text[:3000]
            except Exception:
                continue
        except Exception:
            continue

    return None

# ── SESSION STATE ─────────────────────────────────────────────
if "scraped_text" not in st.session_state:
    st.session_state.scraped_text = ""
if "scrape_error" not in st.session_state:
    st.session_state.scrape_error = ""

# ── MODEL EVALUATION ─────────────────────────────────────────
def evaluate_model(df):
    df_valid = df[df["label"].isin(VALID_LABELS)].copy().reset_index(drop=True)
    if df_valid.empty:
        return [], [], [], pd.DataFrame(), 0.0
    _, df_test = train_test_split(df_valid, test_size=EVAL_TEST_SIZE, random_state=EVAL_RANDOM_STATE, stratify=df_valid["label"])
    df_test = df_test.reset_index(drop=True)
    y_true, y_pred, y_conf, rows = [], [], [], []
    progress = st.sidebar.progress(0); status = st.sidebar.empty(); start = time.time()
    for i, row in df_test.iterrows():
        status.caption(f"Classifying row {i+1}/{len(df_test)}…")
        text, true_label = str(row["text"]), str(row["label"])
        pred_label, conf = classify_text(text)
        y_true.append(true_label); y_pred.append(pred_label); y_conf.append(round(conf, 4))
        rows.append({"text": text[:80] + ("…" if len(text) > 80 else ""), "true_label": true_label,
                     "predicted": pred_label, "confidence": round(conf, 4), "correct": true_label == pred_label})
        progress.progress((i + 1) / len(df_test))
    elapsed = time.time() - start; progress.empty(); status.empty()
    return y_true, y_pred, y_conf, pd.DataFrame(rows), elapsed

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Trusted Brands DB (RAG)")
    cols = [c for c in ["brand", "certification", "category"] if c in CERTIFIED_BRANDS_DB.columns]
    st.dataframe(CERTIFIED_BRANDS_DB[cols], use_container_width=True)
    st.caption(f"{len(CERTIFIED_BRANDS_DB)} verified brands loaded")
    st.markdown("---")
    st.markdown("**Verification Checks:**")
    st.markdown("- ✅ Certification cross-check\n- ✅ Carbon claim validation\n- ✅ Material claim check\n- ✅ Fuzzy brand matching\n- ✅ Category detection\n- ✅ Trusted brand recommendations")
    st.markdown("---")
    st.markdown("### 📊 Model Evaluation")
    st.caption(f"Two-stage classifier: rule-based pre-filter + DeBERTa-v3-large fallback\nStratified 80/20 test split · confidence threshold {EVAL_CONFIDENCE_THRESH}")
    if st.button("▶ Run Evaluation Metrics"):
        df_eval = load_dataset()
        if df_eval.empty:
            st.warning("dataset.csv not found or empty.")
        else:
            with st.spinner("Evaluating on test set…"):
                y_true, y_pred, y_conf, df_rows, elapsed = evaluate_model(df_eval)
            if not y_true:
                st.error("No valid-label rows found.")
            else:
                n_test = len(y_true); uncertain_n = y_pred.count("uncertain"); avg_conf = sum(y_conf)/len(y_conf)
                acc = accuracy_score(y_true, y_pred)
                f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
                f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
                prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                st.markdown("#### 📈 Metrics")
                c1,c2,c3 = st.columns(3)
                c1.metric("Accuracy", f"{acc*100:.2f}%"); c2.metric("F1 (weighted)", f"{f1_w:.4f}"); c3.metric("F1 (macro)", f"{f1_macro:.4f}")
                c4,c5,c6 = st.columns(3)
                c4.metric("Precision (W)", f"{prec:.4f}"); c5.metric("Recall (W)", f"{rec:.4f}"); c6.metric("Avg Confidence", f"{avg_conf:.4f}")
                c7,c8,c9 = st.columns(3)
                c7.metric("F1 (micro)", f"{f1_micro:.4f}"); c8.metric("Test rows", n_test); c9.metric("Uncertain", f"{uncertain_n}/{n_test}")
                st.caption(f"⏱ {elapsed:.1f}s · {elapsed/n_test:.2f}s/row · split {int((1-EVAL_TEST_SIZE)*100)}/{int(EVAL_TEST_SIZE*100)} stratified")
                st.markdown("#### 📋 Classification Report")
                st.code(classification_report(y_true, y_pred, zero_division=0), language="")
                st.markdown("#### 🔲 Confusion Matrix")
                all_labels = sorted(set(y_true + y_pred))
                cm = confusion_matrix(y_true, y_pred, labels=all_labels)
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor("#0d2137"); ax.set_facecolor("#0d2137")
                sns.heatmap(pd.DataFrame(cm, index=all_labels, columns=all_labels),
                    annot=True, fmt="d", cmap="Greens", linewidths=0.5, linecolor="#1e3a5f", ax=ax,
                    annot_kws={"color": "white", "fontsize": 10})
                ax.set_title(f"Accuracy {acc*100:.2f}%  ·  F1 {f1_w:.4f}  ·  n={n_test}", color="#b7e4c7", fontsize=10, pad=10)
                ax.set_xlabel("Predicted", color="#b7e4c7"); ax.set_ylabel("True", color="#b7e4c7")
                ax.tick_params(colors="#b7e4c7", labelsize=8); plt.xticks(rotation=20, ha="right"); plt.tight_layout()
                buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor()); buf.seek(0)
                st.image(buf, use_column_width=True); plt.close()
                st.markdown("#### 🗂 Per-row Results")
                st.dataframe(df_rows.style.applymap(
                    lambda v: "background-color:#1a3a1a;color:#52b788" if v is True else
                              ("background-color:#3a1a1a;color:#dc3545" if v is False else ""),
                    subset=["correct"]), use_container_width=True, height=280)
                st.download_button("⬇ Download evaluation_results.csv",
                    data=df_rows.to_csv(index=False).encode(), file_name="evaluation_results.csv", mime="text/csv")

# ── INPUT UI ─────────────────────────────────────────────────
st.subheader("🔍 Analyze a Product")
input_method = st.radio("Choose input type:", ["Text", "URL"], horizontal=True)
user_text = ""

if input_method == "Text":
    user_text = st.text_area("Enter product description", height=180,
        placeholder='e.g. "Our organic cotton T-shirts are carbon neutral and GOTS certified..."')
else:
    url_input = st.text_input("Enter product URL",
        placeholder="https://www.patagonia.com/our-footprint/",
        help="Works best with brand sustainability pages. Some sites block scrapers — use Text mode as fallback.")

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        scrape_clicked = st.button("🌐 Scrape URL")
    with col_clear:
        if st.button("🗑 Clear"):
            st.session_state.scraped_text = ""
            st.session_state.scrape_error = ""

    if scrape_clicked:
        if not url_input.strip():
            st.warning("Please enter a URL first.")
        else:
            st.session_state.scraped_text = ""
            st.session_state.scrape_error = ""
            with st.spinner(f"Scraping {url_input.strip()} ..."):
                result = scrape_url(url_input.strip())
            if result and len(result.strip()) > 100:
                st.session_state.scraped_text = result
            else:
                st.session_state.scrape_error = (
                    "⚠️ Could not extract enough text from that URL.\n\n"
                    "**Common reasons:** The site blocks bots, requires login, or loads content via JavaScript.\n\n"
                    "**Fix:** Copy the product description text and paste it using the **Text** input mode instead."
                )

    if st.session_state.scrape_error:
        st.error(st.session_state.scrape_error)

    if st.session_state.scraped_text:
        st.success(f"✅ Scraped {len(st.session_state.scraped_text):,} characters successfully!")
        with st.expander("📄 Preview scraped text (first 1000 chars)"):
            st.write(st.session_state.scraped_text[:1000] + "…")
        user_text = st.session_state.scraped_text

# ── ANALYSIS ─────────────────────────────────────────────────
if st.button("🌿 Run Greenwashing Audit"):
    if not user_text.strip():
        st.warning("Please enter a product description or scrape a URL first.")
    else:
        with st.spinner("Running NLP audit + claim verification..."):
            buzz              = detect_buzzwords(user_text)
            certs             = check_certifications(user_text)
            brands            = check_brand(user_text)
            keywords          = extract_keywords(user_text)
            detected_category = detect_category(user_text)
            label, confidence = classify_text(user_text)
            if confidence < EVAL_CONFIDENCE_THRESH:
                label = "uncertain"
            cert_checks       = verify_certification_claims(user_text, certs)
            carbon_checks     = verify_carbon_claims(user_text)
            material_checks   = verify_material_claims(user_text)
            all_verifications = cert_checks + carbon_checks + material_checks
            explanation       = generate_explanation(label, buzz, certs, brands, all_verifications)
            score             = calculate_score(label, buzz, certs, all_verifications)

        # ── DYNAMIC BACKGROUND + BLACK TEXT EVERYWHERE ────────────
        if score < 20:
            bg_color = "#32a858"
        elif score <= 50:
            bg_color = "#9e9a2c"
        else:
            bg_color = "#c72e2e"

        st.markdown(f"""
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        section[data-testid="stSidebar"] {{
            background: {bg_color} !important;
        }}

        /* Black text everywhere after result */
        .stApp, .stApp *,
        [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] *,
        .stMarkdown, .stMarkdown *,
        .stText, .stText *,
        p, span, div, h1, h2, h3, h4, h5, h6, li, label, small,
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
        [data-testid="stCaptionContainer"],
        .stRadio label, .stCheckbox label,
        [data-baseweb="radio"] label, [data-baseweb="checkbox"] label,
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span, [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {{
            color: #000000 !important;
        }}

        /* Dark textboxes after result — keep readable */
        textarea,
        input[type="text"],
        input[type="number"],
        [data-baseweb="textarea"] textarea,
        [data-baseweb="input"] input,
        .stTextArea textarea,
        .stTextInput input {{
            background-color: #111111 !important;
            color: #eeeeee !important;
            border: 1px solid #555 !important;
        }}

        /* Dark notification/info/success/warning boxes */
        [data-testid="stNotification"] {{
            background-color: #111111 !important;
            color: #eeeeee !important;
        }}

        /* Dark dataframe background */
        [data-testid="stDataFrame"] {{
            background-color: #111111 !important;
        }}

        /* Dark expander */
        details, summary,
        [data-testid="stExpander"] > div {{
            background-color: #111111 !important;
            color: #eeeeee !important;
        }}

        /* Dark code blocks */
        .stCodeBlock, pre, code {{
            background-color: #111111 !important;
            color: #eeeeee !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        # ── RESULTS ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Audit Results")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Classification",    label.title())
        col2.metric("Confidence",        f"{confidence:.0%}")
        col3.metric("Buzzwords",         len(buzz))
        col4.metric("Claims Checked",    len(all_verifications))
        col5.metric("Detected Category", detected_category)

        if score < 20:
            score_bg, score_icon, score_label_text = "#2a8558", "✅", "Low Risk — claims appear credible"
        elif score <= 50:
            score_bg, score_icon, score_label_text = "#dbca2a", "⚠️", "Moderate Risk — verify claims independently"
        else:
            score_bg, score_icon, score_label_text = "#c72e2e", "🚨", "High Greenwashing Risk"

        st.markdown(f"""
<style>
div[data-testid="stProgress"] > div > div > div {{ background-color: {score_bg} !important; }}
</style>
<div style="background-color:{score_bg};border-radius:12px;padding:1.2rem 1.5rem;
            margin-bottom:0.8rem;border-left:6px solid rgba(0,0,0,0.2);">
    <p style="color:black;font-size:0.85rem;margin:0 0 0.3rem 0;opacity:0.85;">📈 Greenwashing Risk Score</p>
    <p style="color:black;font-size:2rem;font-weight:700;margin:0 0 0.2rem 0;">
        {score}<span style="font-size:1rem;font-weight:400;opacity:0.8;"> / 100</span></p>
    <p style="color:black;font-size:1rem;margin:0;">{score_icon} {score_label_text}</p>
</div>""", unsafe_allow_html=True)
        st.progress(score / 100)

        st.subheader("🤖 AI Reasoning Summary")
        st.info(explanation)

        if all_verifications:
            st.subheader("🔬 Claim Verification Results")
            st.caption("Each claim checked against certification standards and our brand database.")
            for v in all_verifications:
                st.markdown(f"""
<div class="{v['css']}">
    {v['icon']} <strong>{v['status']}</strong> — {v['claim']}<br>
    <small style="color:#aaa">{v['note']}</small>
</div>""", unsafe_allow_html=True)
        else:
            st.info("No specific verifiable claims detected in text.")

        st.subheader("🚩 Buzzword Flags")
        if buzz:
            st.markdown(" ".join([f'<span class="buzzword-badge">⚑ {w}</span>' for w in buzz]), unsafe_allow_html=True)
        else:
            st.success("No greenwashing buzzwords detected.")

        if keywords:
            st.subheader("🔑 Key Terms")
            st.write(" · ".join(keywords))

        st.subheader("🏆 Certified Brand Check (RAG)")
        if brands:
            for b in brands:
                st.success(f"✅ **{b['brand']}** ({b['certification']}) — verified in database (match: {b['score']}%)")
        else:
            st.warning("No certified brand matched in our database.")

        if score <= 70:
            st.subheader(f"💡 Trusted Alternatives in '{detected_category}'")
            st.caption("These brands are verified B-Corp or GOTS certified in the same category.")
            recs = recommend_brands(detected_category, n=3)
            if recs:
                for r in recs:
                    cert  = r.get("certification", "Certified")
                    notes = str(r.get("notes", ""))[:180] if r.get("notes") else ""
                    url_  = r.get("detail_url", "") or r.get("source", "")
                    st.markdown(f"""
<div class="rec-card">
    🏅 <strong>{r.get('brand','')}</strong>
    <span style="background:#1f6feb33;color:#58a6ff;padding:2px 8px;border-radius:12px;font-size:0.78rem;margin-left:6px">{cert}</span>
    <span style="background:#3fb95022;color:#3fb950;padding:2px 8px;border-radius:12px;font-size:0.78rem;margin-left:4px">{r.get('category','')}</span>
    {"<br><small style='color:#8b949e'>" + notes + "...</small>" if notes else ""}
    {"<br><small><a href='" + url_ + "' target='_blank' style='color:#58a6ff'>🔗 View profile</a></small>" if url_ else ""}
</div>""", unsafe_allow_html=True)
            else:
                st.info("No recommendations available for this category yet.")

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Green Lens | Synapse 3.0 | Stack: Rule-based pre-filter + DeBERTa-v3-large (NLI fallback) + Anthropic Claude + RAG + TF-IDF + Claim Verification")