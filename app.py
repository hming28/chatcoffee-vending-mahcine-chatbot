"""
app.py - ChatCoffee Unmanned Smart Kiosk Backend Server
Powered by Supervised ML Intent Classification + Unified Coffee Knowledge Base
"""

import os
import re
import json
import math
import asyncio
from collections import Counter
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Response, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

try:
    import joblib
except ImportError:
    joblib = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="ChatCoffee NLP Kiosk API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 1. Load ML Artifacts, Slang Dictionary & Knowledge Base
# -------------------------------------------------------------
WEB_MODEL_PATH = os.path.join(BASE_DIR, "models", "chatcoffee_model_web.json")
JOBLIB_MODEL_PATH = os.path.join(BASE_DIR, "models", "chatcoffee_intent_classifier.joblib")
SLANG_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "shortform-dict.json")
if not os.path.exists(SLANG_PATH):
    SLANG_PATH = os.path.join(BASE_DIR, "shortform-dict.json")

KB_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "unified_coffee_kb.csv")
if not os.path.exists(KB_PATH):
    KB_PATH = os.path.join(BASE_DIR, "unified_coffee_kb.csv")
if not os.path.exists(KB_PATH):
    KB_PATH = os.path.join(BASE_DIR, "data", "unified_coffee_kb.csv")

GOLD_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "chatcoffee_gold_850.csv")
if not os.path.exists(GOLD_PATH):
    GOLD_PATH = os.path.join(BASE_DIR, "chatcoffee_gold_850.csv")

# Load & cleanly flatten slang dictionary (exclude metadata, intensifiers, and Kopitiam 'kau')
slang_dict: Dict[str, str] = {}
if os.path.exists(SLANG_PATH):
    with open(SLANG_PATH, "r", encoding="utf-8") as f:
        raw_slang = json.load(f)
        if isinstance(raw_slang, dict):
            for k, v in raw_slang.items():
                if k in ("metadata", "intensifiers"):
                    continue
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sv, str) and sk.lower() != "kau":
                            slang_dict[sk.lower()] = sv.lower()
                elif isinstance(v, str) and k.lower() != "kau":
                    slang_dict[k.lower()] = v.lower()
    print(f"[Preprocess] Loaded {len(slang_dict)} clean slang normalization rules")

LEARNED_SLANG_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "learned_slang.json")
learned_slang_dict: Dict[str, str] = {}
if os.path.exists(LEARNED_SLANG_PATH):
    try:
        with open(LEARNED_SLANG_PATH, "r", encoding="utf-8") as f:
            raw_learned = json.load(f)
            if isinstance(raw_learned, dict):
                for lk, lv in raw_learned.items():
                    if isinstance(lv, str) and lk.strip():
                        learned_slang_dict[lk.strip().lower()] = lv.strip().lower()
                        slang_dict[lk.strip().lower()] = lv.strip().lower()
        print(f"[HITL Learning] Loaded {len(learned_slang_dict)} persistent user-taught slang rules")
    except Exception:
        pass


def save_learned_slang(slang: str, meaning: str) -> Dict[str, str]:
    k = slang.strip().lower()
    v = meaning.strip().lower()
    if k and v and k != "kau":
        learned_slang_dict[k] = v
        slang_dict[k] = v
        os.makedirs(os.path.dirname(LEARNED_SLANG_PATH), exist_ok=True)
        with open(LEARNED_SLANG_PATH, "w", encoding="utf-8") as f:
            json.dump(learned_slang_dict, f, ensure_ascii=False, indent=2)
    return {"slang": k, "meaning": v}


def detect_tone(text: str) -> str:
    t_low = text.lower()
    if re.search(r"\b(?:cepat|urgent|rush|hurry|fast|quick|catch\s+(?:lrt|mrt|bus|train|flight)|late\s+already)\b", t_low) or any(w in text for w in ["赶时间", "快点", "快一点", "很赶", "来不及"]):
        return "Urgent"
    if re.search(r"\b(?:wrong|salah|tak\s+faham|stupid|useless|annoying|frust|kenapa\s+salah)\b", t_low) or any(w in text for w in ["不对", "错了", "听不懂", "搞错"]):
        return "Frustrated"
    if re.search(r"\b(?:recommend|suggest|cadang|rekomen|tak\s+tau|first\s+time|apa\s+sedap|which\s+one)\b", t_low) or any(w in text for w in ["推荐", "什么好喝", "第一次", "不知道喝什么"]):
        return "Exploratory"
    return "Neutral"

# Load Knowledge Base
df_kb = pd.DataFrame()
if os.path.exists(KB_PATH):
    df_kb = pd.read_csv(KB_PATH)
    print(f"[Knowledge Base] Loaded {len(df_kb)} beverage records from {KB_PATH}")

dataset_rows_count = 0
if os.path.exists(GOLD_PATH):
    try:
        dataset_rows_count = len(pd.read_csv(GOLD_PATH))
    except Exception:
        dataset_rows_count = 0

# Load ML Model (prefer synced chatcoffee_model_web.json for 100% frontend/backend parity, fallback to joblib)
web_ml_model = None
tfidf_vec = None
intent_model = None
classes = []
model_name_str = "LogisticRegression (TF-IDF N-Gram 1-2, Dual-Script CJK)"

if os.path.exists(WEB_MODEL_PATH):
    with open(WEB_MODEL_PATH, "r", encoding="utf-8") as f:
        web_ml_model = json.load(f)
    classes = list(web_ml_model["classes"])
    model_name_str = web_ml_model.get("model_name", model_name_str)
    if isinstance(web_ml_model.get("slang_dict"), dict):
        # Merge only flat string entries
        for sk, sv in web_ml_model["slang_dict"].items():
            if isinstance(sv, str) and sk.lower() != "kau":
                slang_dict.setdefault(sk.lower(), sv.lower())
    print(f"[ML Pipeline] Loaded dual-script JSON model from {WEB_MODEL_PATH}")
elif joblib is not None and os.path.exists(JOBLIB_MODEL_PATH):
    ml_artifact = joblib.load(JOBLIB_MODEL_PATH)
    tfidf_vec = ml_artifact["vectorizer"]
    intent_model = ml_artifact["model"]
    classes = list(ml_artifact.get("classes") or ml_artifact.get("intent_labels") or getattr(intent_model, "classes_", []))
    model_name_str = ml_artifact.get("model_name", model_name_str)
    print(f"[ML Pipeline] Loaded joblib intent model from {JOBLIB_MODEL_PATH}")
else:
    raise FileNotFoundError(f"No model found at {WEB_MODEL_PATH} or {JOBLIB_MODEL_PATH}.")


# -------------------------------------------------------------
# 2. Preprocessing, ML Inference & Dual-Script NER
# -------------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    # Apply multi-word learned slang phrases first (longest match)
    for phrase in sorted((k for k in learned_slang_dict if " " in k), key=len, reverse=True):
        t = re.sub(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", learned_slang_dict[phrase], t)
    words = t.split()
    expanded = [slang_dict.get(w, w) for w in words]
    t = " ".join(expanded)
    t = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def predict_intent_ml(c_text: str):
    if web_ml_model is not None:
        vocab = web_ml_model["vocabulary"]
        idf = np.asarray(web_ml_model["idf"], dtype=np.float64)
        coef = np.asarray(web_ml_model["coef"], dtype=np.float64)
        intercept = np.asarray(web_ml_model["intercept"], dtype=np.float64)
        use_sublinear = web_ml_model.get("sublinear_tf", True)

        spaced = re.sub(r"([\u4e00-\u9fff])", r" \1 ", c_text)
        toks = [w for w in spaced.split() if len(w) >= 1]
        ngrams = list(toks) + [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]
        cnts = Counter(ng for ng in ngrams if ng in vocab)

        vec = np.zeros(len(idf), dtype=np.float64)
        for ng, cnt in cnts.items():
            idx = vocab[ng]
            tf = (1.0 + math.log(cnt)) if use_sublinear else float(cnt)
            vec[idx] = tf * idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        logits = coef @ vec + intercept
        logits -= np.max(logits)
        exp_l = np.exp(logits)
        probs = exp_l / np.sum(exp_l)
        best_idx = int(np.argmax(probs))
        pred = classes[best_idx]
        prob_dict = {c: float(round(p, 4)) for c, p in zip(classes, probs)}
        return pred, float(round(probs[best_idx], 4)), prob_dict

    X_vec = tfidf_vec.transform([c_text])
    pred = intent_model.predict(X_vec)[0]
    if hasattr(intent_model, "predict_proba"):
        probs = intent_model.predict_proba(X_vec)[0]
        prob_dict = {c: float(round(p, 4)) for c, p in zip(classes, probs)}
        conf = float(round(probs[list(classes).index(pred)], 4))
    else:
        prob_dict = {c: (1.0 if c == pred else 0.0) for c in classes}
        conf = 1.0
    return pred, conf, prob_dict


def extract_slots(text: str) -> Dict[str, str]:
    t_low = text.lower()
    slots: Dict[str, str] = {}

    # 1. Base Drinks
    if (
        "柠檬茶" in text or "檸檬茶" in text or
        re.search(r"\b(?:teh\s*o?\s*(?:ais|ice|peng|panas)?\s*limau|teh\s*limau|lemon\s*tea|limau\s*(?:tea|teh))\b", t_low)
    ):
        slots["base_drink"] = "Teh O Limau"
        slots["milk_type"] = "None"
    elif "鸳鸯" in text or "鴛鴦" in text or re.search(r"\b(?:cham|yuan yang|yuanyang)\b", t_low):
        slots["base_drink"] = "Cham"
    elif "白咖啡" in text or re.search(r"\b(?:white coffee)\b", t_low):
        slots["base_drink"] = "White Coffee"
    elif re.search(r"\bneslo\b", t_low):
        slots["base_drink"] = "Neslo"
    elif "美禄" in text or "美祿" in text or re.search(r"\b(?:milo|tak kiu|tut kiu)\b", t_low):
        slots["base_drink"] = "Milo"
    elif "美式" in text or re.search(r"\bamericano\b", t_low):
        slots["base_drink"] = "Americano"
        slots["milk_type"] = "None"
    elif "咖啡" in text or re.search(r"\b(?:kopi|coffee)\b", t_low):
        slots["base_drink"] = "Kopi"
    elif "奶茶" in text or "茶" in text or re.search(r"\b(?:teh|tea)\b", t_low):
        slots["base_drink"] = "Teh"

    # 2. Temperature & Ice/Warmth Modifiers
    has_ice_negation = bool(
        re.search(r"\b(?:bukan\s+(?:ais|ice)|don't\s+want\s+ice|do\s+not\s+want\s+ice|tak\s+mau\s+ais)\b", t_low) or
        any(w in text for w in ["不要冰", "不要加冰", "不是冰"])
    )
    if "温" in text or re.search(r"\b(?:suam|warm)\b", t_low):
        slots["temperature"] = "Hot"
        slots["temp_detail"] = "Warm"
    elif "热" in text or "熱" in text or re.search(r"\b(?:panas|hot)\b", t_low):
        slots["temperature"] = "Hot"
        slots["temp_detail"] = "Hot"
    elif has_ice_negation:
        slots["temperature"] = "Hot"
        slots["temp_detail"] = "Hot"
    elif (
        "冰" in text or "冻" in text or "凍" in text or "冷" in text or
        (re.search(r"\b(?:peng|ping|ice|iced|ais|sejuk|dingin|cold)\b", t_low) and not re.search(r"\b(?:price|pricing)\b", t_low))
    ):
        slots["temperature"] = "Iced"

    if "去冰" in text or "走冰" in text or "无冰" in text or re.search(r"\b(?:kosong ais|tanpa ais|no ice|tak nak ais)\b", t_low):
        slots["temperature"] = "Iced"
        slots["ice_level"] = "No Ice"
    elif "少冰" in text or "微冰" in text or re.search(r"\b(?:kurang ais|less ice|sikit ais|tak nak banyak ais)\b", t_low):
        slots["temperature"] = "Iced"
        slots["ice_level"] = "Less Ice"

    # 3. Sweetness (tightened to avoid 'banyak ais' / 'sedikit ais' false positives)
    if (
        "无糖" in text or "無糖" in text or "走糖" in text or
        re.search(r"\b(?:kosong|zero sugar|no sugar|tanpa gula|zero|0)\b|\b0\s*%", t_low)
    ):
        slots["sweetness_level"] = "Zero Sugar"
    elif (
        "少糖" in text or "微糖" in text or "少甜" in text or "半糖" in text or
        re.search(r"\b(?:siu dai|siew dai|less sugar|kurang manis|sedikit manis|sedikit gula|half sugar|less sweet)\b|\b(?:50|30|25)\s*%", t_low)
    ):
        slots["sweetness_level"] = "Less Sugar"
    elif (
        "多糖" in text or "加糖" in text or "多甜" in text or
        re.search(r"\b(?:gah dai|ga dai|extra sweet|lebih manis|banyak manis|banyak gula|extra sugar)\b|\b120\s*%", t_low)
    ):
        slots["sweetness_level"] = "Extra Sugar"
    elif (
        "正常糖" in text or "普通糖" in text or "标准糖" in text or
        re.search(r"\b(?:normal sugar|biasa manis|manis biasa|std sugar)\b|\b100\s*%", t_low)
    ):
        slots["sweetness_level"] = "Normal"
    elif (
        re.search(r"\b(?:kopi\s*o|teh\s*o|cham\s*o|milo\s*o|kopi\s*c|teh\s*c|cham\s*c|teh\s*tarik|white\s*coffee|neslo)\b", t_low) or
        any(w in text for w in ["咖啡乌", "茶乌", "美禄乌", "咖啡c", "茶c", "咖啡昔", "茶昔", "拉茶", "白咖啡", "雀巢美禄"])
    ):
        slots["sweetness_level"] = "Normal"

    # 4. Milk Type (Teh O Limau & Americano locked to None; 炼奶 correctly mapped to Condensed Milk)
    if slots.get("base_drink") in ("Teh O Limau", "Americano"):
        slots["milk_type"] = "None"
    elif (
        "淡奶" in text or "花奶" in text or
        re.search(r"\b(?:kopi c|teh c|cham c|neslo c)\b|\b(?:evaporated|susu sejat|susu cair)\b", t_low)
    ):
        slots["milk_type"] = "Evaporated Milk"
    elif (
        "乌" in text or "黑咖啡" in text or "不加奶" in text or "不要奶" in text or "不用奶" in text or
        re.search(r"\b(?:kopi o|teh o|cham o|milo o)\b|\b(?:kosong milk|no milk|tanpa susu|black|tak perlu susu|tak payah susu|tak nak susu|tak mau susu)\b", t_low) or
        (re.search(r"\bkosong\b", t_low) and not re.search(r"\b(?:susu|condensed|evaporated|c|tarik|biasa)\b", t_low))
    ):
        slots["milk_type"] = "None"
    elif (
        "炼奶" in text or "炼乳" in text or "普通奶" in text or "正常奶" in text or "拉茶" in text or
        re.search(r"\b(?:condensed|susu pekat|tarik|susu biasa|normal milk)\b", t_low)
    ):
        slots["milk_type"] = "Condensed Milk"

    return slots


def lookup_price_and_nutrition(state: Dict[str, Any]) -> Dict[str, Any]:
    base_prices = {
        "Cham": 4.20, "White Coffee": 4.30, "Neslo": 4.20, "Milo": 3.20,
        "Kopi": 3.20, "Teh": 3.20, "Teh O Limau": 3.20, "Americano": 5.00
    }
    base = state.get("base_drink") or "Kopi"
    temp = state.get("temperature")
    milk = state.get("milk_type")
    sweet = state.get("sweetness_level")

    price = base_prices.get(base, 3.70)
    if base not in ("Teh O Limau", "Americano"):
        if milk == "Condensed Milk" and base in ("Kopi", "Teh", "Milo"):
            price += 0.50
        elif milk == "Evaporated Milk":
            price += 0.70
    if temp == "Iced":
        price += 0.80

    calories, sugars_g = 99, 12.0
    if state.get("calories") is not None:
        calories = int(state["calories"])
    if state.get("sugars_g") is not None:
        sugars_g = float(state["sugars_g"])

    contains_lactose = False
    contains_gluten = False
    allergens_summary = "None (Allergen-Free)"
    display_name = base

    # Candidate name resolution for Kopitiam drink combos
    candidate_names = []
    is_iced = (temp == "Iced")
    if base in ("Kopi", "Teh", "Cham", "Milo"):
        if milk == "None":
            if is_iced:
                if sweet == "Zero Sugar":
                    candidate_names.append(f"{base} O Kosong Peng")
                    candidate_names.append(f"{base} O Peng Kosong")
                elif sweet == "Less Sugar":
                    candidate_names.append(f"{base} O Peng Siew Dai")
                    candidate_names.append(f"{base} O Siew Dai")
                candidate_names.append(f"{base} O Peng")
            else:
                if sweet == "Zero Sugar":
                    candidate_names.append(f"{base} O Kosong")
                elif sweet == "Less Sugar":
                    candidate_names.append(f"{base} O Siew Dai")
                candidate_names.append(f"{base} O")
        elif milk == "Evaporated Milk":
            if is_iced:
                if sweet == "Zero Sugar":
                    candidate_names.append(f"{base} C Peng Kosong")
                    candidate_names.append(f"{base} C Kosong")
                elif sweet == "Less Sugar":
                    candidate_names.append(f"{base} C Peng Siew Dai")
                    candidate_names.append(f"{base} C Siew Dai")
                candidate_names.append(f"{base} C Peng")
            else:
                if sweet == "Zero Sugar":
                    candidate_names.append(f"{base} C Kosong")
                elif sweet == "Less Sugar":
                    candidate_names.append(f"{base} C Siew Dai")
                candidate_names.append(f"{base} C")
        else:
            if is_iced:
                if sweet == "Less Sugar":
                    candidate_names.append(f"{base} Peng Siew Dai")
                candidate_names.append(f"{base} Peng")
            else:
                if sweet == "Zero Sugar":
                    candidate_names.append(f"{base} Kosong")
                elif sweet == "Less Sugar":
                    candidate_names.append(f"{base} Siew Dai")
    elif base == "Teh Tarik":
        if is_iced:
            candidate_names.append("Teh Tarik Peng")
        else:
            candidate_names.append("Teh Tarik")
    elif base == "White Coffee":
        if is_iced:
            if sweet == "Less Sugar":
                candidate_names.append("White Coffee Peng Siew Dai")
            candidate_names.append("White Coffee Peng")
        else:
            if sweet == "Less Sugar":
                candidate_names.append("White Coffee Siew Dai")
            elif sweet == "Zero Sugar":
                candidate_names.append("White Coffee Kosong")
            candidate_names.append("White Coffee")
    elif base == "Teh O Limau":
        if is_iced:
            if sweet == "Less Sugar":
                candidate_names.append("Teh O Ais Limau Less Sugar")
            elif sweet == "Zero Sugar":
                candidate_names.append("Teh O Ais Limau Kosong")
            candidate_names.append("Teh O Ais Limau")
            candidate_names.append("Teh O ice limau")
        else:
            candidate_names.append("Teh O Limau Panas")
    elif base == "Americano":
        if is_iced:
            candidate_names.append("Iced Americano")
        else:
            candidate_names.append("Hot Americano")
    candidate_names.append(base)

    if not df_kb.empty:
        sub = pd.DataFrame()
        for cname in candidate_names:
            m = df_kb[df_kb["drink_name"].str.lower() == cname.lower()]
            if not m.empty:
                sub = m
                break
        if sub.empty:
            sub = df_kb[df_kb["drink_name"].str.contains(re.escape(base), case=False, na=False)]
        if temp:
            t_sub = sub[sub["temperature"].str.lower() == temp.lower()]
            if not t_sub.empty:
                sub = t_sub
        if sweet:
            s_sub = sub[sub["sweetness_level"].str.lower() == sweet.lower()]
            if not s_sub.empty:
                sub = s_sub
        else:
            s_sub = sub[sub["sweetness_level"].str.lower() == "normal"]
            if not s_sub.empty:
                sub = s_sub
        if milk:
            m_sub = sub[sub["milk_type"].fillna("None").astype(str).str.lower() == milk.lower()]
            if not m_sub.empty:
                sub = m_sub
        if not sub.empty:
            row = sub.iloc[0]
            display_name = row["drink_name"]
            calories = int(row["calories"]) if pd.notna(row["calories"]) else calories
            sugars_g = float(row["sugars_g"]) if pd.notna(row["sugars_g"]) else sugars_g
            if "contains_lactose" in row:
                contains_lactose = bool(row["contains_lactose"])
            if "contains_gluten" in row:
                contains_gluten = bool(row["contains_gluten"])
            if "allergens_summary" in row and pd.notna(row["allergens_summary"]):
                allergens_summary = str(row["allergens_summary"])
        else:
            if base in ("Milo", "Neslo"):
                contains_gluten = True
                contains_lactose = True
            elif milk in ("Condensed Milk", "Evaporated Milk", "Standard Milk", "Whole Milk", "2% Milk"):
                contains_lactose = True
            elif base not in ("Americano", "Teh O Limau") and milk != "None":
                contains_lactose = True
            parts = []
            if contains_lactose:
                parts.append("Dairy (Lactose)")
            if contains_gluten:
                parts.append("Gluten (Barley Malt)")
            allergens_summary = ", ".join(parts) if parts else "None (Allergen-Free)"
    else:
        if base in ("Milo", "Neslo"):
            contains_gluten = True
            contains_lactose = True
        elif milk in ("Condensed Milk", "Evaporated Milk", "Standard Milk", "Whole Milk", "2% Milk"):
            contains_lactose = True
        elif base not in ("Americano", "Teh O Limau") and milk != "None":
            contains_lactose = True
        parts = []
        if contains_lactose:
            parts.append("Dairy (Lactose)")
        if contains_gluten:
            parts.append("Gluten (Barley Malt)")
        allergens_summary = ", ".join(parts) if parts else "None (Allergen-Free)"

    return {
        "display_name": display_name,
        "price_rm": round(price, 2),
        "calories": calories,
        "sugars_g": sugars_g,
        "contains_lactose": contains_lactose,
        "contains_gluten": contains_gluten,
        "allergens_summary": allergens_summary
    }


def detect_lang(text: str) -> str:
    if not text:
        return "en"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    t_low = text.lower()
    bm_words = [
        "nak", "mau", "satu", "dua", "tiga", "panas", "sejuk", "ais", "kurang", "manis",
        "tak", "xnak", "taknak", "tanpa", "gula", "susu", "cair", "pekat", "batal", "semula",
        "bayar", "boleh", "sedap", "minum", "cawan", "tambah", "bancuh", "tolong", "bagi",
        "saya", "sihat", "berapa", "apa", "ini", "tu", "ni", "ke", "lah", "ia", "ada", "air",
        "laktosa", "alahan", "alergen", "intoleransi", "tinggi", "rendah", "gemuk", "bikin",
        "pilihan", "senarai", "cadang", "cadangan", "saran", "sarankan", "sedikit", "minuman"
    ]
    en_words = [
        "the", "is", "this", "cup", "drink", "drinks", "high", "calorie", "calories", "fattening", "make",
        "gain", "weight", "what", "which", "how", "much", "does", "have", "contain", "contains",
        "lactose", "gluten", "allergy", "allergic", "allergen", "allergens", "intolerant",
        "intolerance", "safe", "recommend", "recommendation", "suggest", "healthy", "diet", "free",
        "any", "options", "choices"
    ]
    bm_score = sum(1 for w in bm_words if re.search(r"\b" + re.escape(w) + r"\b", t_low))
    en_score = sum(1 for w in en_words if re.search(r"\b" + re.escape(w) + r"\b", t_low))
    if bm_score > en_score and bm_score > 0:
        return "bm"
    if en_score > bm_score and en_score > 0:
        return "en"
    if bm_score > 0:
        return "bm"
    if en_score > 0:
        return "en"
    return "bm" if any(w in t_low for w in ["nak", "ni", "ke", "ada", "air", "kalori", "laktosa"]) else "en"


def query_healthy_drinks_from_kb(lang: str) -> str:
    if not df_kb.empty:
        kopitiam = df_kb[
            (df_kb["channel"].str.contains("Kopitiam", case=False, na=False)) &
            (df_kb["sugars_g"] <= 8)
        ].sort_values(["calories", "sugars_g"]).drop_duplicates(subset=["drink_name"]).head(3)
        if not kopitiam.empty:
            items_txt = [
                f"{r['drink_name']} ({int(r['calories'])} kcal, {r['sugars_g']:g}g sugar, RM {float(r['price_rm']):.2f})"
                for _, r in kopitiam.iterrows()
            ]
            joined = "、".join(items_txt) if lang == "zh" else ", ".join(items_txt)
            if lang == "zh":
                return f"健康低卡推荐（来自营养知识库）：{joined}。需要为您点哪一杯？"
            elif lang == "bm":
                return f"Pilihan sihat & rendah kalori (KB Nutrisi): {joined}. Nak saya siapkan yang mana?"
            else:
                return f"Healthy & low-calorie recommendations (from Nutrition KB): {joined}. Which one would you like?"
    if lang == "zh":
        return "健康推荐：Kopi O Kosong（24 kcal，0g 糖，RM 3.20）或 Iced Americano（10 kcal，0g 糖，RM 5.80）！需要为您来一杯吗？"
    elif lang == "bm":
        return "Untuk pilihan sihat & rendah kalori, kami cadangkan Kopi O Kosong (24 kcal, 0g gula, RM 3.20) atau Iced Americano (10 kcal, RM 5.80)! Nak saya siapkan sekarang?"
    else:
        return "For healthy and low-calorie options, we recommend Kopi O Kosong (24 kcal, 0g sugar, RM 3.20) or Iced Americano (10 kcal, RM 5.80)! Would you like one?"


def classify_calorie_tier(cal: int, drink_name: str, sugar: float, lang: str) -> str:
    if cal < 50:
        if lang == "zh":
            return f"【{drink_name}】约 {cal} kcal（含糖 {sugar:g}g），属于【第 1 级：极低卡（< 50 kcal）】！非常健康，0 蔗糖且近乎零脂肪，非常适合减脂、控糖或日常健身人士饮用，完全不会发胖！"
        elif lang == "bm":
            return f"【{drink_name}】 mengandungi kira-kira {cal} kcal ({sugar:g}g gula), tergolong dalam 【Tahap 1: Sangat Rendah Kalori (< 50 kcal)】! Sangat sihat, 0 gula tambahan dan hampir sifar lemak. Amat sesuai untuk diet kurus badan atau kawalan gula, langsung tidak menggemukkan!"
        else:
            return f"【{drink_name}】 contains approx. {cal} kcal ({sugar:g}g sugar), classified as 【Tier 1: Very Low Calorie (< 50 kcal)】! Very healthy with zero added sugar and virtually zero fat. Perfectly suitable for weight loss and diet-conscious lifestyles—won't make you gain weight!"
    elif cal <= 120:
        if lang == "zh":
            return f"【{drink_name}】约 {cal} kcal（含糖 {sugar:g}g），属于【第 2 级：适中微甜（50 - 120 kcal）】！热量与糖分适中，属于轻盈清爽型饮品。适量饮用不会造成身体负担，是兼顾风味与健康的平衡之选！"
        elif lang == "bm":
            return f"【{drink_name}】 mengandungi kira-kira {cal} kcal ({sugar:g}g gula), tergolong dalam 【Tahap 2: Rendah-Sederhana (50 - 120 kcal)】! Kalori dan kemanisan yang sederhana serta menyegarkan. Pilihan seimbang yang tidak membebankan pengambilan kalori harian anda!"
        else:
            return f"【{drink_name}】 contains approx. {cal} kcal ({sugar:g}g sugar), classified as 【Tier 2: Low-to-Moderate (50 - 120 kcal)】! Moderate calories with light sweetness. A well-balanced, refreshing option that satisfies cravings without heavy caloric burden!"
    elif cal <= 180:
        if lang == "zh":
            return f"【{drink_name}】约 {cal} kcal（含糖 {sugar:g}g），属于【第 3 级：常规传统茶室水平（120 - 180 kcal）】！热量主要来自甜炼乳与糖分。如果是为了控糖或减脂，建议选择「少甜（Siew Dai）」减少炼乳，或者改选无奶黑咖啡（Kopi O / Americano）哦！"
        elif lang == "bm":
            return f"【{drink_name}】 mengandungi kira-kira {cal} kcal ({sugar:g}g gula), tergolong dalam 【Tahap 3: Sedang-Tinggi / Piawai Kopitiam (120 - 180 kcal)】! Kalori ini datang terutamanya daripada susu pekat manis. Jika ingin kawal berat badan atau gula, disyorkan pilih 'Kurang Manis (Siew Dai)' atau beralih ke kopi hitam (Kopi O / Americano)!"
        else:
            return f"【{drink_name}】 contains approx. {cal} kcal ({sugar:g}g sugar), classified as 【Tier 3: Standard Kopitiam (120 - 180 kcal)】! Calories primarily originate from sweetened condensed milk and sugar. If watching calories or sugar, we suggest ordering 'Less Sugar (Siew Dai)' or switching to black coffee (Kopi O / Americano)!"
    else:
        if lang == "zh":
            return f"【{drink_name}】约 {cal} kcal（含糖 {sugar:g}g），属于【第 4 级：偏高能量（> 180 kcal）】！由于加入了麦芽浓缩粉、可可或多重炼乳，热量与糖分偏高，多喝容易增重发胖。建议当作偶尔犒赏；若想喝得轻盈，建议改选少甜或 Kopi O Kosong！"
        elif lang == "bm":
            return f"【{drink_name}】 mengandungi kira-kira {cal} kcal ({sugar:g}g gula), tergolong dalam 【Tahap 4: Kalori Tinggi (> 180 kcal)】! Mengandungi tenaga dan gula yang tinggi daripada serbuk malt dan susu pekat manis. Sesuai sebagai hidangan tenaga sesekali; untuk kurangkan kalori, pilih versi kurang manis atau Kopi O Kosong!"
        else:
            return f"【{drink_name}】 contains approx. {cal} kcal ({sugar:g}g sugar), classified as 【Tier 4: High Calorie (> 180 kcal)】! Rich in malt powder, cocoa, and condensed milk with higher energy and sugar content. Best enjoyed as an occasional treat; for diet or fitness goals, opt for less sweetness or Kopi O Kosong instead!"


def get_drink_allergen_reply(drink_name: str, has_lactose: bool, has_gluten: bool, inquiry_type: str, lang: str) -> str:
    if inquiry_type == "lactose":
        if has_lactose:
            if lang == "zh":
                return f"这杯【{drink_name}】含有乳糖（因为含有甜炼乳/淡奶）哦！如果您有乳糖不耐受，建议为您换成 100% 无奶的健康饮品，如黑咖啡（Kopi O / Americano）或纯茶（Teh O / Teh O Limau），完全不含乳糖！"
            elif lang == "bm":
                return f"Cawan 【{drink_name}】 ini mengandungi laktosa (daripada susu pekat/sejat)! Jika anda ada intoleransi laktosa, kami syorkan minuman tanpa susu seperti Kopi O, Americano, Teh O, atau Teh O Limau—100% bebas laktosa!"
            else:
                return f"This cup of 【{drink_name}】 contains lactose (from condensed/evaporated milk)! If you are lactose intolerant, we recommend dairy-free alternatives such as Kopi O, Americano, Teh O, or Teh O Limau—all 100% lactose-free!"
        else:
            if lang == "zh":
                return f"这杯【{drink_name}】完全不含乳糖（无任何奶制品成分）！乳糖不耐受的朋友可以放心安心饮用！"
            elif lang == "bm":
                return f"Cawan 【{drink_name}】 ini 100% bebas laktosa (tiada sebarang produk susu)! Selamat dan sesuai dinikmati oleh sesiapa yang ada intoleransi laktosa!"
            else:
                return f"This cup of 【{drink_name}】 is completely dairy and lactose-free (contains zero milk products)! Perfectly safe for anyone with lactose intolerance!"
    elif inquiry_type == "gluten":
        if has_gluten:
            if lang == "zh":
                return f"这杯【{drink_name}】含有麸质（含有 Protomalt 大麦麦芽提取物）！如果您对麸质过敏或患有乳糜泻，建议选择纯咖啡（Kopi O / Americano）或纯茶（Teh O），都是天然 100% 无麸质的！"
            elif lang == "bm":
                return f"Cawan 【{drink_name}】 ini mengandungi gluten (ekstrak malt barli Protomalt)! Jika anda ada alahan gluten, kami syorkan kopi tulen (Kopi O / Americano) atau teh (Teh O) yang sememangnya 100% bebas gluten!"
            else:
                return f"This cup of 【{drink_name}】 contains gluten (due to Protomalt barley malt extract)! If you have a gluten allergy or celiac condition, we recommend pure coffee (Kopi O / Americano) or tea (Teh O), which are naturally 100% gluten-free!"
        else:
            if lang == "zh":
                return f"这杯【{drink_name}】完全不含麸质（天然无大麦或小麦成分）！对麸质敏感或过敏的朋友请放心饮用！"
            elif lang == "bm":
                return f"Cawan 【{drink_name}】 ini 100% bebas gluten (tiada bahan barli atau gandum)! Selamat diminum untuk mereka yang ada alahan atau sensitiviti gluten!"
            else:
                return f"This cup of 【{drink_name}】 is 100% gluten-free (contains no barley, wheat, or malt)! Completely safe for anyone with gluten sensitivity or celiac disease!"
    else:
        if lang == "zh":
            lac_str = "含（来自炼乳/淡奶）" if has_lactose else "不含"
            glu_str = "含（大麦麦芽提取物）" if has_gluten else "不含"
            advice = "如需避开过敏原，推荐选择 100% 无奶无麸质的 Kopi O Kosong、Teh O 或 Americano！" if (has_lactose or has_gluten) else "此饮品天然安全，过敏人士可放心饮用！"
            return f"【{drink_name}】的过敏原检测：乳糖【{lac_str}】、麸质【{glu_str}】。{advice}"
        elif lang == "bm":
            lac_str = "Mengandungi (Susu Pekat/Sejat)" if has_lactose else "Bebas"
            glu_str = "Mengandungi (Ekstrak Malt Barli)" if has_gluten else "Bebas"
            advice = "Untuk elakkan alergen, kami syorkan Kopi O Kosong, Teh O, atau Americano!" if (has_lactose or has_gluten) else "Minuman ini bebas alergen dan selamat dinikmati!"
            return f"Semakan alergen untuk 【{drink_name}】: Laktosa: {lac_str}, Gluten: {glu_str}. {advice}"
        else:
            lac_str = "Contains (Dairy)" if has_lactose else "Free"
            glu_str = "Contains (Barley Malt)" if has_gluten else "Free"
            advice = "To avoid allergens, we recommend allergen-free options like Kopi O Kosong, Teh O, or Americano!" if (has_lactose or has_gluten) else "This drink is allergen-free and safe to enjoy!"
            return f"Allergen check for 【{drink_name}】: Lactose: {lac_str}, Gluten: {glu_str}. {advice}"


def get_allergen_recommendation(allergen_type: str, lang: str) -> str:
    if allergen_type == "lactose":
        if lang == "zh":
            return "为您推荐 100% 不含乳糖的健康饮品：Kopi O Kosong（24 kcal，RM 3.20）、Teh O Kosong（24 kcal，RM 3.20）、Teh O Ais Limau（88 kcal，RM 4.00）以及 Iced Americano（10 kcal，RM 5.80）！全部不含任何牛奶或炼乳，乳糖不耐受可安心饮用。需要为您来一杯吗？"
        elif lang == "bm":
            return "ChatCoffee syorkan pilihan 100% bebas laktosa & bebas susu: Kopi O Kosong (24 kcal, RM 3.20), Teh O Kosong (24 kcal, RM 3.20), Teh O Ais Limau (88 kcal, RM 4.00), dan Iced Americano (10 kcal, RM 5.80)! Tiada sebarang produk tenusu. Nak saya sediakan yang mana satu?"
        else:
            return "ChatCoffee recommends 100% dairy and lactose-free options: Kopi O Kosong (24 kcal, RM 3.20), Teh O Kosong (24 kcal, RM 3.20), Teh O Ais Limau (88 kcal, RM 4.00), and Iced Americano (10 kcal, RM 5.80)! All are completely milk-free. Which one would you like?"
    elif allergen_type == "gluten":
        if lang == "zh":
            return "为您推荐天然无麸质（Gluten-Free）饮品：纯咖啡系列（Kopi O、Kopi、Americano）以及茶饮系列（Teh O、Teh Tarik、Teh O Limau）均不含任何麦芽或小麦麸质！请避免点选 Milo 或 Neslo（含大麦麦芽提取物）。请问您想来哪一杯？"
        elif lang == "bm":
            return "ChatCoffee syorkan pilihan bebas gluten semula jadi: semua kopi tulen (Kopi O, Kopi, Americano) dan teh (Teh O, Teh Tarik, Teh O Limau) adalah 100% bebas gluten! Sila elakkan Milo dan Neslo kerana mengandungi ekstrak malt barli. Nak minum yang mana boss?"
        else:
            return "ChatCoffee recommends naturally Gluten-Free drinks: all pure coffee (Kopi O, Kopi, Americano) and tea options (Teh O, Teh Tarik, Teh O Limau) are 100% gluten-free! Please avoid Milo and Neslo as they contain barley malt extract. What would you like to order?"
    else:
        if lang == "zh":
            return "为您推荐 100% 无乳糖且无麸质的安全饮品：Kopi O Kosong（24 kcal，RM 3.20）、Teh O Kosong（24 kcal，RM 3.20）、Teh O Ais Limau（88 kcal，RM 4.00）以及 Iced Americano（10 kcal，RM 5.80）！完全不含乳制品及麦芽麸质，请问需要为您准备哪一杯？"
        elif lang == "bm":
            return "ChatCoffee syorkan pilihan 100% bebas alergen (tanpa laktosa & tanpa gluten): Kopi O Kosong (24 kcal, RM 3.20), Teh O Kosong (24 kcal, RM 3.20), Teh O Ais Limau (88 kcal, RM 4.00), dan Iced Americano (10 kcal, RM 5.80)! Nak saya sediakan yang mana satu?"
        else:
            return "ChatCoffee recommends 100% allergen-free options (both lactose-free and gluten-free): Kopi O Kosong (24 kcal, RM 3.20), Teh O Kosong (24 kcal, RM 3.20), Teh O Ais Limau (88 kcal, RM 4.00), and Iced Americano (10 kcal, RM 5.80)! Which one would you prefer?"


def get_no_drink_prompt(query_type: str, lang: str) -> str:
    if query_type == "calorie":
        if lang == "zh":
            return "您目前还没有选择饮品哦！请先告诉我您想了解哪一杯（例如 Kopi、Teh Tarik 或 Americano），或者直接点一杯，我来为您评估热量！"
        elif lang == "bm":
            return "Anda belum pilih sebarang minuman lagi! Sila beritahu minuman mana yang ingin disemak (cth. Kopi, Teh Tarik, atau Americano), atau buat pesanan dahulu untuk saya semak kalorinya!"
        else:
            return "You haven't selected a drink yet! Please let me know which drink you'd like to check (e.g., Kopi, Teh Tarik, or Americano), or order one and I'll assess its calories for you!"
    else:
        if lang == "zh":
            return "您目前还没选择饮品哦！请先告诉我您想了解哪一杯，或者我可以为您推荐 100% 无乳糖、无麸质的纯茶与黑咖啡（如 Kopi O、Teh O、Americano）！"
        elif lang == "bm":
            return "Anda belum pilih sebarang minuman lagi! Sila pilih minuman dahulu, atau saya boleh syorkan pilihan 100% bebas laktosa & bebas gluten seperti Kopi O, Teh O, atau Americano!"
        else:
            return "You haven't selected a drink yet! Please tell me which drink you'd like to check, or I can recommend 100% lactose-free and gluten-free options like Kopi O, Teh O, or Americano!"


# -------------------------------------------------------------
# 3. Request Schema & Session Management
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    utterance: str
    session_id: Optional[str] = "default"
    state: Optional[Dict[str, Any]] = None


MAX_SESSIONS = 500
ALLOWED_STATE_KEYS = {
    "status", "base_drink", "temperature", "sweetness_level", "milk_type",
    "temp_detail", "ice_level", "pending_slot", "price_rm", "calories",
    "sugars_g", "caffeine_mg", "contains_lactose", "contains_gluten", "allergens_summary", "history"
}
sessions: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            oldest_key = next(iter(sessions))
            sessions.pop(oldest_key, None)
        sessions[session_id] = {
            "status": "IDLE",
            "base_drink": None,
            "temperature": None,
            "sweetness_level": None,
            "milk_type": None,
            "temp_detail": "Hot",
            "ice_level": "Normal Ice",
            "pending_slot": None,
            "history": []
        }
    return sessions[session_id]


# -------------------------------------------------------------
# 4. API Endpoints
# -------------------------------------------------------------
def get_groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        env_file = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_file):
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("GROQ_API_KEY="):
                            key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass
    return key


class TTSRequest(BaseModel):
    text: str
    lang: Optional[str] = "en"
    voice: Optional[str] = None
    tone: Optional[str] = "female"


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "brand": "ChatCoffee",
        "model": model_name_str,
        "dataset_rows": dataset_rows_count or 850,
        "classes": list(classes),
        "kb_items": len(df_kb),
        "stt_engine": "Groq Whisper Large V3 / WebSpeech",
        "tts_engine": "Microsoft Edge-TTS Neural"
    }


@app.post("/api/transcribe")
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    lang: Optional[str] = Form("ms"),
    model: Optional[str] = Form("whisper-large-v3")
):
    groq_key = get_groq_api_key()
    if not groq_key:
        return JSONResponse(status_code=503, content={"error": "GROQ_API_KEY is not configured on server", "status": "error"})

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) > 10 * 1024 * 1024:
            return JSONResponse(status_code=413, content={"error": "Audio file exceeds 10MB limit", "status": "error"})

        filename = file.filename or "recording.webm"
        mime_type = file.content_type or "audio/webm"
        selected_model = model if model in ["whisper-large-v3", "whisper-large-v3-turbo"] else "whisper-large-v3"

        prompt = "Pesanan Kopitiam: Kopi, Teh C Peng, Cham, Siu Dai, Kosong, Panas, Kurang Manis, Milo, Ais, Limau, Satu Milo suam, Kopi O"
        if lang == "zh":
            prompt = "南洋咖啡店点单：咖啡，奶茶，鸳鸯，美禄，少甜，无糖，冰，热，少冰"
        elif lang == "en":
            prompt = "Kopitiam drinks order: Kopi, Teh, Milo, Cham, less sugar, no sugar, iced, hot"

        data = {"model": selected_model, "prompt": prompt, "temperature": "0.0"}
        if lang and lang != "auto":
            data["language"] = lang

        def _call_groq():
            return requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {groq_key}"},
                files={"file": (filename, audio_bytes, mime_type)},
                data=data,
                timeout=20
            )

        res = await asyncio.to_thread(_call_groq)
        if res.status_code == 200:
            result = res.json()
            return {"text": result.get("text", "").strip(), "status": "success", "engine": f"Groq {selected_model}"}
        return JSONResponse(status_code=res.status_code, content={"error": "Transcription service error", "status": "error"})
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Audio processing failed", "status": "error"})


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Text is empty"})
    if len(text) > 500:
        return JSONResponse(status_code=400, content={"error": "Text exceeds 500 character limit"})

    clean_speech = re.sub(r"（[^）]*）|\([^)]*\)", "", text).replace("*", "").replace("#", "").strip()
    if not clean_speech:
        clean_speech = text

    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", clean_speech))
    has_malay = bool(re.search(r"\b(?:boss|sila|minum|panas|sejuk|siap|boleh|pesanan|maaf|mesin|cawan|satu|gula|susu|nak|tukar|kurang|manis|terima|kasih|jom)\b", clean_speech.lower()))

    tone = (req.tone or "female").lower()
    if req.voice:
        voice = req.voice
    elif has_chinese:
        voice = "zh-CN-YunxiNeural" if tone == "male" else "zh-CN-XiaoxiaoNeural"
    elif has_malay:
        voice = "ms-MY-OsmanNeural" if tone == "male" else "ms-MY-YasminNeural"
    else:
        # Use valid Singapore English neural voices (en-MY-* does not exist in Edge-TTS)
        voice = "en-SG-WayneNeural" if tone == "male" else "en-SG-LunaNeural"

    try:
        if edge_tts is None:
            raise RuntimeError("edge-tts package is not installed")
        communicate = edge_tts.Communicate(clean_speech, voice, rate="+25%")
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return Response(content=bytes(audio_data), media_type="audio/mpeg", headers={"X-Voice-Used": voice})
    except Exception as e:
        try:
            glang = "zh-CN" if has_chinese else ("ms" if has_malay else "en")
            g_url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={requests.utils.quote(clean_speech)}&tl={glang}&client=tw-ob"
            g_res = await asyncio.to_thread(requests.get, g_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if g_res.status_code == 200:
                return Response(content=g_res.content, media_type="audio/mpeg", headers={"X-Voice-Used": f"Google-Natural-{glang}"})
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"error": "Speech synthesis failed"})


@app.get("/api/tts")
async def tts_get_endpoint(text: str, voice: Optional[str] = None, tone: Optional[str] = "female"):
    return await tts_endpoint(TTSRequest(text=text, voice=voice, tone=tone))


class SlangLearnRequest(BaseModel):
    slang: str
    meaning: str


@app.get("/api/slang")
def get_learned_slang_endpoint():
    return {
        "total_slang_rules": len(slang_dict),
        "learned_count": len(learned_slang_dict),
        "learned_slang": learned_slang_dict
    }


@app.post("/api/learn_slang")
def learn_slang_endpoint(req: SlangLearnRequest):
    if not req.slang.strip() or not req.meaning.strip():
        return JSONResponse(status_code=400, content={"error": "Both slang and meaning are required"})
    saved = save_learned_slang(req.slang, req.meaning)
    return {
        "status": "learned",
        "rule": saved,
        "learned_count": len(learned_slang_dict)
    }


@app.post("/api/predict_intent")
def predict_intent_endpoint(req: ChatRequest):
    c_text = clean_text(req.utterance)
    pred, conf, prob_dict = predict_intent_ml(c_text)
    return {
        "raw_text": req.utterance,
        "clean_text": c_text,
        "predicted_intent": pred,
        "confidence": conf,
        "tone": detect_tone(req.utterance),
        "probabilities": prob_dict,
        "slots": extract_slots(c_text)
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    utterance = req.utterance.strip()
    if len(utterance) > 500:
        return JSONResponse(status_code=400, content={"error": "Utterance exceeds 500 character limit"})

    session_id = req.session_id or "default"
    if req.state is not None and isinstance(req.state, dict):
        state = {k: v for k, v in req.state.items() if k in ALLOWED_STATE_KEYS}
    else:
        state = get_session(session_id)

    # Auto-reset completed order state upon new utterance
    if state.get("status") in ("CONFIRMED", "DISPENSING"):
        state.update({
            "status": "IDLE", "base_drink": None, "temperature": None,
            "sweetness_level": None, "milk_type": None, "temp_detail": "Hot",
            "ice_level": "Normal Ice", "pending_slot": None, "price_rm": 0.0
        })

    is_zh = bool(re.search(r"[\u4e00-\u9fff]", utterance))
    t_low = utterance.lower()

    if not utterance:
        response = "请问有什么可以帮您？可随时说出想喝的饮品。" if is_zh else "Boleh saya bantu boss? Sila beritahu minuman yang diingini."
        return {
            "response": response, "intent": "fallback", "model_intent": "fallback",
            "confidence": 0.0, "probabilities": {}, "clean_text": "", "slots": {}, "state": state
        }

    # 0. Persistent Human-in-the-Loop Slang Teaching (e.g., "teach: kopi beng = kopi peng")
    teach_match = re.match(r"^(?:teach|learn|slang|教俚语|学习)\s*[:：]?\s*(.+?)\s*(?:=|->|=>|means|是)\s*(.+)$", utterance, re.IGNORECASE)
    if teach_match:
        new_slang, new_meaning = teach_match.group(1).strip(), teach_match.group(2).strip()
        save_learned_slang(new_slang, new_meaning)
        response = (
            f"已永久学会新俚语：「{new_slang}」→「{new_meaning}」（已保存至 learned_slang.json，下次点单立即生效）。"
            if is_zh else
            f"Got it boss! Permanently learned slang '{new_slang}' -> '{new_meaning}' (saved to learned_slang.json for future orders)."
        )
        return {
            "response": response, "intent": "learn_slang", "model_intent": "learn_slang",
            "tone": "Neutral", "confidence": 1.0, "probabilities": {}, "clean_text": utterance.lower(),
            "slots": {}, "state": state
        }

    detected_tone = detect_tone(utterance)
    c_text = clean_text(utterance)
    pred_intent, conf, prob_dict = predict_intent_ml(c_text)
    # Extract slots from both raw utterance and learned-slang normalized text
    extracted_slots = extract_slots(c_text)
    for sk, sv in extract_slots(utterance).items():
        extracted_slots.setdefault(sk, sv)
    was_awaiting_confirm = state.get("status") == "AWAITING_CONFIRMATION"

    # 1. Single-cup vending machine limit
    is_ingredient_mod = bool(re.search(r"\b(?:tambah|extra)\s+(?:gula|manis|susu|ais|kurang)\b", t_low))
    is_multi_cup = (
        bool(re.search(r"\b(?:[23]\s+cups?|two\s+cups?|three\s+cups?|dua\s+cawan|tiga\s+cawan|two\s+coffees?|three\s+coffees?|another\s+cup|extra\s+cup|add\s+another|tambah\s+lagi)\b", t_low)) or
        bool(re.search(r"\btambah\s+(?:cawan|gelas|cup|drink|air|satu|lagi\s+satu)\b", t_low)) or
        any(w in utterance for w in ["再加一杯", "加多一杯", "加一杯", "两杯", "三杯", "两杯咖啡", "三杯咖啡"])
    )
    if is_multi_cup and not is_ingredient_mod:
        response = (
            "抱歉老板！ChatCoffee 无人咖啡机每次只能单杯出料（Single-Cup Dispenser Limit）。请先确认并取走当前这杯，再点下一杯哦！"
            if is_zh else
            "Maaf boss! Mesin ChatCoffee ini cuma boleh bancuh 1 cawan pada satu masa (Single-Cup Dispenser Limit). Sila sahkan & ambil cawan ini dulu ya!"
        )
        return {
            "response": response, "intent": "single_cup_limit", "model_intent": pred_intent,
            "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # 2. Reset / Cancel command
    if t_low in ["reset", "clear", "semula", "batal", "cancel", "重置", "取消", "取消订单"] or re.search(r"\b(?:cancel\s+(?:my\s+)?order|batal\s+pesanan)\b", t_low):
        state.update({
            "status": "IDLE", "base_drink": None, "temperature": None,
            "sweetness_level": None, "milk_type": None, "temp_detail": "Hot",
            "ice_level": "Normal Ice", "pending_slot": None
        })
        response = "好的，已重置订单！请问今天想喝什么？" if is_zh else "Pesanan dikosongkan. Jom order semula boss! Nak minum apa hari ini?"
        return {
            "response": response, "intent": "system_reset", "model_intent": pred_intent,
            "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": {}, "state": state
        }

    # 3. Strict confirmation check (ONLY when already awaiting confirmation from a previous turn)
    has_modify_cue = bool(re.search(r"\b(?:change|tukar|ganti|switch|instead|bukan|ubah|wait)\b", t_low) or re.search(r"[换改]", utterance))
    is_strict_confirm = bool(
        re.match(
            r"^(?:(?:ok(?:ay)?|yes|ya|yep|yeah|sure|boleh|ngam|betul)[\s,，]+)*"
            r"(?:ok(?:ay)?|confirm(?:\s+(?:order|it|please|lah))?|yes|ya|yep|yeah|boleh|bayar|ngam|betul|sure|proceed|place\s+order|go\s+ahead)[\s!.,?]*$",
            t_low
        ) or
        re.match(r"^(?:(?:好|好的|可以|没问题)[\s,，]*)*(?:好|好的|确认|确认下单|可以|没问题|下单)[\s！。!.,?]*$", utterance)
    )
    if was_awaiting_confirm and is_strict_confirm and not has_modify_cue and not extracted_slots:
        state["status"] = "DISPENSING"
        response = (
            "订单已确认！正在为您现磨冲泡，请稍候从出料口取杯，谢谢老板！"
            if is_zh else
            "Order confirmed! Mesin sedang bancuh kopi anda. Sila ambil cawan di dispenser, terima kasih boss!"
        )
        return {
            "response": response, "intent": "order_confirm", "model_name": model_name_str,
            "confidence": 1.0, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # 3.5 Price Inquiry (prevent hijacking to health_query when asking "how much / berapa")
    is_price_inquiry = bool(
        re.search(r"\b(?:how\s+much|price|pricing|cost|harga|berapa(?:\s+ringgit)?)\b", t_low) or
        any(w in utterance for w in ["多少钱", "几块", "几钱", "价钱", "价格"])
    )
    if is_price_inquiry and not re.search(r"\b(?:calorie|calories|kalori|sugar|gula|卡路里|热量|lactose|gluten|allergen|乳糖|麸质|过敏|高不高|胖不胖)\b", t_low):
        if extracted_slots.get("base_drink"):
            temp_state = dict(extracted_slots)
            if not temp_state.get("milk_type"):
                temp_state["milk_type"] = "None" if temp_state["base_drink"] in ("Teh O Limau", "Americano") else "Condensed Milk"
            kb_info = lookup_price_and_nutrition(temp_state)
            milk_tag = " O" if temp_state.get("milk_type") == "None" and temp_state["base_drink"] in ("Kopi", "Teh", "Cham", "Milo") else (" C" if temp_state.get("milk_type") == "Evaporated Milk" else "")
            temp_tag = " Peng (Iced)" if temp_state.get("temperature") == "Iced" else (" Hot" if temp_state.get("temperature") == "Hot" else "")
            label = f"{temp_state['base_drink']}{milk_tag}{temp_tag}".strip()
            response = (
                f"{label} 的价格是 RM {kb_info['price_rm']:.2f}（约 {kb_info['calories']} kcal，{kb_info['sugars_g']:g}g 糖）。需要现在下单吗？"
                if is_zh else
                f"Harga untuk {label} ialah RM {kb_info['price_rm']:.2f} (approx. {kb_info['calories']} kcal, {kb_info['sugars_g']:g}g sugar). Want to order this boss?"
            )
        else:
            response = (
                "ChatCoffee 传统饮品价格介于 RM 3.20 – RM 5.80（如 Kopi O RM 3.20、Kopi C Peng RM 4.70、Cham Peng RM 5.00）。请问您想查询哪一杯？"
                if is_zh else
                "Harga minuman ChatCoffee bermula dari RM 3.20 hingga RM 5.80 (cth: Kopi O RM 3.20, Kopi C Peng RM 4.70, Cham Peng RM 5.00). Nak check minuman mana boss?"
            )
        return {
            "response": response, "intent": "price_query", "model_intent": pred_intent,
            "tone": detected_tone, "confidence": conf, "probabilities": prob_dict,
            "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # 4. Health, Calorie & Allergen RAG query grounded on unified_coffee_kb.csv (Direct lookup with 4-tier calorie evaluation and allergen check, no cart mutation)
    is_allergen_inquiry = bool(
        re.search(r"\b(?:lactose|laktosa|gluten|allergen|allergens|allergi|alergi|alahan|alergik|allergic|allergy|intoleran|intolerant|intolerance|intoleransi|dairy|celiac)\b", t_low) or
        any(w in utterance for w in ["乳糖", "麸质", "过敏", "乳糜泻", "过敏原", "奶制品"])
    )
    is_subjective_calorie = bool(
        re.search(r"\b(?:high\s*calori\w*|is\s+it\s+high|fattening|make\s+me\s+fat|gain\s+weight|tinggi\s+kalori|bikin\s+gemuk|gemuk|kalori\s+tinggi|banyak\s+kalori)\b", t_low) or
        any(w in utterance for w in ["高不高", "胖不胖", "会胖吗", "容易胖吗", "长胖", "发胖", "长肉", "高吗", "会发胖", "会长胖"])
    )
    is_calorie_inquiry = bool(
        is_subjective_calorie or
        re.search(r"\b(?:calorie|calories|kalori|khasiat|nutrisi|kcal|low\s*calori\w*)\b", t_low) or
        any(w in utterance for w in ["卡路里", "热量", "多少卡", "几卡", "含糖量", "低卡", "低热量"])
    )
    is_general_health = bool(
        re.search(r"\b(?:healthy|health|diet|sihat|rendah\s+kalori)\b", t_low) or
        any(w in utterance for w in ["健康", "减脂", "减肥", "控糖", "养生"])
    )
    has_health_question = bool(
        re.search(r"\b(?:what|apa|which|mana|ada|rekomen|recommend|suggest|cadang)\b", t_low) or
        "?" in utterance or "？" in utterance
    )

    is_health_rag = (
        is_allergen_inquiry or
        is_subjective_calorie or
        is_general_health or
        (is_calorie_inquiry and not was_awaiting_confirm) or
        (pred_intent == "health_query" and (has_health_question or is_calorie_inquiry or is_allergen_inquiry or is_general_health))
    )

    if is_health_rag:
        user_lang = detect_lang(utterance)
        is_coref = bool(
            re.search(r"(?:这杯|这个|这款|当前这杯|刚点的|选的这杯|那杯|this\s+cup|this\s+drink|this\s+one|current\s+drink|my\s+drink|air\s+ni|cawan\s+ni|minuman\s+ni|air\s+ini|cawan\s+ini|minuman\s+ini|air\s+tu|cawan\s+tu)", t_low)
        )
        is_recommendation_query = bool(
            re.search(r"\b(?:what|apa|which|mana|ada|rekomen|recommend|suggest|suggestion|cadang|cadangan|saran|sarankan|pilihan|senarai|list|tunjuk|options?|choices?|any|drinks?)\b", t_low) or
            re.search(r"\b(?:nak|boleh|apa)\s+minum\s+apa\b|\bminum\s+apa\b|\bapa\s+boleh\s+minum\b|\bwhat\s+(?:can|should|to)\s+drink\b", t_low) or
            any(w in utterance for w in ["推荐", "介绍", "建议", "有什么", "有哪些", "能喝什么", "可以喝什么", "喝什么", "什么好喝", "哪些可以喝", "有什么喝", "选择"])
        )

        target_drink = None
        for b_name in ["Milo Dinosaur", "Milo Godzilla", "Teh O Ais Limau", "Teh O Limau", "Kopi O Kosong", "Teh O Kosong", "White Coffee", "Milo", "Neslo", "Cham", "Kopi C", "Kopi O", "Kopi", "Teh C", "Teh O", "Teh Tarik", "Teh", "Americano"]:
            if b_name.lower() in t_low:
                target_drink = b_name
                break

        if not target_drink:
            target_drink = extracted_slots.get("base_drink")
            if target_drink == "Milo" and "dinosaur" in t_low:
                target_drink = "Milo Dinosaur"
            elif target_drink == "Teh" and "tarik" in t_low:
                target_drink = "Teh Tarik"

        target_milk = extracted_slots.get("milk_type")
        target_temp = extracted_slots.get("temperature")
        target_sweet = extracted_slots.get("sweetness_level")

        if not target_drink and (is_coref or (state.get("base_drink") and not is_recommendation_query)):
            if state.get("base_drink"):
                target_drink = state["base_drink"]
                if not target_milk:
                    target_milk = state.get("milk_type")
                if not target_temp:
                    target_temp = state.get("temperature")
                if not target_sweet:
                    target_sweet = state.get("sweetness_level")

        # Allergen inquiry
        if is_allergen_inquiry:
            if not target_drink:
                if is_coref:
                    response = get_no_drink_prompt("allergen", user_lang)
                else:
                    if re.search(r"lactose|laktosa|dairy|susu", t_low) or any(w in utterance for w in ["乳糖", "无奶"]):
                        response = get_allergen_recommendation("lactose", user_lang)
                    elif re.search(r"gluten|barli", t_low) or "麸质" in utterance:
                        response = get_allergen_recommendation("gluten", user_lang)
                    else:
                        response = get_allergen_recommendation("all", user_lang)
                return {
                    "response": response, "intent": "health_query", "model_intent": pred_intent,
                    "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
                }

            temp_state = {"base_drink": target_drink, "milk_type": target_milk, "temperature": target_temp, "sweetness_level": target_sweet}
            kb_info = lookup_price_and_nutrition(temp_state)
            inquiry_type = "general"
            if re.search(r"lactose|laktosa|dairy|susu", t_low) or any(w in utterance for w in ["乳糖", "牛奶"]):
                inquiry_type = "lactose"
            elif re.search(r"gluten|barli", t_low) or "麸质" in utterance:
                inquiry_type = "gluten"

            dname = kb_info.get("display_name") or target_drink
            response = get_drink_allergen_reply(
                dname,
                kb_info["contains_lactose"],
                kb_info["contains_gluten"],
                inquiry_type,
                user_lang
            )
            return {
                "response": response, "intent": "health_query", "model_intent": pred_intent,
                "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
            }

        # Calorie inquiry
        if is_calorie_inquiry:
            if not target_drink:
                if is_coref or is_subjective_calorie:
                    response = get_no_drink_prompt("calorie", user_lang)
                else:
                    response = query_healthy_drinks_from_kb(user_lang)
                return {
                    "response": response, "intent": "health_query", "model_intent": pred_intent,
                    "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
                }

            temp_state = {"base_drink": target_drink, "milk_type": target_milk, "temperature": target_temp, "sweetness_level": target_sweet}
            kb_info = lookup_price_and_nutrition(temp_state)
            cal = kb_info["calories"]
            sug = kb_info["sugars_g"]
            dname = kb_info.get("display_name") or target_drink
            response = classify_calorie_tier(cal, dname, sug, user_lang)
            return {
                "response": response, "intent": "health_query", "model_intent": pred_intent,
                "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
            }

        # General health query
        response = query_healthy_drinks_from_kb(user_lang)
        return {
            "response": response, "intent": "health_query", "model_intent": pred_intent,
            "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # Low-confidence out-of-domain rejection
    if conf < 0.25 and not extracted_slots and not was_awaiting_confirm and not state.get("pending_slot"):
        response = (
            "抱歉，我不太明白您的意思。请问您想喝什么饮料呢？可选咖啡（Kopi）、奶茶（Teh）、美禄（Milo）等。"
            if is_zh else
            "Maaf boss, saya tak berapa faham. Boleh ulang semula? Kita ada Kopi, Teh, Cham, Milo, White Coffee."
        )
        return {
            "response": response, "intent": "fallback", "model_intent": pred_intent,
            "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": {}, "state": state
        }

    # 5. Merge ALL extracted slots (preserving multi-slot replies during pending_slot clarification)
    is_drink_changed = bool(extracted_slots.get("base_drink") and extracted_slots["base_drink"] != state.get("base_drink"))
    is_explicit_base_mention = bool(extracted_slots.get("base_drink") and not extracted_slots.get("milk_type"))

    if state.get("pending_slot") and state["pending_slot"] in extracted_slots:
        state["pending_slot"] = None
    for k, v in extracted_slots.items():
        state[k] = v

    if is_drink_changed or is_explicit_base_mention:
        if not extracted_slots.get("milk_type"):
            state["milk_type"] = "None" if state.get("base_drink") in ("Teh O Limau", "Americano") else "Condensed Milk"
        if not extracted_slots.get("sweetness_level"):
            state["sweetness_level"] = "Zero Sugar" if state.get("base_drink") == "Americano" else "Normal"

    # Enforce milk & condensed-milk zero-sugar constraints across ALL intents
    if state.get("base_drink") in ("Teh O Limau", "Americano"):
        state["milk_type"] = "None"
    if state.get("base_drink") == "Americano" and not state.get("sweetness_level"):
        state["sweetness_level"] = "Zero Sugar"
    condensed_drinks = ("Cham", "Kopi", "Teh", "White Coffee", "Milo", "Neslo")
    is_condensed = state.get("milk_type") == "Condensed Milk" or (
        state.get("base_drink") in condensed_drinks and state.get("milk_type") not in ("None", "Evaporated Milk")
    )
    if state.get("sweetness_level") == "Zero Sugar" and is_condensed:
        if state.get("base_drink") in ("Kopi", "Teh", "Milo") and state.get("milk_type") is None:
            state["milk_type"] = "None"
        else:
            state["sweetness_level"] = "Less Sugar"

    # 6. Check missing mandatory slots
    if not state.get("base_drink"):
        response = "请问您想喝什么饮料呢？我们有 Kopi、Teh、Cham、Milo、White Coffee 等。" if is_zh else "Nak minum apa boss? Kita ada Kopi, Teh, Cham, Milo, White Coffee dan lain-lain."
        state["pending_slot"] = "base_drink"
        state["status"] = "GATHERING_SLOTS"
    elif not state.get("temperature"):
        drink = state["base_drink"]
        response = f"您选了 {drink}。请问要热的（Hot）还是加冰（Iced / Peng）呢？" if is_zh else f"Bancuh {drink}. Nak panas atau sejuk / ais (Peng) boss?"
        state["pending_slot"] = "temperature"
        state["status"] = "GATHERING_SLOTS"
    elif not state.get("sweetness_level"):
        response = "甜度想要少甜（Siu Dai / Less Sugar）、无糖（Kosong）还是正常甜？" if is_zh else "Manis macam mana boss? Kurang manis (Siu Dai), Kosong (Zero Sugar), atau Normal?"
        state["pending_slot"] = "sweetness_level"
        state["status"] = "GATHERING_SLOTS"
    else:
        if not state.get("milk_type"):
            state["milk_type"] = "None" if state["base_drink"] in ("Teh O Limau", "Americano") else "Condensed Milk"
        state["status"] = "AWAITING_CONFIRMATION"
        state["pending_slot"] = None
        kb_info = lookup_price_and_nutrition(state)
        state["price_rm"] = kb_info["price_rm"]
        drink_name = f"{state['base_drink']} {state.get('temperature', '')}".strip()
        sweet = state.get("sweetness_level", "Normal")
        milk = state.get("milk_type", "Condensed Milk")
        if detected_tone == "Urgent":
            response = (
                f"⚡ 极速模式：{drink_name}（{sweet}，{milk}），RM {kb_info['price_rm']:.2f}。说「确认」立刻出杯！"
                if is_zh else
                f"⚡ Express mode boss! {drink_name} ({sweet}, {milk}), RM {kb_info['price_rm']:.2f}. Say Confirm to brew immediately!"
            )
        else:
            response = (
                f"好的！一杯 {drink_name}（{sweet}，{milk}）。总共 RM {kb_info['price_rm']:.2f}。请问确认下单吗？（请回答 OK / Confirm）"
                if is_zh else
                f"Beres boss! Satu {drink_name} ({sweet}, {milk}). Total RM {kb_info['price_rm']:.2f}. Nak confirm order sekarang? (Jawab Confirm / Boleh)"
            )

    return {
        "response": response,
        "intent": pred_intent,
        "model_name": model_name_str,
        "tone": detected_tone,
        "confidence": conf,
        "probabilities": prob_dict,
        "clean_text": c_text,
        "slots": extracted_slots,
        "state": state
    }


# -------------------------------------------------------------
# 5. Serve Frontend Kiosk UI (Single source of truth: index.html)
# -------------------------------------------------------------
HTML_FILE = os.path.join(BASE_DIR, "index.html")


@app.get("/", response_class=HTMLResponse)
def serve_kiosk():
    if os.path.exists(HTML_FILE):
        with open(HTML_FILE, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>ChatCoffee Kiosk index.html not found</h1>")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting ChatCoffee ML Kiosk on http://localhost:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
