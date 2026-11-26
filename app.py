"""
app.py - ChatCoffee Unmanned Smart Kiosk Backend Server
Powered by Supervised ML Intent Classification + Unified Coffee Knowledge Base
"""

import os
import re
import json
import math
import asyncio
import threading
from collections import Counter
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Response, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

_SLANG_LOCK = threading.Lock()
_HTTP_SESSION = requests.Session()
_RESERVED_SLANG_KEYS = frozenset({
    "kau", "kopi", "teh", "cham", "milo", "neslo", "americano",
    "kosong", "siew", "siu", "dai", "peng", "panas", "suam", "tarik", "limau"
})
_MAX_LEARNED_SLANG = 500

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

V15_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "coffee-rojak.csv")
GOLD_PATH = V15_PATH if os.path.exists(V15_PATH) else os.path.join(BASE_DIR, "NLP_Datasets", "chatcoffee_gold_850.csv")
if not os.path.exists(GOLD_PATH):
    GOLD_PATH = os.path.join(BASE_DIR, "chatcoffee_gold_850.csv")

UI_ACTION_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "order_drink": ("prepare_selection", "detail"),
    "modify_order": ("update_selection", "detail"),
    "clarify_order": ("answer_or_clarify", None),
    "health_query": ("lookup_nutrition", None),
    "price_query": ("lookup_price", None),
    "order_confirm": ("review_and_confirm", "payment"),
    "cancel_order": ("request_cancel", "home"),
    "system_reset": ("request_cancel", "home"),
    "single_cup_limit": ("enforce_single_cup", None),
    "payment_query": ("answer_payment_policy", "payment"),
    "navigate_menu": ("show_menu", "menu"),
    "navigate_payment": ("show_payment", "payment"),
    "review_order": ("show_order", "detail"),
    "navigate_home": ("show_home", "home"),
    "out_of_scope": ("safe_fallback", None),
    "fallback": ("safe_fallback", None),
}

_PROTECTED_SLANG_KEYS = {"kau", "on", "confirm"}

# Load & cleanly flatten slang dictionary (exclude metadata, intensifiers, and protected keywords)
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
                        if isinstance(sv, str) and sk.lower() not in _PROTECTED_SLANG_KEYS:
                            slang_dict[sk.lower()] = sv.lower()
                elif isinstance(v, str) and k.lower() not in _PROTECTED_SLANG_KEYS:
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


_MULTIWORD_SLANG_PATTERNS: List[Tuple[re.Pattern, str]] = []


def _rebuild_multiword_slang_cache() -> None:
    global _MULTIWORD_SLANG_PATTERNS
    combined_multi = {k: v for k, v in slang_dict.items() if " " in k}
    for k, v in learned_slang_dict.items():
        if " " in k:
            combined_multi[k] = v
    _MULTIWORD_SLANG_PATTERNS = [
        (re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"), combined_multi[phrase])
        for phrase in sorted(combined_multi.keys(), key=len, reverse=True)
    ]


_rebuild_multiword_slang_cache()


def save_learned_slang(slang: str, meaning: str) -> Dict[str, str]:
    k = re.sub(r"\s+", " ", slang.strip().lower())
    v = re.sub(r"\s+", " ", meaning.strip().lower())
    if not k or not v or len(k) > 60 or len(v) > 80:
        raise ValueError("Slang and meaning must be non-empty and under 60/80 characters.")
    if k in _RESERVED_SLANG_KEYS:
        raise ValueError(f"Reserved menu keyword '{k}' cannot be overwritten.")
    with _SLANG_LOCK:
        if k not in learned_slang_dict and len(learned_slang_dict) >= _MAX_LEARNED_SLANG:
            raise ValueError("Learned slang capacity limit reached.")
        learned_slang_dict[k] = v
        slang_dict[k] = v
        _rebuild_multiword_slang_cache()
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
kb_by_name_lower: Dict[str, Dict[str, Any]] = {}
if os.path.exists(KB_PATH):
    df_kb = pd.read_csv(KB_PATH, keep_default_na=False)
    df_kb["milk_type"] = df_kb["milk_type"].astype(str).str.strip().replace("", "None")
    for _, _kb_row in df_kb.iterrows():
        _dname_l = str(_kb_row["drink_name"]).strip().lower()
        if _dname_l and _dname_l not in kb_by_name_lower:
            kb_by_name_lower[_dname_l] = _kb_row.to_dict()
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
            if isinstance(sv, str) and sk.lower() not in _PROTECTED_SLANG_KEYS:
                slang_dict.setdefault(sk.lower(), sv.lower())
        _rebuild_multiword_slang_cache()
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
def split_kopitiam_compounds(text: str) -> str:
    """Split concatenated kopitiam drink/modifier tokens (e.g. champeng -> cham peng, tehc -> teh c)."""
    if not text:
        return text
    t = re.sub(r"\b(teh)(o)?(limau)(peng|ais|ice|panas)?\b", lambda m: " ".join(p for p in m.groups() if p), text)
    t = re.sub(r"\b(kopi|teh|cham|milo|neslo)(o|c)(peng|ping|ais|ice|iced|panas|suam|hot|kosong)?\b", lambda m: " ".join(p for p in m.groups() if p), t)
    t = re.sub(r"\b(kopi|teh|cham|milo|neslo)(peng|ping|ais|ice|iced|panas|suam|hot|tarik|kosong)\b", r"\1 \2", t)
    t = re.sub(r"\b(siu|siew|gah)(dai)\b", r"\1 \2", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    # Separate CJK and Latin/digit boundaries so mixed tokens like "少糖pls" or "nk1杯" expand cleanly
    t = re.sub(r"([\u4e00-\u9fff])([a-z0-9])", r"\1 \2", t)
    t = re.sub(r"([a-z0-9])([\u4e00-\u9fff])", r"\1 \2", t)
    # Strip punctuation BEFORE phrase/token expansion so e.g. "brp?" -> "brp" -> "berapa"
    t = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", t)
    # Split concatenated kopitiam compounds (e.g. champeng, kopipeng, tehc, kopio)
    t = split_kopitiam_compounds(t)
    # Apply cached multi-word slang phrases first (longest match)
    for pat, repl in _MULTIWORD_SLANG_PATTERNS:
        t = pat.sub(repl, t)
    words = t.split()
    expanded = [slang_dict.get(w, w) for w in words]
    t = " ".join(expanded)
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

    spaced_text = re.sub(r"([\u4e00-\u9fff])", r" \1 ", c_text)
    X_vec = tfidf_vec.transform([spaced_text])
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
    t_low = split_kopitiam_compounds(text.lower())
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
    elif (
        re.search(r"美式(?:咖啡)?(?=$|[\s，。！？,.!?]|(?:冰|热|温|少|无|正常|打包|大包))|(?:冰|热|温|冻|冷|来|要|点|杯)美式", text) or
        re.search(r"\bamericano\b", t_low)
    ):
        slots["base_drink"] = "Americano"
        slots["milk_type"] = "None"
    elif "咖啡" in text or re.search(r"\b(?:kopi|coffee)\b", t_low):
        slots["base_drink"] = "Kopi"
    elif "奶茶" in text or "茶" in text or re.search(r"\b(?:teh|tea)\b", t_low):
        slots["base_drink"] = "Teh"

    # 2. Temperature & Ice/Warmth Modifiers
    has_hot_negation = bool(
        re.search(r"\b(?:(?:tak|tidak|x)\s*(?:mau|mahu|nak)|xnak|xmau|bukan|jangan|don't\s+want|do\s+not\s+want|not|no)\s+(?:panas|hot)\b", t_low) or
        any(w in text for w in ["不要热", "不要热的", "不是热", "别热", "不热"])
    )
    has_ice_negation = bool(
        re.search(r"\b(?:bukan\s+(?:ais|ice)|don't\s+want\s+(?:ice|iced)|do\s+not\s+want\s+(?:ice|iced)|tak\s+(?:mau|mahu)\s+ais)\b", t_low) or
        any(w in text for w in ["不要冰", "不要加冰", "不是冰"])
    )
    has_warm_zh = bool(
        re.search(r"温的|温饮|水温|温一点|微温|喝温", text) or
        ("温" in text and not any(w in text for w in ["温度", "室温", "气温", "保温"]))
    )
    if has_hot_negation:
        slots["temperature"] = "Iced"
    elif has_warm_zh or re.search(r"\b(?:suam|warm)\b", t_low):
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
        re.search(r"\bikat\s+tepi\b", t_low) or
        (re.search(r"\b(?:peng|ping|ice|iced|ais|sejuk|dingin|cold)\b", t_low) and not re.search(r"\b(?:price|pricing)\b", t_low))
    ):
        slots["temperature"] = "Iced"

    if "去冰" in text or "走冰" in text or "无冰" in text or re.search(r"\b(?:kosong (?:ais|ice)|tanpa (?:ais|ice)|no (?:ais|ice)|(?:tak|tidak|x)\s*(?:nak|mau)\s*(?:ais|ice)|xnak\s*(?:ais|ice))\b", t_low):
        slots["temperature"] = "Iced"
        slots["ice_level"] = "No Ice"
    elif "少冰" in text or "微冰" in text or re.search(r"\b(?:kurang (?:ais|ice)|less (?:ais|ice)|(?:sikit|sedikit) (?:ais|ice)|tak nak banyak (?:ais|ice))\b", t_low):
        slots["temperature"] = "Iced"
        slots["ice_level"] = "Less Ice"

    # 3. Sweetness (tightened to avoid 'banyak ais', 'RM 4.0', or 'normal milk' false positives)
    if (
        "无糖" in text or "無糖" in text or "走糖" in text or
        re.search(r"\b(?:kosong|ksg|zero\s+sugar|zero\s+gula|no\s+sugar|no\s+gula|tanpa\s+gula|tak\s+(?:nak|mau)\s+gula|tidak\s+(?:nak|mau)\s+gula|x\s*(?:nak|mau)\s*gula|xnak\s+gula|zero)\b|(?<!\d[.\s])\b0\b(?!\.\d)|\b0\s*(?:%|sugar|gula)\b", t_low)
    ):
        slots["sweetness_level"] = "Zero Sugar"
    elif (
        "少糖" in text or "微糖" in text or "少甜" in text or "半糖" in text or
        re.search(r"\b(?:siu dai|siew dai|sd|less sugar|kurang manis|kurang gula|kurang sweet|krg\s+mns|sedikit manis|sedikit gula|sikit manis|sikit gula|half sugar|less sweet)\b|\b(?:50|30|25)\s*%", t_low)
    ):
        slots["sweetness_level"] = "Less Sugar"
    elif (
        "多糖" in text or "加糖" in text or "多甜" in text or
        re.search(r"\b(?:gah dai|ga dai|gd|extra sweet|extra manis|lebih manis|lebih gula|tambah gula|banyak manis|banyak gula|extra sugar)\b|\b120\s*%", t_low)
    ):
        slots["sweetness_level"] = "Extra Sugar"
    elif (
        "正常糖" in text or "普通糖" in text or "标准糖" in text or
        ("正常" in text and "正常奶" not in text) or
        re.search(r"\b(?:normal\s+sugar|biasa\s+manis|manis\s+biasa|std\s+sugar)\b|\b(?:normal|biasa)\b(?!\s*(?:milk|susu|ice|ais))|\b100\s*%", t_low)
    ):
        slots["sweetness_level"] = "Normal"
    elif (
        re.search(r"\b(?:kopi|coffee|teh|tea|cham|milo|neslo)(?:\s+(?:panas|hot|ais|ice|iced|peng|suam))?\s*(?:o|c)\b|\b(?:teh\s*tarik|white\s*coffee|neslo)\b", t_low) or
        any(w in text for w in ["咖啡乌", "茶乌", "美禄乌", "咖啡c", "茶c", "咖啡昔", "茶昔", "拉茶", "白咖啡", "雀巢美禄"])
    ):
        slots["sweetness_level"] = "Normal"

    # 4. Milk Type (Teh O Limau & Americano locked to None; 炼奶 correctly mapped to Condensed Milk)
    if slots.get("base_drink") in ("Teh O Limau", "Americano"):
        slots["milk_type"] = "None"
    elif (
        "淡奶" in text or "花奶" in text or
        re.search(r"\b(?:kopi|coffee|teh|tea|cham|neslo)(?:\s+(?:panas|hot|ais|ice|iced|peng|suam))?\s*c\b|\b(?:evaporated|susu sejat|susu cair)\b", t_low) or
        re.match(r"^\s*(?:c|si)\s*$", t_low)
    ):
        slots["milk_type"] = "Evaporated Milk"
    elif (
        "乌" in text or "黑咖啡" in text or "不加奶" in text or "不要加奶" in text or "不要奶" in text or "不用奶" in text or "不放奶" in text or "无奶" in text or
        re.search(r"\b(?:kopi|coffee|teh|tea|cham|milo|neslo)(?:\s+(?:panas|hot|ais|ice|iced|peng|suam))?\s*o\b|\b(?:kosong milk|no milk|tanpa susu|black|tak perlu susu|tak payah susu|tak nak susu|tak mau susu|tidak nak susu|tidak mau susu|x\s*nak susu|xnak susu)\b", t_low) or
        re.match(r"^\s*o\s*$", t_low) or
        (re.search(r"\b(?:kosong|ksg)\b", t_low) and not re.search(r"\b(?:susu|condensed|evaporated|c|tarik|biasa)\b", t_low))
    ):
        slots["milk_type"] = "None"
    elif (
        "炼奶" in text or "炼乳" in text or "普通奶" in text or "正常奶" in text or "拉茶" in text or
        re.search(r"\b(?:condensed|susu pekat|tarik|susu biasa|normal milk)\b", t_low)
    ):
        slots["milk_type"] = "Condensed Milk"

    return slots


def _fallback_allergens(base: str, milk: Optional[str]) -> Tuple[bool, bool, str]:
    contains_lactose = False
    contains_gluten = False
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
    return contains_lactose, contains_gluten, ", ".join(parts) if parts else "None (Allergen-Free)"


def _normalize_cand(
    base: str, temp: Optional[str], milk: Optional[str], sweet: Optional[str]
) -> List[str]:
    is_iced = (temp == "Iced")
    cands: List[str] = []
    if base in ("Kopi", "Teh", "Cham", "Milo"):
        if milk == "None":
            if is_iced:
                if sweet == "Zero Sugar":
                    cands.extend([f"{base} O Kosong Peng", f"{base} O Peng Kosong"])
                elif sweet == "Less Sugar":
                    cands.extend([f"{base} O Peng Siew Dai", f"{base} O Siew Dai"])
                elif sweet == "Extra Sugar":
                    cands.extend([f"{base} O Peng Ga Dai", f"{base} O Ga Dai"])
                cands.append(f"{base} O Peng")
            else:
                if sweet == "Zero Sugar":
                    cands.append(f"{base} O Kosong")
                elif sweet == "Less Sugar":
                    cands.append(f"{base} O Siew Dai")
                elif sweet == "Extra Sugar":
                    cands.extend([f"{base} O Ga Dai", f"{base} O Gah Dai"])
                cands.append(f"{base} O")
        elif milk == "Evaporated Milk":
            if is_iced:
                if sweet == "Zero Sugar":
                    cands.extend([f"{base} C Peng Kosong", f"{base} C Kosong"])
                elif sweet == "Less Sugar":
                    cands.extend([f"{base} C Peng Siew Dai", f"{base} C Siew Dai"])
                cands.append(f"{base} C Peng")
            else:
                if sweet == "Zero Sugar":
                    cands.append(f"{base} C Kosong")
                elif sweet == "Less Sugar":
                    cands.append(f"{base} C Siew Dai")
                elif sweet == "Extra Sugar":
                    cands.extend([f"{base} C Ga Dai", f"{base} C Gah Dai"])
                cands.append(f"{base} C")
        else:
            if is_iced:
                if sweet == "Less Sugar":
                    cands.append(f"{base} Peng Siew Dai")
                elif sweet == "Extra Sugar":
                    cands.extend([f"{base} Peng Ga Dai", f"{base} Peng Gah Dai"])
                cands.append(f"{base} Peng")
            else:
                if sweet == "Zero Sugar":
                    cands.append(f"{base} Kosong")
                elif sweet == "Less Sugar":
                    cands.append(f"{base} Siew Dai")
                elif sweet == "Extra Sugar":
                    cands.extend([f"{base} Ga Dai", f"{base} Gah Dai"])
    elif base == "Teh Tarik":
        if is_iced:
            if sweet == "Less Sugar":
                cands.append("Teh Tarik Peng Siew Dai")
            cands.append("Teh Tarik Peng")
        else:
            if sweet == "Less Sugar":
                cands.append("Teh Tarik Siew Dai")
            cands.append("Teh Tarik")
    elif base == "Neslo":
        if milk == "None" or sweet == "Zero Sugar":
            cands.append("Neslo Peng Kosong" if is_iced else "Neslo Kosong")
        if is_iced:
            if sweet == "Less Sugar":
                cands.append("Neslo Peng Siew Dai")
            cands.append("Neslo Peng")
        else:
            if sweet == "Less Sugar":
                cands.append("Neslo Siew Dai")
            cands.append("Neslo")
    elif base == "White Coffee":
        if is_iced:
            if sweet == "Zero Sugar":
                cands.append("White Coffee Peng Kosong")
            elif sweet == "Less Sugar":
                cands.append("White Coffee Peng Siew Dai")
            cands.append("White Coffee Peng")
        else:
            if sweet == "Less Sugar":
                cands.append("White Coffee Siew Dai")
            elif sweet == "Zero Sugar":
                cands.append("White Coffee Kosong")
            cands.append("White Coffee")
    elif base == "Teh O Limau":
        if is_iced:
            if sweet == "Less Sugar":
                cands.extend(["Teh O Ais Limau Siu Dai", "Teh O Ais Limau Less Sugar"])
            elif sweet == "Zero Sugar":
                cands.append("Teh O Ais Limau Kosong")
            cands.append("Teh O Ais Limau")
        else:
            if sweet == "Less Sugar":
                cands.append("Teh O Limau Panas Siu Dai")
            elif sweet == "Zero Sugar":
                cands.append("Teh O Limau Panas Kosong")
            cands.append("Teh O Limau Panas")
    elif base == "Americano":
        cands.append("Iced Americano" if is_iced else "Hot Americano")
    cands.append(base)
    return cands


def lookup_price_and_nutrition(state: Dict[str, Any]) -> Dict[str, Any]:
    # Zero-milk base prices: Cham/Neslo=3.70 (+0.50 condensed / +0.70 evap), Kopi/Teh/Milo/Limau=3.20
    base_prices_no_milk = {
        "Cham": 3.70, "White Coffee": 4.30, "Neslo": 3.70, "Milo": 3.20,
        "Kopi": 3.20, "Teh": 3.20, "Teh O Limau": 3.20, "Americano": 5.00
    }
    base = state.get("base_drink") or "Kopi"
    temp = state.get("temperature")
    milk = state.get("milk_type")
    sweet = state.get("sweetness_level")

    price = base_prices_no_milk.get(base, 3.20)
    if base not in ("Teh O Limau", "Americano", "White Coffee"):
        if milk == "Condensed Milk":
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

    contains_lactose, contains_gluten, allergens_summary = _fallback_allergens(base, milk)
    default_caffeine = {
        "Cham": 110.0, "Kopi": 100.0, "Teh": 50.0, "Milo": 25.0,
        "Neslo": 70.0, "White Coffee": 130.0, "Teh O Limau": 50.0,
        "Americano": 150.0 if temp == "Iced" else 75.0
    }
    caffeine_mg = float(state["caffeine_mg"]) if state.get("caffeine_mg") is not None else default_caffeine.get(base, 80.0)
    resolved_milk = milk or ("None" if base in ("Teh O Limau", "Americano") else "Condensed Milk")
    display_name = base
    candidate_names = _normalize_cand(base, temp, milk, sweet)

    matched_row: Optional[Dict[str, Any]] = None
    if kb_by_name_lower:
        for cname in candidate_names:
            row = kb_by_name_lower.get(cname.lower())
            if row is not None:
                matched_row = row
                break

        if matched_row is None and not df_kb.empty:
            sub = df_kb[df_kb["drink_name"].str.contains(re.escape(base), case=False, na=False)]
            if temp and not sub.empty:
                t_sub = sub[sub["temperature"].str.lower() == temp.lower()]
                if not t_sub.empty:
                    sub = t_sub
            if milk and not sub.empty:
                m_sub = sub[sub["milk_type"].astype(str).str.lower() == milk.lower()]
                sub = m_sub
            if sweet and not sub.empty:
                s_sub = sub[sub["sweetness_level"].str.lower() == sweet.lower()]
                if not s_sub.empty:
                    sub = s_sub
            elif not sweet and not sub.empty:
                s_sub = sub[sub["sweetness_level"].str.lower() == "normal"]
                if not s_sub.empty:
                    sub = s_sub
            if not sub.empty:
                matched_row = sub.iloc[0].to_dict()

    if matched_row is not None:
        display_name = str(matched_row["drink_name"])
        if matched_row.get("price_rm") not in ("", None):
            price = float(matched_row["price_rm"])
        if matched_row.get("calories") not in ("", None):
            calories = int(float(matched_row["calories"]))
        if matched_row.get("sugars_g") not in ("", None):
            sugars_g = float(matched_row["sugars_g"])
        if "contains_lactose" in matched_row:
            contains_lactose = str(matched_row["contains_lactose"]).strip().lower() == "true"
        if "contains_gluten" in matched_row:
            contains_gluten = str(matched_row["contains_gluten"]).strip().lower() == "true"
        if matched_row.get("allergens_summary"):
            allergens_summary = str(matched_row["allergens_summary"])
        if matched_row.get("caffeine_mg") not in ("", None):
            caffeine_mg = float(matched_row["caffeine_mg"])
        if matched_row.get("milk_type"):
            resolved_milk = str(matched_row["milk_type"]).strip()

    return {
        "display_name": display_name,
        "price_rm": round(price, 2),
        "calories": calories,
        "sugars_g": sugars_g,
        "caffeine_mg": caffeine_mg,
        "milk_type": resolved_milk,
        "contains_lactose": contains_lactose,
        "contains_gluten": contains_gluten,
        "allergens_summary": allergens_summary
    }


_BM_WORDS = frozenset({
    "nak", "mau", "satu", "dua", "tiga", "panas", "sejuk", "ais", "kurang", "manis",
    "tak", "xnak", "taknak", "tanpa", "gula", "susu", "cair", "pekat", "batal", "semula",
    "bayar", "boleh", "sedap", "minum", "cawan", "tambah", "bancuh", "tolong", "bagi",
    "saya", "sihat", "berapa", "apa", "ini", "tu", "ni", "ke", "lah", "ia", "ada", "air",
    "laktosa", "alahan", "alergen", "intoleransi", "tinggi", "rendah", "gemuk", "bikin",
    "pilihan", "senarai", "cadang", "cadangan", "saran", "sarankan", "sedikit", "minuman",
    "tukar", "mana", "ganti", "ubah", "mahu", "suam", "ikat", "tepi"
})
_EN_WORDS = frozenset({
    "the", "is", "this", "cup", "drink", "drinks", "high", "calorie", "calories", "fattening", "make",
    "gain", "weight", "what", "which", "how", "much", "does", "have", "contain", "contains",
    "lactose", "gluten", "allergy", "allergic", "allergen", "allergens", "intolerant",
    "intolerance", "safe", "recommend", "recommendation", "suggest", "healthy", "diet", "free",
    "any", "options", "choices"
})


def detect_lang(text: str) -> str:
    if not text:
        return "en"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    bm_score = len(tokens & _BM_WORDS)
    en_score = len(tokens & _EN_WORDS)
    if bm_score > en_score and bm_score > 0:
        return "bm"
    if en_score > bm_score and en_score > 0:
        return "en"
    if bm_score > 0:
        return "bm"
    if en_score > 0:
        return "en"
    return "bm" if any(w in tokens for w in ["nak", "ni", "ke", "ada", "air", "kalori", "laktosa"]) else "en"


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


def get_drink_milk_caffeine_reply(
    drink_name: str,
    base_drink: str,
    milk_type: str,
    has_lactose: bool,
    caffeine_mg: float,
    cal: int,
    sugar: float,
    ask_milk: bool,
    ask_caffeine: bool,
    lang: str,
) -> str:
    caf_val = int(round(caffeine_mg))
    has_o_opt = base_drink in ("Kopi", "Teh", "Cham", "Milo")
    if lang == "zh":
        parts = []
        if ask_milk:
            if has_lactose or milk_type not in ("None", ""):
                opt_hint = f"（如不想加奶可点 {base_drink} O / {base_drink} Kosong）" if has_o_opt else ""
                parts.append(f"含奶类（默认使用 {milk_type}，含乳糖）{opt_hint}")
            else:
                parts.append("不含牛奶/乳糖（纯饮品）")
        if ask_caffeine:
            parts.append(f"咖啡因含量约 {caf_val} mg/杯" if caf_val > 0 else "不含咖啡因（0 mg）")
        if not parts:
            parts.append(f"约 {cal} kcal，含糖 {sugar:g}g，咖啡因约 {caf_val} mg，奶类：{milk_type}")
        return f"【{drink_name}】营养查询（来自 KB）：" + "；".join(parts) + "。请问需要为您来一杯吗？"
    elif lang == "bm":
        parts = []
        if ask_milk:
            if has_lactose or milk_type not in ("None", ""):
                opt_hint = f" (boleh pilih {base_drink} O / {base_drink} Kosong jika mahu tanpa susu)" if has_o_opt else ""
                parts.append(f"mengandungi susu ({milk_type}, ada laktosa){opt_hint}")
            else:
                parts.append("bebas susu & bebas laktosa")
        if ask_caffeine:
            parts.append(f"mengandungi kira-kira {caf_val} mg kafein" if caf_val > 0 else "bebas kafein (0 mg)")
        if not parts:
            parts.append(f"kira-kira {cal} kcal, {sugar:g}g gula, {caf_val} mg kafein, susu: {milk_type}")
        return f"Info 【{drink_name}】 (KB): " + "; ".join(parts) + ". Nak order satu boss?"
    else:
        parts = []
        if ask_milk:
            if has_lactose or milk_type not in ("None", ""):
                opt_hint = f" (you can order {base_drink} O / {base_drink} Kosong for dairy-free)" if has_o_opt else ""
                parts.append(f"contains milk ({milk_type}, dairy/lactose){opt_hint}")
            else:
                parts.append("dairy-free & milk-free")
        if ask_caffeine:
            parts.append(f"contains approx. {caf_val} mg caffeine" if caf_val > 0 else "caffeine-free (0 mg)")
        if not parts:
            parts.append(f"approx. {cal} kcal, {sugar:g}g sugar, {caf_val} mg caffeine, milk: {milk_type}")
        return f"【{drink_name}】 (from Nutrition KB): " + "; ".join(parts) + ". Would you like to order one?"


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
def _load_env_groq_key() -> str:
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GROQ_API_KEY="):
                        return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


_ENV_FILE_GROQ_KEY = _load_env_groq_key()


def get_groq_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "") or _ENV_FILE_GROQ_KEY


class TTSRequest(BaseModel):
    text: str
    lang: Optional[str] = "en"
    voice: Optional[str] = None
    tone: Optional[str] = "female"


@app.get("/api/health")
def health_check():
    groq_k = get_groq_api_key()
    return {
        "status": "healthy",
        "brand": "ChatCoffee",
        "model": model_name_str,
        "dataset_rows": dataset_rows_count or 850,
        "classes": list(classes),
        "kb_items": len(df_kb),
        "groq_configured": bool(groq_k),
        "stt_engine": "Groq Whisper Large V3 / WebSpeech" if groq_k else "WebSpeech API (Fallback)",
        "tts_engine": "Microsoft Edge-TTS Neural"
    }


# -------------------------------------------------------------
# Multilingual ASR (Groq Whisper Multi-Track + Domain Rescoring)
# -------------------------------------------------------------
DEFAULT_MULTILINGUAL_PROMPT = (
    "Pesanan Kopitiam drinks order: Kopi, Teh, Teh C Peng, Cham, Siu Dai, Gah Dai, "
    "Kosong, Panas, Suam, Ais, Peng, Kurang Manis, Milo, Neslo, Limau, Teh O Limau, "
    "Americano, White Coffee, Tapau, Bungkus, Takeaway. "
    "南洋咖啡店点单：咖啡，奶茶，茶C，鸳鸯，美禄，美式，白咖啡，少甜，少糖，微糖，无糖，加糖，热，冰，走冰，少冰，打包。"
)
MS_PROMPT = "Pesanan Kopitiam: Kopi, Teh C Peng, Cham, Siu Dai, Kosong, Panas, Suam, Kurang Manis, Milo, Neslo, Ais, Limau, Teh O Limau, Bungkus, Tapau"
ZH_PROMPT = "南洋咖啡店点单：咖啡，奶茶，茶C，鸳鸯，美禄，少甜，无糖，微糖，冰，热，走冰，少冰，打包，白咖啡，美式"
EN_PROMPT = "Kopitiam drinks order: Kopi, Teh, Milo, Cham, Neslo, Americano, less sugar, no sugar, iced, hot, takeaway, tapau"

STT_PHONETIC_FIXES_PY = [
    # Multi-word sweetness / modifiers first
    (re.compile(r"\b(?:see\s*you\s*die|shoot\s*i|she\s*died|siu\s*die|sue\s*die|see\s*die|siew\s*dai|silver\s*die|siu\s*day|sho\s*dai|soda\s*die|seal\s*die)\b", re.IGNORECASE), "siu dai"),
    (re.compile(r"\b(?:god\s*die|gah\s*die|ga\s*die|got\s*die|gar\s*die|guard\s*die)\b", re.IGNORECASE), "gah dai"),
    (re.compile(r"\bpengkura(?:\s*manis)?\b", re.IGNORECASE), "peng kurang manis"),
    (re.compile(r"\b(?:crang\s*manis|orang\s*manis|kura\s*manis|kura2\s*manis|kura-kura\s*manis|kuran\s*manis|cool\s*rang\s*manis)\b", re.IGNORECASE), "kurang manis"),
    (re.compile(r"\bkorang\s*manus\b", re.IGNORECASE), "kurang manis"),
    (re.compile(r"\b(?:go\s*song|ko\s*song|car\s*song|cause\s*song|call\s*song|co\s*song|grow\s*song)\b", re.IGNORECASE), "kosong"),
    (re.compile(r"\b(?:to\s*pow|that\s*pow|top\s*up|ta\s*bao|tarbao|ta\s*pao|da\s*bao|tarpow|tabao|tar\s*bao)\b", re.IGNORECASE), "tapau"),
    (re.compile(r"\b(?:bangkus|boom\s*goose|bonkers|bung\s*kus|bankers)\b", re.IGNORECASE), "bungkus"),
    (re.compile(r"\b(?:milo\s*swam|mellow\s*swam|milo\s*warm)\b", re.IGNORECASE), "milo suam"),
    (re.compile(r"\b(?:swam|swan|soom|suan)\b", re.IGNORECASE), "suam"),
    (re.compile(r"\b(?:pan\s*us|pun\s*us|punas|penis|pa\s*nas)\b", re.IGNORECASE), "panas"),
    # Kopi / Teh combos with numbers / modifiers (covers 'the', 'tea', 'they', 'kopi', 'copy')
    (re.compile(r"\b(?:copy|coffee|coppie|corpy|kopi)\s+o\s+0\b", re.IGNORECASE), "kopi o kosong"),
    (re.compile(r"\b(?:the|tea|they|tay|teh)\s+o\s+0\b", re.IGNORECASE), "teh o kosong"),
    (re.compile(r"\b(?:copy|coffee|coppie|corpy|kopi)\s+0\b", re.IGNORECASE), "kopi o kosong"),
    (re.compile(r"\b(?:the|tea|they|tay|teh)\s+0\b", re.IGNORECASE), "teh o kosong"),
    (re.compile(r"\b(?:copy|coffee|coppie|corpy|kopi)\s*(?:all|old|oh|or|awl|ow|o)\b", re.IGNORECASE), "kopi o"),
    (re.compile(r"\b(?:the|tea|they|tay|teh)\s*(?:all|old|oh|or|awl|ow|o)\b", re.IGNORECASE), "teh o"),
    (re.compile(r"\b(?:milo|mellow|my\s*low)\s*(?:all|old|oh|or|awl|ow|o)\b", re.IGNORECASE), "milo o"),
    (re.compile(r"\b(?:copy|coffee|coppie|corpy|kopi)\s*(?:c|see|sea)\b", re.IGNORECASE), "kopi c"),
    (re.compile(r"\b(?:the|tea|they|tay|teh)\s*(?:c|see|sea)\b", re.IGNORECASE), "teh c"),
    (re.compile(r"\b(?:copy|coffee|coppie|corpy|kopi)\s+kosong\b", re.IGNORECASE), "kopi o kosong"),
    (re.compile(r"\b(?:the|tea|they|tay|teh)\s+kosong\b", re.IGNORECASE), "teh o kosong"),
    (re.compile(r"\b(?:milo|mellow|my\s*low)\s+kosong\b", re.IGNORECASE), "milo o kosong"),
    (re.compile(r"\b(?:copy\s*peng|coffee\s*peng|copy\s*pain|copy\s*pan|kopi\s*pong)\b", re.IGNORECASE), "kopi peng"),
    (re.compile(r"\b(?:kopi|teh|cham|milo|neslo|limau)\s+(?:pain|pen|pang|ping|pong|pan)\b", re.IGNORECASE), lambda m: re.sub(r"\b(?:pain|pen|pang|ping|pong|pan)\b", "peng", m.group(0), flags=re.IGNORECASE)),
    (re.compile(r"\b(?:white\s*copy|white\s*corpy)\b", re.IGNORECASE), "white coffee"),
    (re.compile(r"\b(?:the|tea|they|tay|teh)\s*tar[ie]k\b", re.IGNORECASE), "teh tarik"),
    (re.compile(r"\b(?:teh|tea|the)\s+o?\s*(?:ais|ice|peng)?\s*(?:limau|limo|lime|lemon)\b", re.IGNORECASE), "teh o ais limau"),
    # Single word drinks / words
    (re.compile(r"^\s*bus\b", re.IGNORECASE), "Boss"),
    (re.compile(r"\bbus\b(?=\s*[,.]?\s*(?:bagi|tolong|order|nak|saya|i|can|one|satu|mau))", re.IGNORECASE), "Boss"),
    (re.compile(r"\b(?:copy|coppie|corpy)\b", re.IGNORECASE), "kopi"),
    (re.compile(r"\b(?:tea|tay)\b", re.IGNORECASE), "teh"),
    (re.compile(r"\b(?:charm|chum|champ|chon|chan)\b(?!\s+(?:school|luck))", re.IGNORECASE), "cham"),
    (re.compile(r"\b(?:mellow|my\s*low|milow|mayo|me\s*low)\b", re.IGNORECASE), "milo"),
    (re.compile(r"\b(?:nestle|nesle|nest\s*lo|naslo)\b", re.IGNORECASE), "neslo"),
    (re.compile(r"\bping\b", re.IGNORECASE), "peng"),
    (re.compile(r"\b(?:limua|li\s*mau|lee\s*mau)\b", re.IGNORECASE), "limau"),
    # Chinese Kopitiam fixes
    (re.compile(r"咖啡[瓶病]", re.IGNORECASE), "咖啡冰"),
    (re.compile(r"美[露鹿路陆]", re.IGNORECASE), "美禄"),
    (re.compile(r"(?:原样|远洋|鸳阳)(?=\s*(?:冰|热|打包|一杯|两杯|少甜|少糖|微糖|无糖))", re.IGNORECASE), "鸳鸯"),
    (re.compile(r"少[填田天]", re.IGNORECASE), "少甜"),
    (re.compile(r"走兵", re.IGNORECASE), "走冰"),
    (re.compile(r"((?:咖啡|奶茶|美禄|白咖啡|茶|茶C|鸳鸯|美式|柠檬茶)\s*)大包", re.IGNORECASE), r"\1打包"),
]

def apply_stt_phonetic_fix_py(text: str) -> Tuple[str, bool]:
    fixed = text
    changed = False
    for pattern, repl in STT_PHONETIC_FIXES_PY:
        new_fixed = pattern.sub(repl, fixed)
        if new_fixed != fixed:
            changed = True
            fixed = new_fixed
    return fixed, changed

def score_asr_candidate(text: str) -> Tuple[float, Dict[str, Any]]:
    if not text or not text.strip():
        return -100.0, {"fixed_text": "", "final_score": -100.0, "slots": {}, "changed_by_phonetic_fix": False}

    fixed_text, changed = apply_stt_phonetic_fix_py(text.strip())
    slots = extract_slots(fixed_text)
    cleaned = clean_text(fixed_text)
    intent, conf, _ = predict_intent_ml(cleaned)

    domain_keywords = [
        "kopi", "coffee", "teh", "tea", "cham", "milo", "neslo", "limau", 
        "americano", "white coffee", "latte", "cappuccino",
        "peng", "ping", "ais", "ice", "iced", "panas", "hot", "suam", "warm",
        "kosong", "siu dai", "siew dai", "gah dai", "ga dai", "kurang manis", 
        "kurang ais", "less sugar", "no sugar", "zero sugar", "less ice", "no ice",
        "tapau", "bungkus", "takeaway", "dabao", "ta bao",
        "satu", "dua", "tiga", "cawan", "gelas", "cup", "cups",
        "咖啡", "奶茶", "鸳鸯", "美禄", "白咖啡", "美式",
        "少甜", "少糖", "微糖", "半糖", "无糖", "走糖", "加糖", "多糖",
        "冰", "热", "温", "少冰", "去冰", "走冰", "打包", "一杯", "两杯"
    ]

    text_lower = fixed_text.lower()
    vocab_hits = sum(1 for kw in domain_keywords if kw in text_lower)

    slot_score = (
        (3.5 if "base_drink" in slots else 0.0) +
        (2.0 if "sweetness_level" in slots else 0.0) +
        (2.0 if "temperature" in slots else 0.0) +
        (1.5 if "milk_type" in slots else 0.0) +
        (1.0 if "ice_level" in slots else 0.0)
    )
    num_slots = len(slots)
    if num_slots >= 2:
        slot_score += num_slots * 1.5

    vocab_score = min(vocab_hits * 1.5, 9.0)

    intent_score = 0.0
    if intent in ("order_drink", "modify_order"):
        intent_score = 2.5 * conf
    elif intent in ("clarify_order", "health_query"):
        intent_score = 1.5 * conf

    pack_score = 1.5 if any(w in text_lower for w in ["tapau", "bungkus", "takeaway", "打包"]) else 0.0

    penalty = 0.0
    words = text_lower.split()
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            penalty += 15.0

    # CJK character / loop repetition detection
    cjk_chars = [c for c in fixed_text if '\u4e00' <= c <= '\u9fff']
    if len(cjk_chars) > 6:
        cjk_unique = len(set(cjk_chars)) / len(cjk_chars)
        if cjk_unique < 0.35:
            penalty += 20.0
    if re.search(r"(.{2,10}?)\1{2,}", fixed_text):
        penalty += 15.0

    final_score = slot_score + vocab_score + intent_score + pack_score - penalty

    breakdown = {
        "raw_text": text.strip(),
        "fixed_text": fixed_text,
        "slots": slots,
        "intent": intent,
        "intent_conf": conf,
        "vocab_hits": vocab_hits,
        "slot_score": round(slot_score, 2),
        "vocab_score": round(vocab_score, 2),
        "intent_score": round(intent_score, 2),
        "pack_score": round(pack_score, 2),
        "final_score": round(final_score, 2),
        "changed_by_phonetic_fix": changed
    }
    return final_score, breakdown


@app.post("/api/transcribe")
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    lang: Optional[str] = Form(None),
    model: Optional[str] = Form("whisper-large-v3"),
    multitrack: Optional[str] = Form("true")
):
    groq_key = get_groq_api_key()
    if not groq_key:
        return JSONResponse(status_code=503, content={"error": "GROQ_API_KEY is not configured on server", "status": "error"})

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) > 10 * 1024 * 1024:
            return JSONResponse(status_code=413, content={"error": "Audio file exceeds 10MB limit", "status": "error"})

        # Handle empty / silence audio cleanly
        if len(audio_bytes) < 100:
            return {
                "text": "",
                "raw_text": "",
                "status": "empty",
                "best_track": "none",
                "score": 0.0,
                "phonetic_fix_applied": False,
                "engine": "none",
                "message": "Audio recording is empty or too short"
            }

        filename = file.filename or "recording.webm"
        mime_type = file.content_type or "audio/webm"
        selected_model = model if model in ["whisper-large-v3", "whisper-large-v3-turbo"] else "whisper-large-v3"

        is_multitrack = multitrack and multitrack.lower() in ("true", "1", "yes")

        # -----------------------------------------------------------------
        # Single-Track Mode (Explicit single language or single-pass auto)
        # -----------------------------------------------------------------
        if not is_multitrack:
            use_lang = lang if lang and lang not in ("auto", "multi", "none") else None
            prompt = DEFAULT_MULTILINGUAL_PROMPT
            if use_lang == "zh":
                prompt = ZH_PROMPT
            elif use_lang == "en":
                prompt = EN_PROMPT
            elif use_lang == "ms":
                prompt = MS_PROMPT

            data = {"model": selected_model, "prompt": prompt, "temperature": "0.0"}
            if use_lang:
                data["language"] = use_lang

            def _call_single():
                try:
                    r = _HTTP_SESSION.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {groq_key}"},
                        files={"file": (filename, audio_bytes, mime_type)},
                        data=data,
                        timeout=20
                    )
                    if r.status_code == 429 and data.get("model") != "whisper-large-v3-turbo":
                        data["model"] = "whisper-large-v3-turbo"
                        r = _HTTP_SESSION.post(
                            "https://api.groq.com/openai/v1/audio/transcriptions",
                            headers={"Authorization": f"Bearer {groq_key}"},
                            files={"file": (filename, audio_bytes, mime_type)},
                            data=data,
                            timeout=20
                        )
                    return r
                except Exception as e:
                    return e

            res = await asyncio.to_thread(_call_single)
            if isinstance(res, Exception):
                return JSONResponse(status_code=502, content={"error": f"Groq ASR connection failed: {str(res)}", "status": "error"})
            if res.status_code == 200:
                raw_text = res.json().get("text", "").strip()
                if not raw_text:
                    return {
                        "text": "",
                        "raw_text": "",
                        "status": "empty",
                        "best_track": use_lang or "auto",
                        "score": 0.0,
                        "phonetic_fix_applied": False,
                        "engine": f"Groq {selected_model} (Auto-Detect)" if not use_lang else f"Groq {selected_model} ({use_lang})",
                        "message": "No speech detected"
                    }
                fixed_text, changed = apply_stt_phonetic_fix_py(raw_text)
                return {
                    "text": fixed_text,
                    "raw_text": raw_text,
                    "status": "success",
                    "best_track": use_lang or "auto",
                    "phonetic_fix_applied": changed,
                    "engine": f"Groq {selected_model} (Auto-Detect)" if not use_lang else f"Groq {selected_model} ({use_lang})"
                }
            return JSONResponse(status_code=res.status_code, content={"error": f"Transcription service error: {res.text}", "status": "error"})

        # -----------------------------------------------------------------
        # Multi-Track Parallel Recognition + Domain NLU Rescoring
        # -----------------------------------------------------------------
        tracks = [
            ("auto", None, DEFAULT_MULTILINGUAL_PROMPT),
            ("ms", "ms", MS_PROMPT),
            ("en", "en", EN_PROMPT),
            ("zh", "zh", ZH_PROMPT)
        ]

        def _call_track(track_name: str, track_lang: Optional[str], track_prompt: str):
            payload = {"model": selected_model, "prompt": track_prompt, "temperature": "0.0"}
            if track_lang:
                payload["language"] = track_lang
            try:
                r = _HTTP_SESSION.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    files={"file": (filename, audio_bytes, mime_type)},
                    data=payload,
                    timeout=20
                )
                if r.status_code == 429 and payload.get("model") != "whisper-large-v3-turbo":
                    payload["model"] = "whisper-large-v3-turbo"
                    r = _HTTP_SESSION.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {groq_key}"},
                        files={"file": (filename, audio_bytes, mime_type)},
                        data=payload,
                        timeout=20
                    )
                if r.status_code == 200:
                    return track_name, r.json().get("text", "").strip(), None, 200
                return track_name, "", f"Status {r.status_code}: {r.text}", r.status_code
            except Exception as e:
                return track_name, "", str(e), 500

        results = await asyncio.gather(*[asyncio.to_thread(_call_track, t[0], t[1], t[2]) for t in tracks])

        candidates = []
        any_success_200 = False
        api_errors = []
        for track_name, text, err, status_code in results:
            if status_code == 200:
                any_success_200 = True
                if text:
                    score, breakdown = score_asr_candidate(text)
                    candidates.append({
                        "track": track_name,
                        "text": breakdown["fixed_text"],
                        "raw_text": text,
                        "score": score,
                        "breakdown": breakdown
                    })
            else:
                api_errors.append(f"{track_name}: {err}")

        # If tracks succeeded with 200 but all returned empty text, it was silence/no speech
        if any_success_200 and not candidates:
            return {
                "text": "",
                "raw_text": "",
                "status": "empty",
                "best_track": "none",
                "score": 0.0,
                "phonetic_fix_applied": False,
                "candidates": [],
                "engine": f"Groq {selected_model} (Multi-Track NLU Rescored)",
                "message": "No speech detected"
            }

        # If no track succeeded with 200, it was an actual API / network error
        if not candidates:
            error_details = "; ".join(api_errors) or "All ASR tracks failed"
            return JSONResponse(status_code=502, content={"error": f"Groq ASR failed: {error_details}", "status": "error"})

        # Pick candidate with highest score (tie-break favoring auto track)
        track_priority = {"auto": 3, "zh": 2, "ms": 1, "en": 0}
        candidates.sort(key=lambda c: (c["score"], track_priority.get(c["track"], 0)), reverse=True)
        winner = candidates[0]
        final_text = winner["text"]
        raw_winner_text = winner["raw_text"]

        return {
            "text": final_text,
            "raw_text": raw_winner_text,
            "status": "success",
            "best_track": winner["track"],
            "score": winner["score"],
            "phonetic_fix_applied": winner["breakdown"]["changed_by_phonetic_fix"],
            "candidates": [
                {
                    "track": c["track"],
                    "text": c["text"],
                    "score": c["score"],
                    "slots": c["breakdown"]["slots"]
                }
                for c in candidates
            ],
            "engine": f"Groq {selected_model} (Multi-Track NLU Rescored)"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Audio processing failed: {str(e)}", "status": "error"})


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
        return JSONResponse(status_code=503, content={"error": f"Neural TTS synthesis unavailable: {str(e)}"})


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
    try:
        saved = save_learned_slang(req.slang, req.meaning)
    except ValueError as err:
        return JSONResponse(status_code=400, content={"error": str(err)})
    return {
        "status": "learned",
        "rule": saved,
        "learned_count": len(learned_slang_dict)
    }


@app.post("/api/predict_intent")
def predict_intent_endpoint(req: ChatRequest):
    c_text = clean_text(req.utterance)
    pred, conf, prob_dict = predict_intent_ml(c_text)
    merged_slots = extract_slots(c_text)
    for sk, sv in extract_slots(req.utterance).items():
        merged_slots.setdefault(sk, sv)
    return {
        "raw_text": req.utterance,
        "clean_text": c_text,
        "predicted_intent": pred,
        "confidence": conf,
        "tone": detect_tone(req.utterance),
        "probabilities": prob_dict,
        "slots": merged_slots
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
            "model_name": model_name_str, "tone": "Neutral",
            "confidence": 0.0, "probabilities": {}, "clean_text": "", "slots": {}, "state": state
        }

    # 0. Persistent Human-in-the-Loop Slang Teaching (e.g., "teach: kopi beng = kopi peng")
    teach_match = re.match(r"^(?:teach|learn|slang|教俚语|学习)\s*[:：]?\s*(.+?)\s*(?:=|->|=>|means|是)\s*(.+)$", utterance, re.IGNORECASE)
    if teach_match:
        new_slang, new_meaning = teach_match.group(1).strip(), teach_match.group(2).strip()
        try:
            save_learned_slang(new_slang, new_meaning)
            response = (
                f"已永久学会新俚语：「{new_slang}」→「{new_meaning}」（已保存至 learned_slang.json，下次点单立即生效）。"
                if is_zh else
                f"Got it boss! Permanently learned slang '{new_slang}' -> '{new_meaning}' (saved to learned_slang.json for future orders)."
            )
            intent_tag = "learn_slang"
        except ValueError as err:
            response = f"无法学习该俚语：{err}" if is_zh else f"Sorry boss, cannot learn that slang: {err}"
            intent_tag = "fallback"
        return {
            "response": response, "intent": intent_tag, "model_intent": intent_tag,
            "model_name": model_name_str, "tone": "Neutral", "confidence": 1.0,
            "probabilities": {}, "clean_text": utterance.lower(), "slots": {}, "state": state
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
            "model_name": model_name_str, "tone": detected_tone,
            "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # 2. Reset / Cancel command
    is_cancel_cmd = (
        t_low in ["reset", "clear", "semula", "batal", "cancel", "重置", "取消", "取消订单"] or
        bool(re.search(r"\b(?:cancel\s+(?:my\s+)?order|batal\s+pesanan)\b", t_low)) or
        (pred_intent == "cancel_order" and conf >= 0.30 and not extracted_slots)
    )
    if is_cancel_cmd:
        state.update({
            "status": "IDLE", "base_drink": None, "temperature": None,
            "sweetness_level": None, "milk_type": None, "temp_detail": "Hot",
            "ice_level": "Normal Ice", "pending_slot": None
        })
        response = "好的，已重置订单！请问今天想喝什么？" if is_zh else "Pesanan dikosongkan. Jom order semula boss! Nak minum apa hari ini?"
        ui_act, tgt_scr = UI_ACTION_MAP["system_reset"]
        return {
            "response": response, "intent": "system_reset", "model_intent": pred_intent,
            "ui_action": ui_act, "target_screen": tgt_scr,
            "model_name": model_name_str, "tone": detected_tone,
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
    if was_awaiting_confirm and (is_strict_confirm or (pred_intent == "order_confirm" and conf >= 0.30)) and not has_modify_cue and not extracted_slots:
        state["status"] = "DISPENSING"
        response = (
            "订单已确认！正在为您现磨冲泡，请稍候从出料口取杯，谢谢老板！"
            if is_zh else
            "Order confirmed! Mesin sedang bancuh kopi anda. Sila ambil cawan di dispenser, terima kasih boss!"
        )
        ui_act, tgt_scr = UI_ACTION_MAP["order_confirm"]
        return {
            "response": response, "intent": "order_confirm", "model_intent": pred_intent,
            "ui_action": ui_act, "target_screen": tgt_scr,
            "model_name": model_name_str, "tone": detected_tone,
            "confidence": 1.0, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # 3.5 Price Inquiry (prevent hijacking drink orders like "Teh O Peng" when no price cue is present)
    has_price_cue = bool(
        re.search(r"\b(?:how\s+much|price|pricing|cost|harga|brp|berapa(?:\s+ringgit)?)\b", t_low) or
        any(w in utterance for w in ["多少钱", "几块", "几钱", "价钱", "价格"])
    )
    is_price_inquiry = bool(
        has_price_cue or
        (pred_intent == "price_query" and conf >= 0.35 and not extracted_slots)
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
            "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict,
            "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # 4. Health, Calorie & Allergen RAG query grounded on unified_coffee_kb.csv (Direct lookup with 4-tier calorie evaluation and allergen check, no cart mutation)
    is_allergen_inquiry = bool(
        re.search(r"\b(?:lactose|laktosa|gluten|allergen|allergens|allergi|alergi|alahan|alergik|allergic|allergy|intoleran|intolerant|intolerance|intoleransi|dairy|celiac)\b", t_low) or
        any(w in utterance for w in ["乳糖", "麸质", "过敏", "乳糜泻", "过敏原", "奶制品"])
    )
    is_milk_inquiry = bool(
        re.search(r"\b(?:milk|susu|dairy)\b", t_low) or
        any(w in utterance for w in ["含奶", "有奶", "加奶", "牛奶", "炼奶", "炼乳", "淡奶"])
    )
    is_caffeine_inquiry = bool(
        re.search(r"\b(?:caffeine|kafein|decaf)\b", t_low) or
        any(w in utterance for w in ["咖啡因", "提神"])
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
        re.search(r"\b(?:what|apa|which|mana|ada|rekomen|recommend|suggest|cadang|does|contain|contains|have)\b", t_low) or
        "?" in utterance or "？" in utterance
    )

    is_health_rag = (
        is_allergen_inquiry or
        is_caffeine_inquiry or
        (is_milk_inquiry and (has_health_question or pred_intent == "health_query")) or
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

        # Milk / Caffeine specific KB query (e.g., "Does cham contain milk or caffeine?")
        if (is_milk_inquiry or is_caffeine_inquiry) and target_drink and (is_caffeine_inquiry or not is_allergen_inquiry):
            temp_state = {"base_drink": target_drink, "milk_type": target_milk, "temperature": target_temp, "sweetness_level": target_sweet}
            kb_info = lookup_price_and_nutrition(temp_state)
            dname = kb_info.get("display_name") or target_drink
            response = get_drink_milk_caffeine_reply(
                drink_name=dname,
                base_drink=target_drink,
                milk_type=kb_info.get("milk_type") or (target_milk or "Condensed Milk"),
                has_lactose=kb_info["contains_lactose"],
                caffeine_mg=kb_info["caffeine_mg"],
                cal=kb_info["calories"],
                sugar=kb_info["sugars_g"],
                ask_milk=is_milk_inquiry,
                ask_caffeine=is_caffeine_inquiry,
                lang=user_lang,
            )
            return {
                "response": response, "intent": "health_query", "model_intent": pred_intent,
                "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
            }

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
                    "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
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
                "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
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
                    "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
                }

            temp_state = {"base_drink": target_drink, "milk_type": target_milk, "temperature": target_temp, "sweetness_level": target_sweet}
            kb_info = lookup_price_and_nutrition(temp_state)
            cal = kb_info["calories"]
            sug = kb_info["sugars_g"]
            dname = kb_info.get("display_name") or target_drink
            response = classify_calorie_tier(cal, dname, sug, user_lang)
            return {
                "response": response, "intent": "health_query", "model_intent": pred_intent,
                "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
            }

        # General health query (if a specific drink was asked, answer with its KB nutrition profile)
        if target_drink and not is_recommendation_query:
            temp_state = {"base_drink": target_drink, "milk_type": target_milk, "temperature": target_temp, "sweetness_level": target_sweet}
            kb_info = lookup_price_and_nutrition(temp_state)
            dname = kb_info.get("display_name") or target_drink
            response = get_drink_milk_caffeine_reply(
                drink_name=dname,
                base_drink=target_drink,
                milk_type=kb_info.get("milk_type") or (target_milk or "Condensed Milk"),
                has_lactose=kb_info["contains_lactose"],
                caffeine_mg=kb_info["caffeine_mg"],
                cal=kb_info["calories"],
                sugar=kb_info["sugars_g"],
                ask_milk=False,
                ask_caffeine=False,
                lang=user_lang,
            )
        else:
            response = query_healthy_drinks_from_kb(user_lang)
        return {
            "response": response, "intent": "health_query", "model_intent": pred_intent,
            "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # 4.1 Lightweight keyword routing for Menu Browse & QR/Cashless Payment QA in 8-intent mode
    is_menu_browse_kw = (
        not extracted_slots.get("base_drink") and not state.get("pending_slot") and (
            bool(re.search(r"\b(?:open|show|view|see)\b.*\bmenu\b|\b(?:drink|drinks)\s+menu\b|\btunjuk\s+menu\b", t_low)) or
            any(w in utterance for w in ["打开菜单", "看菜单", "饮品菜单", "打开饮品菜单"])
        )
    )
    if is_menu_browse_kw:
        user_lang = detect_lang(utterance)
        ui_act, tgt_scr = UI_ACTION_MAP["navigate_menu"]
        response = (
            "已为您打开菜单！可选：Kopi、Teh、Cham、White Coffee、Milo、Neslo、Teh O Limau、Americano（支持热饮/温饮/少冰/去冰及甜度定制）。"
            if user_lang == "zh" else
            ("Menu dibuka boss! Pilihan: Kopi, Teh, Cham, White Coffee, Milo, Neslo, Teh O Limau & Americano."
             if user_lang == "bm" else
             "Menu opened! Available drinks: Kopi, Teh, Cham, White Coffee, Milo, Neslo, Teh O Limau & Americano (Hot/Warm/Iced with custom ice & sugar).")
        )
        return {
            "response": response, "intent": "navigate_menu", "model_intent": pred_intent,
            "ui_action": ui_act, "target_screen": tgt_scr,
            "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict,
            "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    is_payment_qa_kw = (
        not extracted_slots.get("base_drink") and not state.get("pending_slot") and not was_awaiting_confirm and (
            bool(re.search(r"\b(?:pay|payment|qr(?:\s*code)?|duitnow|tng|ewallet|e-wallet|cashless|paywave|bayar)\b", t_low)) or
            any(w in utterance for w in ["扫码", "怎么付", "付款方式", "支付方式", "能用qr", "二维码"])
        )
    )
    if is_payment_qa_kw:
        user_lang = detect_lang(utterance)
        ui_act, tgt_scr = UI_ACTION_MAP["payment_query"]
        if user_lang == "zh":
            response = "ChatCoffee 支持无现金扫码支付：DuitNow QR、Touch 'n Go eWallet、GrabPay、Boost、ShopeePay，以及 Visa/Mastercard PayWave 与 MyDebit 银行卡（暂不收现金）。"
        elif user_lang == "bm":
            response = "ChatCoffee menyokong pembayaran tanpa tunai: DuitNow QR, Touch 'n Go eWallet, GrabPay, Boost, ShopeePay serta kad Visa/Mastercard PayWave & MyDebit."
        else:
            response = "ChatCoffee is a cashless kiosk supporting DuitNow QR, Touch 'n Go eWallet, GrabPay, Boost, ShopeePay, and Visa/Mastercard PayWave / MyDebit cards."
        return {
            "response": response, "intent": "payment_query", "model_intent": pred_intent,
            "ui_action": ui_act, "target_screen": tgt_scr,
            "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict,
            "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # 4.2 Payment Policy QA & Kiosk UI Navigation Intents (v1.5 14-intent support)
    if pred_intent == "payment_query" and not extracted_slots:
        user_lang = detect_lang(utterance)
        if user_lang == "zh":
            response = "ChatCoffee 支持无现金扫码支付：DuitNow QR、Touch 'n Go eWallet、GrabPay、Boost、ShopeePay，以及 Visa/Mastercard PayWave 与 MyDebit 银行卡（暂不收现金，屏幕会显示电子收据）。"
        elif user_lang == "bm":
            response = "ChatCoffee menyokong pembayaran tanpa tunai: DuitNow QR, Touch 'n Go eWallet, GrabPay, Boost, ShopeePay serta kad Visa/Mastercard PayWave & MyDebit (resit dipaparkan di skrin)."
        else:
            response = "ChatCoffee is a cashless kiosk supporting DuitNow QR, Touch 'n Go eWallet, GrabPay, Boost, ShopeePay, and Visa/Mastercard PayWave / MyDebit cards (with on-screen digital receipt)."
        ui_act, tgt_scr = UI_ACTION_MAP["payment_query"]
        return {
            "response": response, "intent": "payment_query", "model_intent": pred_intent,
            "ui_action": ui_act, "target_screen": tgt_scr,
            "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict,
            "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    if pred_intent in ("navigate_menu", "navigate_payment", "review_order", "navigate_home") and not extracted_slots:
        user_lang = detect_lang(utterance)
        ui_act, tgt_scr = UI_ACTION_MAP[pred_intent]
        if pred_intent == "navigate_menu":
            response = (
                "已为您打开菜单！可选：Kopi、Teh、Cham、White Coffee、Milo、Neslo、Teh O Limau、Americano（支持热饮/温饮/少冰/去冰及甜度定制）。"
                if user_lang == "zh" else
                "Menu dibuka boss! Pilihan: Kopi, Teh, Cham, White Coffee, Milo, Neslo, Teh O Limau & Americano."
            )
        elif pred_intent == "navigate_home":
            response = "已返回首页画面！随时告诉我您想喝什么。" if user_lang == "zh" else "Kembali ke skrin utama! Sila beritahu minuman pilihan anda."
        elif pred_intent == "review_order":
            if state.get("base_drink"):
                summary = f"{state.get('base_drink')} {state.get('temperature') or ''} ({state.get('sweetness_level') or 'Normal'}, {state.get('milk_type') or 'Condensed Milk'})"
                response = (
                    f"您当前的订单：{summary}，合计 RM {float(state.get('price_rm') or 0.0):.2f}。"
                    if user_lang == "zh" else
                    f"Pesanan semasa anda: {summary}, jumlah RM {float(state.get('price_rm') or 0.0):.2f}."
                )
            else:
                response = "当前购物车还是空的，请问想喝哪一款饮品？" if user_lang == "zh" else "Troli masih kosong boss! Nak order minuman apa?"
        else:  # navigate_payment
            if state.get("base_drink") and state.get("temperature") and state.get("sweetness_level"):
                kb_info = lookup_price_and_nutrition(state)
                state["price_rm"] = kb_info["price_rm"]
                response = (
                    f"已为您打开付款页面！应付金额 RM {kb_info['price_rm']:.2f}，请扫码 DuitNow QR / TnG eWallet 完成支付。"
                    if user_lang == "zh" else
                    f"Skrin pembayaran dibuka! Jumlah RM {kb_info['price_rm']:.2f}. Sila imbas DuitNow QR / TnG eWallet untuk bayar."
                )
            else:
                tgt_scr = "detail"
                response = (
                    "请先选好饮品、冷热与甜度后再前往付款哦！"
                    if user_lang == "zh" else
                    "Sila lengkapkan pilihan minuman, suhu & tahap manis sebelum ke skrin pembayaran ya!"
                )
        return {
            "response": response, "intent": pred_intent, "model_intent": pred_intent,
            "ui_action": ui_act, "target_screen": tgt_scr,
            "model_name": model_name_str, "tone": detected_tone, "confidence": conf, "probabilities": prob_dict,
            "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # Out-of-scope / low-confidence rejection (tuned for 14-class probability scale)
    is_high_conf_oos = pred_intent == "out_of_scope" and conf >= 0.35 and not extracted_slots.get("base_drink")
    if (is_high_conf_oos or (conf < 0.20 and not extracted_slots)) and not was_awaiting_confirm and not state.get("pending_slot"):
        response = (
            "抱歉，我只负责 ChatCoffee 无人咖啡机点单服务。请问您想喝什么饮料呢？可选 Kopi、Teh、Cham、Milo、White Coffee 等。"
            if is_zh else
            "Maaf boss, saya hanya bantu uruskan pesanan ChatCoffee kiosk. Nak minum apa? Kita ada Kopi, Teh, Cham, Milo, White Coffee."
        )
        ui_act, tgt_scr = UI_ACTION_MAP["fallback"]
        return {
            "response": response, "intent": "fallback", "model_intent": pred_intent,
            "ui_action": ui_act, "target_screen": tgt_scr,
            "model_name": model_name_str, "tone": detected_tone,
            "confidence": conf, "probabilities": prob_dict, "clean_text": c_text, "slots": {}, "state": state
        }

    # 5. Guard empty modify_order (e.g. "tukar mana, bro???" or "tukar") before slot merge
    if pred_intent == "modify_order" and not extracted_slots and state.get("base_drink") and not state.get("pending_slot"):
        user_lang = detect_lang(utterance)
        if user_lang == "zh":
            response = "好的！请问您想修改哪一项呢？（例如更换饮品、改冷/热、调整糖度或奶类）"
        elif user_lang == "bm":
            response = "Boleh boss! Nak tukar yang mana satu? (Contoh: tukar air, Panas/Ais, kurang manis, atau susu)"
        else:
            response = "Sure boss! What would you like to change—drink, Hot/Iced, sweetness, or milk type?"
        ui_act, tgt_scr = UI_ACTION_MAP.get(pred_intent, ("update_selection", "detail"))
        return {
            "response": response, "intent": pred_intent, "model_intent": pred_intent,
            "ui_action": ui_act, "target_screen": tgt_scr,
            "model_name": model_name_str, "tone": detected_tone,
            "confidence": conf, "probabilities": prob_dict,
            "clean_text": c_text, "slots": extracted_slots, "state": state
        }

    # 5.1 Merge ALL extracted slots (preserving multi-slot replies during pending_slot clarification)
    is_drink_changed = bool(extracted_slots.get("base_drink") and extracted_slots["base_drink"] != state.get("base_drink"))
    is_explicit_base_mention = bool(extracted_slots.get("base_drink") and not extracted_slots.get("milk_type"))
    explicit_milk_specified = bool(extracted_slots.get("milk_type"))

    if state.get("pending_slot") and state["pending_slot"] in extracted_slots:
        state["pending_slot"] = None
    for k, v in extracted_slots.items():
        state[k] = v

    if is_drink_changed or is_explicit_base_mention:
        if not explicit_milk_specified:
            state["milk_type"] = "None" if state.get("base_drink") in ("Teh O Limau", "Americano") else "Condensed Milk"
        if not extracted_slots.get("sweetness_level"):
            state["sweetness_level"] = "Zero Sugar" if state.get("base_drink") == "Americano" else "Normal"

    # Normalize temperature-dependent detail fields
    if state.get("temperature") == "Iced":
        state["temp_detail"] = None
        if not state.get("ice_level"):
            state["ice_level"] = "Normal Ice"
    elif state.get("temperature") == "Hot":
        state["temp_detail"] = state.get("temp_detail") or "Hot"
        state["ice_level"] = None

    # Enforce milk & condensed-milk zero-sugar constraints across ALL intents
    if state.get("base_drink") in ("Teh O Limau", "Americano"):
        state["milk_type"] = "None"
    if state.get("base_drink") == "Americano" and not state.get("sweetness_level"):
        state["sweetness_level"] = "Zero Sugar"
    condensed_drinks = ("Cham", "Kopi", "Teh", "White Coffee", "Milo", "Neslo")
    is_condensed = state.get("milk_type") == "Condensed Milk" or (
        state.get("base_drink") in condensed_drinks and state.get("milk_type") not in ("None", "Evaporated Milk")
    )
    zero_sugar_adjusted = False
    if state.get("sweetness_level") == "Zero Sugar" and is_condensed:
        if state.get("base_drink") in ("Kopi", "Teh", "Milo", "Cham", "Neslo") and not explicit_milk_specified and "tarik" not in t_low:
            state["milk_type"] = "None"
        else:
            state["sweetness_level"] = "Less Sugar"
            zero_sugar_adjusted = True

    zero_sugar_notice = ""
    if zero_sugar_adjusted:
        user_lang = detect_lang(utterance)
        if user_lang == "zh":
            zero_sugar_notice = "（提示：炼奶本身含糖，无法做到完全无糖 Kosong，已为您调整为少糖 Less Sugar。）"
        elif user_lang == "bm":
            zero_sugar_notice = " (Nota: Susu pekat mengandungi gula dan tidak boleh Kosong / Tanpa Gula, jadi telah dilaraskan ke Kurang Manis / Less Sugar.)"
        else:
            zero_sugar_notice = " (Note: Condensed milk contains sugar, so Zero Sugar / Kosong is not available; adjusted to Less Sugar.)"

    # 6. Check missing mandatory slots
    if not state.get("base_drink"):
        response = "请问您想喝什么饮料呢？我们有 Kopi、Teh、Cham、Milo、White Coffee 等。" if is_zh else "Nak minum apa boss? Kita ada Kopi, Teh, Cham, Milo, White Coffee dan lain-lain."
        state["pending_slot"] = "base_drink"
        state["status"] = "GATHERING_SLOTS"
    elif not state.get("temperature"):
        drink = state["base_drink"]
        if drink in ("Kopi", "Teh", "Cham", "Milo"):
            if state.get("milk_type") == "None":
                drink = f"{drink} O"
            elif state.get("milk_type") == "Evaporated Milk":
                drink = f"{drink} C"
        zs_prefix = (zero_sugar_notice.strip() + " ") if zero_sugar_notice else ""
        response = f"{zs_prefix}您选了 {drink}。请问要热的（Hot）还是加冰（Iced / Peng）呢？" if is_zh else f"{zs_prefix}Bancuh {drink}. Nak panas atau sejuk / ais (Peng) boss?"
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
        temp_val = state.get("temperature", "")
        if temp_val == "Hot" and state.get("temp_detail") == "Warm":
            temp_label = "Warm (Suam)"
        else:
            temp_label = temp_val
        drink_name = f"{state['base_drink']} {temp_label}".strip()
        sweet = state.get("sweetness_level", "Normal")
        milk = state.get("milk_type", "Condensed Milk")
        mod_parts = [sweet, milk]
        if temp_val == "Iced" and state.get("ice_level") in ("Less Ice", "No Ice"):
            mod_parts.append(state["ice_level"])
        mod_str = "，".join(mod_parts) if is_zh else ", ".join(mod_parts)
        if detected_tone == "Urgent":
            response = (
                f"⚡ 极速模式：{drink_name}（{mod_str}），RM {kb_info['price_rm']:.2f}。说「确认」立刻出杯！{zero_sugar_notice}"
                if is_zh else
                f"⚡ Express mode boss! {drink_name} ({mod_str}), RM {kb_info['price_rm']:.2f}. Say Confirm to brew immediately!{zero_sugar_notice}"
            )
        else:
            response = (
                f"好的！一杯 {drink_name}（{mod_str}）。总共 RM {kb_info['price_rm']:.2f}。请问确认下单吗？（请回答 OK / Confirm）{zero_sugar_notice}"
                if is_zh else
                f"Beres boss! Satu {drink_name} ({mod_str}). Total RM {kb_info['price_rm']:.2f}. Nak confirm order sekarang? (Jawab Confirm / Boleh){zero_sugar_notice}"
            )

    ui_act, tgt_scr = UI_ACTION_MAP.get(pred_intent, ("prepare_selection", "detail"))
    return {
        "response": response,
        "intent": pred_intent,
        "model_intent": pred_intent,
        "ui_action": ui_act,
        "target_screen": tgt_scr,
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


@app.get("/models/chatcoffee_model_web.json")
def serve_web_model():
    if os.path.exists(WEB_MODEL_PATH):
        return FileResponse(
            WEB_MODEL_PATH,
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(status_code=404, content={"error": "chatcoffee_model_web.json not found"})


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
