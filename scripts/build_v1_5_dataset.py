"""Build ChatCoffee v1.5 (2,100 rows x 24 columns, 14 intents) from v1 baseline.

Preserves all 950 baseline rows in NLP_Datasets/chatcoffee_gold_850.csv (CC_0001..CC_0950),
upgrades them to the 24-column schema (accurate language_mix, temp_detail, ice_level,
ui_action, shortform_present, stt_noise_present, scenario_group, split), and adds
1,150 curated, deduplicated Malaysian English / Malay / Chinese / Cantonese / Rojak /
shortform / STT-noise / fragment-reply / hard-negative utterances across all 14 intents.

Exports NLP_Datasets/chatcoffee_v1_5_2100.csv with UTF-8 BOM (utf-8-sig) for Excel.
"""

import json
import os
import random
import re
from typing import Dict, List, Tuple

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "chatcoffee_gold_850.csv")
OUT_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "chatcoffee_v1_5_2100.csv")

COLUMNS = [
    "id",
    "category",
    "user_utterance",
    "language_mix",
    "intent",
    "slots",
    "context",
    "base_drink",
    "temperature",
    "temp_detail",
    "ice_level",
    "sweetness_level",
    "milk_type",
    "old_value",
    "new_value",
    "query_type",
    "turn_id",
    "prev_bot_prompt",
    "is_coreference",
    "shortform_present",
    "stt_noise_present",
    "ui_action",
    "scenario_group",
    "split",
]

INTENT_META = {
    "order_drink": ("direct_order", "prepare_selection"),
    "modify_order": ("mid_order_correction", "update_selection"),
    "clarify_order": ("ambiguous_exploratory", "answer_or_clarify"),
    "health_query": ("health_qa", "lookup_nutrition"),
    "price_query": ("price_qa", "lookup_price"),
    "order_confirm": ("order_control", "review_and_confirm"),
    "cancel_order": ("order_control", "request_cancel"),
    "single_cup_limit": ("order_control", "enforce_single_cup"),
    "payment_query": ("payment_qa", "answer_payment_policy"),
    "navigate_menu": ("navigation", "show_menu"),
    "navigate_payment": ("navigation", "show_payment"),
    "review_order": ("navigation", "show_order"),
    "navigate_home": ("navigation", "show_home"),
    "out_of_scope": ("fallback", "safe_fallback"),
}

SHORTFORM_TOKENS = {
    "nk", "xnak", "xnk", "xmau", "xmo", "xbo", "xleh", "xde", "xtau", "x",
    "krg", "krng", "mns", "gla", "ssu", "pns", "sjk", "byr", "hrg", "hrga",
    "brp", "sd", "siudai", "gd", "gadai", "ksg", "kp", "kpi", "wc", "amer",
    "tkr", "cncl", "pls", "plz", "tq", "thx", "bgi", "bg", "bgtau", "blh",
    "utk", "yg", "dgn", "ngn", "dlm", "kt", "dkt", "skrg", "skit", "sgt",
    "tp", "tpi", "tlg", "sy", "td", "bkn", "oder", "req", "tny", "plg",
    "lbh", "xjadi", "jgn", "dpt", "org", "mnm", "stu", "dh", "aje", "je",
}

MS_WORDS = {
    "saya", "nak", "bagi", "tolong", "buat", "satu", "dua", "tiga", "cawan",
    "panas", "suam", "ais", "sejuk", "dingin", "kurang", "manis", "gula",
    "tanpa", "susu", "pekat", "sejat", "cair", "biasa", "lebih", "tukar",
    "ganti", "ubah", "batal", "pesanan", "minuman", "minum", "harga",
    "berapa", "boleh", "ada", "apa", "mana", "yang", "untuk", "dalam",
    "kalori", "kafein", "laktosa", "sihat", "rendah", "tinggi", "bayar",
    "guna", "tunai", "kad", "resit", "menu", "utama", "skrin", "balik",
    "semula", "sahkan", "pesanan", "sekarang", "tunjuk", "lihat", "semak",
    "tak", "tidak", "tiada", "jangan", "bukan", "tadi", "jadi", "kepada",
    "sikit", "banyak", "atau", "dan", "kat", "dekat", "sini", "ni", "tu",
    "ke", "pun", "lagi", "terus", "je", "lah", "ya", "air", "teh", "kopi",
    "tarik", "limau", "kosong", "kencing", "manis", "diet", "pahit",
    "beza", "dengan", "pilihan", "cadang", "cadangan", "sesuai", "pertama",
    "kali", "mahu", "minta", "pesan", "ambil", "pesan", "maaf", "tunggu",
    "sekejap", "jap", "kejap", "awak", "anda", "sila", "terima", "kasih",
    "hari", "ini", "hujan", "cuaca", "kereta", "telefon", "baju", "nasi",
    "lemak", "ayam", "goreng", "roti", "canai", "makan", "lapar", "kerja",
    "kedai", "tutup", "buka", "pergi", "ke", "halaman", "awal", "mula",
    "semakan", "troli", "bakul", "kira", "jumlah", "semua", "baki", "syiling",
    "imbas", "kod", "dompet", "digital", "sahaja", "terus", "cepat",
}

EN_WORDS = {
    "i", "id", "ill", "im", "want", "like", "would", "get", "me", "give",
    "make", "order", "one", "two", "three", "cup", "cups", "coffee", "tea",
    "white", "lemon", "hot", "warm", "iced", "ice", "cold", "less", "zero",
    "no", "normal", "extra", "sugar", "sweet", "sweetness", "milk",
    "condensed", "evaporated", "without", "black", "please", "thanks",
    "thank", "you", "can", "could", "may", "is", "are", "the", "a", "an",
    "this", "that", "it", "my", "your", "drink", "drinks", "beverage",
    "menu", "price", "how", "much", "cost", "many", "calories", "kcal",
    "caffeine", "lactose", "gluten", "allergen", "allergens", "dairy",
    "healthy", "healthiest", "lowest", "low", "high", "fat", "compare",
    "difference", "between", "recommend", "suggest", "suggestion",
    "change", "switch", "replace", "modify", "remove", "instead", "from",
    "to", "cancel", "reset", "clear", "confirm", "proceed", "checkout",
    "pay", "payment", "qr", "card", "cash", "ewallet", "receipt", "back",
    "home", "screen", "cart", "review", "show", "open", "view", "check",
    "current", "what", "which", "any", "do", "have", "available", "for",
    "in", "on", "with", "of", "and", "or", "today", "weather", "forecast",
    "book", "flight", "play", "song", "spotify", "repair", "laptop",
    "phone", "battery", "exam", "homework", "pizza", "burger", "chicken",
    "sorry", "wait", "actually", "never", "mind", "only", "just", "now",
    "first", "time", "bitter", "strong", "diabetic", "friendly", "weight",
    "loss", "loss", "option", "options", "contains", "contain", "free",
    "does", "there", "here", "ready", "go", "ahead", "place", "done",
    "stop", "abort", "start", "over", "again", "another", "more", "fast",
    "hurry", "quick", "catch", "train", "bus", "air", "conditioning",
}

ZH_CANTO_HOKKIEN_OR_MANDARIN = True


def detect_shortform(text: str) -> str:
    t = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text.lower())
    t = re.sub(r"([\u4e00-\u9fff])([a-z0-9])", r"\1 \2", t)
    t = re.sub(r"([a-z0-9])([\u4e00-\u9fff])", r"\1 \2", t)
    toks = set(t.split())
    return "yes" if (toks & SHORTFORM_TOKENS) else "no"


def infer_language_mix(text: str) -> str:
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))
    t = re.sub(r"[\u4e00-\u9fff]", " ", text.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    words = [w for w in t.split() if not w.isdigit()]
    ms_hits = sum(1 for w in words if w in MS_WORDS or w in SHORTFORM_TOKENS)
    en_hits = sum(1 for w in words if w in EN_WORDS)
    # Local drink names like kopi, teh, cham, milo, neslo, limau, peng, kosong, siu dai
    # can appear in English sentences without making them Malay unless Malay grammar/words appear.
    parts = []
    if ms_hits > 0:
        parts.append("ms")
    if en_hits > 0:
        parts.append("en")
    if has_zh:
        parts.append("zh-cmn")
    if not parts:
        return "zh-cmn" if has_zh else "en"
    if parts == ["ms", "en", "zh-cmn"]:
        return "ms, en, zh-cmn"
    if parts == ["ms", "en"]:
        return "ms, en"
    if parts == ["en", "zh-cmn"]:
        return "zh-cmn, en"
    if parts == ["ms", "zh-cmn"]:
        return "ms, zh-cmn"
    return parts[0]


def infer_query_type(text: str, intent: str) -> str:
    t = text.lower()
    if intent == "price_query":
        return "price"
    if intent == "payment_query":
        return "payment_policy"
    if intent != "health_query":
        return ""
    if re.search(r"lactose|laktosa|乳糖|dairy|奶制品", t):
        if re.search(r"recommend|suggest|cadang|pilihan|apa|which|推荐|有哪些|什么|diet", t) and not re.search(r"\b(?:kopi|teh|cham|milo|neslo|americano|white coffee)\b|咖啡|茶|美禄|白咖啡", t):
            return "allergen_rec"
        return "lactose"
    if re.search(r"gluten|麸质|barli|麦芽", t):
        if re.search(r"recommend|suggest|cadang|pilihan|apa|which|推荐|有哪些|什么|free", t) and not re.search(r"\b(?:kopi|teh|cham|milo|neslo|americano|white coffee)\b|咖啡|茶|美禄|白咖啡", t):
            return "allergen_rec"
        return "gluten"
    if re.search(r"allerg|alergi|alahan|过敏", t):
        return "allergen_rec"
    if re.search(r"caffeine|kafein|咖啡因|提神|睡不着|tidur", t):
        return "caffeine"
    if re.search(r"\b(?:vs|atau|or|compare|banding|mana\s+lebih|mana\s+kurang)\b|还是|哪个更|对比", t):
        return "comparison"
    if re.search(r"sugar|gula|糖|diabetic|kencing\s+manis|糖尿", t):
        if re.search(r"recommend|suggest|cadang|ada|which|mana|apa|推荐|什么|适合", t) and not re.search(r"\b(?:kopi|teh|cham|milo|neslo|americano|white coffee)\b|咖啡|茶|美禄|白咖啡", t):
            return "low_cal_rec"
        return "sugar"
    if re.search(r"calori|kalori|kcal|卡路里|热量|卡|fat|gemuk|胖", t):
        if re.search(r"recommend|suggest|cadang|ada|which|mana|apa|推荐|什么|最健康|最低", t) and not re.search(r"\b(?:kopi|teh|cham|milo|neslo|americano|white coffee|this|ni|这杯)\b|咖啡|茶|美禄|白咖啡|这杯", t):
            return "low_cal_rec"
        return "calories"
    if re.search(r"diet|weight|sihat|healthy|减肥|减脂|健康|控糖|养生", t):
        return "low_cal_rec"
    return "nutrition_general"


def infer_coref(text: str, intent: str) -> str:
    if intent not in ("health_query", "price_query", "modify_order"):
        return "no"
    t = text.lower()
    if re.search(
        r"(?:这杯|这个|这款|当前这杯|刚才点的|选的那杯|那杯|this\s+cup|this\s+drink|this\s+one|current\s+(?:drink|order)|my\s+(?:current\s+)?(?:drink|order)|air\s+ni|cawan\s+ni|minuman\s+ni|air\s+ini|air\s+tu|order\s+td|td)",
        t,
    ):
        return "yes"
    return "no"


def build_slots_json(row_dict: Dict[str, str]) -> str:
    keys = ["base_drink", "temperature", "temp_detail", "ice_level", "sweetness_level", "milk_type"]
    d = {k: row_dict[k] for k in keys if row_dict.get(k)}
    return json.dumps(d, ensure_ascii=False)


def make_row(
    idx: int,
    utterance: str,
    intent: str,
    base_drink: str = "",
    temperature: str = "",
    temp_detail: str = "",
    ice_level: str = "",
    sweetness_level: str = "",
    milk_type: str = "",
    old_value: str = "",
    new_value: str = "",
    query_type: str = "",
    turn_id: int = 1,
    prev_bot_prompt: str = "none",
    is_coreference: str = "",
    stt_noise_present: str = "no",
    scenario_group: str = "",
    lang_override: str = "",
) -> Dict[str, object]:
    cat, ui_action = INTENT_META[intent]
    if temperature == "Hot" and not temp_detail:
        temp_detail = "Hot"
    if temperature == "Iced" and not ice_level:
        ice_level = "Normal Ice"
    row = {
        "id": f"CCV15_{idx:04d}",
        "category": cat,
        "user_utterance": utterance.strip(),
        "language_mix": lang_override or infer_language_mix(utterance),
        "intent": intent,
        "context": "walk_in_kiosk",
        "base_drink": base_drink,
        "temperature": temperature,
        "temp_detail": temp_detail,
        "ice_level": ice_level,
        "sweetness_level": sweetness_level,
        "milk_type": milk_type,
        "old_value": old_value,
        "new_value": new_value,
        "query_type": query_type or infer_query_type(utterance, intent),
        "turn_id": turn_id,
        "prev_bot_prompt": prev_bot_prompt,
        "is_coreference": is_coreference or infer_coref(utterance, intent),
        "shortform_present": detect_shortform(utterance),
        "stt_noise_present": stt_noise_present,
        "ui_action": ui_action,
        "scenario_group": scenario_group or f"{intent}_new_{idx:04d}",
        "split": "",
    }
    row["slots"] = build_slots_json(row)
    return row


def main() -> None:
    v1 = pd.read_csv(V1_PATH, keep_default_na=False)
    assert len(v1) == 950, f"Expected 950 baseline rows, got {len(v1)}"

    rows: List[Dict[str, object]] = []

    # 1. Upgrade all 950 existing v1 rows to 24-column schema
    for idx, r in v1.iterrows():
        utt = str(r["user_utterance"]).strip()
        intent = str(r["intent"]).strip()
        cat, ui_action = INTENT_META[intent]
        base = str(r["base_drink"]).strip()
        temp = str(r["temperature"]).strip()
        sweet = str(r["sweetness_level"]).strip()
        milk = str(r["milk_type"]).strip()

        t_low = utt.lower()
        temp_detail = ""
        ice_level = ""
        if temp == "Hot":
            if "温" in utt or re.search(r"\b(?:suam|warm)\b", t_low):
                temp_detail = "Warm"
            else:
                temp_detail = "Hot"
        elif temp == "Iced":
            if any(w in utt for w in ["去冰", "走冰", "无冰"]) or re.search(r"\b(?:kosong ais|tanpa ais|no ice|tak nak ais)\b", t_low):
                ice_level = "No Ice"
            elif any(w in utt for w in ["少冰", "微冰"]) or re.search(r"\b(?:kurang ais|less ice|sikit ais)\b", t_low):
                ice_level = "Less Ice"
            else:
                ice_level = "Normal Ice"

        row = {
            "id": str(r["id"]).strip(),
            "category": cat,
            "user_utterance": utt,
            "language_mix": infer_language_mix(utt),
            "intent": intent,
            "context": "walk_in_kiosk",
            "base_drink": base,
            "temperature": temp,
            "temp_detail": temp_detail,
            "ice_level": ice_level,
            "sweetness_level": sweet,
            "milk_type": milk,
            "old_value": "",
            "new_value": "",
            "query_type": infer_query_type(utt, intent),
            "turn_id": int(r["turn_id"]) if str(r["turn_id"]).strip() else 1,
            "prev_bot_prompt": "none",
            "is_coreference": infer_coref(utt, intent),
            "shortform_present": detect_shortform(utt),
            "stt_noise_present": "no",
            "ui_action": ui_action,
            "scenario_group": f"v1_{intent}_{idx+1:04d}",
            "split": "",
        }
        row["slots"] = build_slots_json(row)
        rows.append(row)

    next_idx = 951

    def append_batch(items):
        nonlocal next_idx
        for item in items:
            rows.append(make_row(next_idx, **item))
            next_idx += 1

    # ------------------------------------------------------------------
    # 2. order_drink (+60 => 350): Warm, Less Ice, No Ice, Cantonese, STT noise, Local combos
    # ------------------------------------------------------------------
    new_orders = [
        # Warm / Suam (10)
        {"utterance": "Bagi satu Kopi O suam kurang manis.", "base_drink": "Kopi", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Less Sugar", "milk_type": "None"},
        {"utterance": "Can I get a warm Teh C without sugar?", "base_drink": "Teh", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Zero Sugar", "milk_type": "Evaporated Milk"},
        {"utterance": "我要一杯温的白咖啡，正常糖", "base_drink": "White Coffee", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Normal", "milk_type": "Condensed Milk"},
        {"utterance": "来杯微温的美禄，少糖谢谢", "base_drink": "Milo", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk"},
        {"utterance": "Boss nak Cham suam siew dai satu.", "base_drink": "Cham", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk"},
        {"utterance": "Warm Americano no sugar please.", "base_drink": "Americano", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "一杯温柠檬茶，不要太甜", "base_drink": "Teh O Limau", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Less Sugar", "milk_type": "None"},
        {"utterance": "Nak Neslo suam biasa je.", "base_drink": "Neslo", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Normal", "milk_type": "Condensed Milk"},
        {"utterance": "Milo O suam kosong satu boss.", "base_drink": "Milo", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "Teh O suam kurang manis pls.", "base_drink": "Teh", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Less Sugar", "milk_type": "None"},
        # Less Ice / No Ice (16)
        {"utterance": "Kopi peng kurang ais siu dai satu.", "base_drink": "Kopi", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk"},
        {"utterance": "Iced Teh C less ice normal sugar please.", "base_drink": "Teh", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Normal", "milk_type": "Evaporated Milk"},
        {"utterance": "冰美式咖啡，去冰无糖谢谢", "base_drink": "Americano", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "我要冻鸳鸯少冰少甜", "base_drink": "Cham", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk"},
        {"utterance": "Teh O Limau ais tanpa ais kurang manis.", "base_drink": "Teh O Limau", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Less Sugar", "milk_type": "None"},
        {"utterance": "Milo ais kurang ais lebih manis pls.", "base_drink": "Milo", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Extra Sugar", "milk_type": "Condensed Milk"},
        {"utterance": "Boss nak White Coffee ais no ice less sugar.", "base_drink": "White Coffee", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk"},
        {"utterance": "来杯冰雀巢美禄，走冰正常糖", "base_drink": "Neslo", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Normal", "milk_type": "Condensed Milk"},
        {"utterance": "Kopi O ais kosong ais satu.", "base_drink": "Kopi", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "Teh tarik ais kurang ais sikit manis.", "base_drink": "Teh", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk"},
        {"utterance": "冻奶茶微冰半糖", "base_drink": "Teh", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk"},
        {"utterance": "冰白咖啡无冰少糖", "base_drink": "White Coffee", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk"},
        {"utterance": "Cham C ais less ice zero sugar.", "base_drink": "Cham", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Zero Sugar", "milk_type": "Evaporated Milk"},
        {"utterance": "Nk teh o ais tak nak ais krg mns", "base_drink": "Teh", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Less Sugar", "milk_type": "None"},
        {"utterance": "Iced Americano less ice no sugar please.", "base_drink": "Americano", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "冰咖啡乌少冰加糖", "base_drink": "Kopi", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Extra Sugar", "milk_type": "None"},
        # Cantonese / Hokkien / Local Kopitiam colloquial (14)
        {"utterance": "唔该一杯热茶走，正常甜", "base_drink": "Teh", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Normal", "milk_type": "Condensed Milk"},
        {"utterance": "老细，咖啡乌走糖冻饮", "base_drink": "Kopi", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "唔该冻鸳鸯少甜加淡奶", "base_drink": "Cham", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Less Sugar", "milk_type": "Evaporated Milk"},
        {"utterance": "来杯热飞砂走奶咖啡", "base_drink": "Kopi", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "Tak kiu peng siu dai satu boss.", "base_drink": "Milo", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk"},
        {"utterance": "Tut kiu panas kosong satu.", "base_drink": "Milo", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "唔该茶C冻走糖", "base_drink": "Teh", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Zero Sugar", "milk_type": "Evaporated Milk"},
        {"utterance": "来一杯冻茶乌少糖", "base_drink": "Teh", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Less Sugar", "milk_type": "None"},
        {"utterance": "老细来杯热咖啡昔少甜", "base_drink": "Kopi", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Less Sugar", "milk_type": "Evaporated Milk"},
        {"utterance": "冻柠檬茶走甜唔该", "base_drink": "Teh O Limau", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "Hot Cham O without sugar pls.", "base_drink": "Cham", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "Neslo O iced zero sugar lah.", "base_drink": "Neslo", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        {"utterance": "Give me an iced Teh O Limau normal sweetness.", "base_drink": "Teh O Limau", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Normal", "milk_type": "None"},
        {"utterance": "Kopi kosong panas satu boss.", "base_drink": "Kopi", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Zero Sugar", "milk_type": "None"},
        # STT Phonetic Noise Orders (20)
        {"utterance": "一杯咖啡瓶少甜", "base_drink": "Kopi", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
        {"utterance": "我要热美露正常糖", "base_drink": "Milo", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Normal", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
        {"utterance": "冻原样少田一杯", "base_drink": "Cham", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
        {"utterance": "冰柠檬茶走兵无糖", "base_drink": "Teh O Limau", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Zero Sugar", "milk_type": "None", "stt_noise_present": "yes"},
        {"utterance": "one copy all ice zero sugar", "base_drink": "Kopi", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Zero Sugar", "milk_type": "None", "stt_noise_present": "yes"},
        {"utterance": "boss nak teh sea peng kurang manis", "base_drink": "Teh", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Less Sugar", "milk_type": "Evaporated Milk", "stt_noise_present": "yes"},
        {"utterance": "give me one coffee bean siu dai", "base_drink": "Kopi", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
        {"utterance": "热美路加多糖", "base_drink": "Milo", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Extra Sugar", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
        {"utterance": "白咖啡冰少田", "base_drink": "White Coffee", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
        {"utterance": "远洋热少糖加炼奶", "base_drink": "Cham", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
        {"utterance": "我要美式咖啡走兵", "base_drink": "Americano", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Zero Sugar", "milk_type": "None", "stt_noise_present": "yes"},
        {"utterance": "热咖啡瓶多糖", "base_drink": "Kopi", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Extra Sugar", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
        {"utterance": "nak coffee o ping zero sugar", "base_drink": "Kopi", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Zero Sugar", "milk_type": "None", "stt_noise_present": "yes"},
        {"utterance": "can i have tea sea hot less sweet", "base_drink": "Teh", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Less Sugar", "milk_type": "Evaporated Milk", "stt_noise_present": "yes"},
        {"utterance": "美鹿冰少冰无糖", "base_drink": "Milo", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Zero Sugar", "milk_type": "None", "stt_noise_present": "yes"},
        {"utterance": "一杯咖啡病无糖去冰", "base_drink": "Kopi", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Zero Sugar", "milk_type": "None", "stt_noise_present": "yes"},
        {"utterance": "鸳阳冰正常糖", "base_drink": "Cham", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Normal", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
        {"utterance": "nestle milo iced siew dai please", "base_drink": "Neslo", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
        {"utterance": "teh o limau ice no sugar thank u", "base_drink": "Teh O Limau", "temperature": "Iced", "ice_level": "Normal Ice", "sweetness_level": "Zero Sugar", "milk_type": "None", "stt_noise_present": "yes"},
        {"utterance": "热美陆少甜谢谢", "base_drink": "Milo", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Less Sugar", "milk_type": "Condensed Milk", "stt_noise_present": "yes"},
    ]
    append_batch([dict(intent="order_drink", **d) for d in new_orders])

    # ------------------------------------------------------------------
    # 3. modify_order (+80 => 300): old->new, multi-slot, ice/warm mod, ingredient add vs cup, negations
    # ------------------------------------------------------------------
    new_modifies = [
        # Explicit old -> new (20)
        {"utterance": "Change from iced to hot please.", "old_value": "Iced", "new_value": "Hot", "temperature": "Hot", "temp_detail": "Hot"},
        {"utterance": "Tukar daripada ais kepada panas.", "old_value": "Iced", "new_value": "Hot", "temperature": "Hot", "temp_detail": "Hot"},
        {"utterance": "从冰的改成热的，谢谢", "old_value": "Iced", "new_value": "Hot", "temperature": "Hot", "temp_detail": "Hot"},
        {"utterance": "Switch from condensed milk to evaporated milk.", "old_value": "Condensed Milk", "new_value": "Evaporated Milk", "milk_type": "Evaporated Milk"},
        {"utterance": "Jangan pakai susu pekat, tukar ke susu sejat.", "old_value": "Condensed Milk", "new_value": "Evaporated Milk", "milk_type": "Evaporated Milk"},
        {"utterance": "不要炼奶了，换成淡奶", "old_value": "Condensed Milk", "new_value": "Evaporated Milk", "milk_type": "Evaporated Milk"},
        {"utterance": "不要少糖了，直接改无糖", "old_value": "Less Sugar", "new_value": "Zero Sugar", "sweetness_level": "Zero Sugar"},
        {"utterance": "Change from normal sugar to less sugar.", "old_value": "Normal", "new_value": "Less Sugar", "sweetness_level": "Less Sugar"},
        {"utterance": "Tukar manis biasa jadi kosong terus.", "old_value": "Normal", "new_value": "Zero Sugar", "sweetness_level": "Zero Sugar"},
        {"utterance": "Replace Kopi with White Coffee instead.", "old_value": "Kopi", "new_value": "White Coffee", "base_drink": "White Coffee"},
        {"utterance": "Tak jadi order Teh, tukar ke Cham.", "old_value": "Teh", "new_value": "Cham", "base_drink": "Cham"},
        {"utterance": "把刚才的美禄换成雀巢美禄", "old_value": "Milo", "new_value": "Neslo", "base_drink": "Neslo"},
        {"utterance": "Change hot to warm instead.", "old_value": "Hot", "new_value": "Warm", "temperature": "Hot", "temp_detail": "Warm"},
        {"utterance": "Jangan terlalu panas, buat suam je.", "old_value": "Hot", "new_value": "Warm", "temperature": "Hot", "temp_detail": "Warm"},
        {"utterance": "不用太烫，改温的就好", "old_value": "Hot", "new_value": "Warm", "temperature": "Hot", "temp_detail": "Warm"},
        {"utterance": "Change normal ice to less ice.", "old_value": "Normal Ice", "new_value": "Less Ice", "temperature": "Iced", "ice_level": "Less Ice"},
        {"utterance": "Tukar ais biasa ke kurang ais.", "old_value": "Normal Ice", "new_value": "Less Ice", "temperature": "Iced", "ice_level": "Less Ice"},
        {"utterance": "从正常冰改成去冰", "old_value": "Normal Ice", "new_value": "No Ice", "temperature": "Iced", "ice_level": "No Ice"},
        {"utterance": "Remove the milk and make it black.", "old_value": "Condensed Milk", "new_value": "None", "milk_type": "None"},
        {"utterance": "Dari susu cair tukar ke tanpa susu.", "old_value": "Evaporated Milk", "new_value": "None", "milk_type": "None"},
        # Multi-slot modifications (20)
        {"utterance": "Switch to Teh C iced with less sugar please.", "base_drink": "Teh", "temperature": "Iced", "sweetness_level": "Less Sugar", "milk_type": "Evaporated Milk", "new_value": "Teh C Iced Less Sugar"},
        {"utterance": "Tukar jadi Kopi O panas kosong terus.", "base_drink": "Kopi", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Zero Sugar", "milk_type": "None", "new_value": "Kopi O Hot Zero Sugar"},
        {"utterance": "改去冰的柠檬茶，还要少糖少冰", "base_drink": "Teh O Limau", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Less Sugar", "milk_type": "None", "new_value": "Teh O Limau Iced Less Ice Less Sugar"},
        {"utterance": "Actually change my drink to warm Cham less sweet.", "base_drink": "Cham", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Less Sugar", "new_value": "Cham Warm Less Sugar"},
        {"utterance": "Eh tukar ke White Coffee ais kurang ais sikit manis.", "base_drink": "White Coffee", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Less Sugar", "new_value": "White Coffee Iced Less Ice Less Sugar"},
        {"utterance": "换成热的美式咖啡，不要糖", "base_drink": "Americano", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Zero Sugar", "milk_type": "None", "new_value": "Americano Hot Zero Sugar"},
        {"utterance": "Change to Milo O iced zero sugar instead.", "base_drink": "Milo", "temperature": "Iced", "sweetness_level": "Zero Sugar", "milk_type": "None", "new_value": "Milo O Iced Zero Sugar"},
        {"utterance": "Tukar ke Neslo panas dengan kurang manis.", "base_drink": "Neslo", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Less Sugar", "new_value": "Neslo Hot Less Sugar"},
        {"utterance": "Make it iced with no ice and extra sugar.", "temperature": "Iced", "ice_level": "No Ice", "sweetness_level": "Extra Sugar", "new_value": "Iced No Ice Extra Sugar"},
        {"utterance": "改用淡奶，顺便帮我换去冰", "temperature": "Iced", "ice_level": "No Ice", "milk_type": "Evaporated Milk", "new_value": "Evaporated Milk No Ice"},
        {"utterance": "Wait, change to Kopi C suam kosong.", "base_drink": "Kopi", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Zero Sugar", "milk_type": "Evaporated Milk", "new_value": "Kopi C Warm Zero Sugar"},
        {"utterance": "Tukar ke Teh Tarik ais kurang ais lah.", "base_drink": "Teh", "temperature": "Iced", "ice_level": "Less Ice", "milk_type": "Condensed Milk", "new_value": "Teh Tarik Iced Less Ice"},
        {"utterance": "改点冰白咖啡，微冰微糖", "base_drink": "White Coffee", "temperature": "Iced", "ice_level": "Less Ice", "sweetness_level": "Less Sugar", "new_value": "White Coffee Iced Less Ice Less Sugar"},
        {"utterance": "Swap to Cham O hot without sugar please.", "base_drink": "Cham", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Zero Sugar", "milk_type": "None", "new_value": "Cham O Hot Zero Sugar"},
        {"utterance": "Tukar kpd kopi c ais sd skrg.", "base_drink": "Kopi", "temperature": "Iced", "sweetness_level": "Less Sugar", "milk_type": "Evaporated Milk", "new_value": "Kopi C Iced Less Sugar"},
        {"utterance": "改去温的鸳鸯，不要加奶", "base_drink": "Cham", "temperature": "Hot", "temp_detail": "Warm", "milk_type": "None", "new_value": "Cham Warm None"},
        {"utterance": "Change to iced Americano with no ice.", "base_drink": "Americano", "temperature": "Iced", "ice_level": "No Ice", "milk_type": "None", "new_value": "Americano Iced No Ice"},
        {"utterance": "Tukar order td jd teh o limau pns ksg.", "base_drink": "Teh O Limau", "temperature": "Hot", "temp_detail": "Hot", "sweetness_level": "Zero Sugar", "milk_type": "None", "new_value": "Teh O Limau Hot Zero Sugar"},
        {"utterance": "把冰咖啡改成温咖啡少糖", "base_drink": "Kopi", "temperature": "Hot", "temp_detail": "Warm", "sweetness_level": "Less Sugar", "new_value": "Kopi Warm Less Sugar"},
        {"utterance": "Change milk to evaporated and sugar to zero.", "sweetness_level": "Zero Sugar", "milk_type": "Evaporated Milk", "new_value": "Evaporated Milk Zero Sugar"},
        # Negations, ingredient additions (NOT multi-cup!), confirm-override modifications (20)
        {"utterance": "Bukan ais boss, nak panas.", "old_value": "Iced", "new_value": "Hot", "temperature": "Hot", "temp_detail": "Hot"},
        {"utterance": "Tak mau ais lah, tukar hot je.", "old_value": "Iced", "new_value": "Hot", "temperature": "Hot", "temp_detail": "Hot"},
        {"utterance": "不要冰了，我要热的", "old_value": "Iced", "new_value": "Hot", "temperature": "Hot", "temp_detail": "Hot"},
        {"utterance": "Tambah gula sikit, buat gah dai.", "new_value": "Extra Sugar", "sweetness_level": "Extra Sugar"},
        {"utterance": "Extra manis sikit boleh?", "new_value": "Extra Sugar", "sweetness_level": "Extra Sugar"},
        {"utterance": "Tambah susu cair je, tak nak susu pekat.", "old_value": "Condensed Milk", "new_value": "Evaporated Milk", "milk_type": "Evaporated Milk"},
        {"utterance": "Tambah kurang ais ya boss.", "new_value": "Less Ice", "temperature": "Iced", "ice_level": "Less Ice"},
        {"utterance": "Ok tapi tukar ke kurang manis dulu.", "new_value": "Less Sugar", "sweetness_level": "Less Sugar", "turn_id": 2, "prev_bot_prompt": "ask_confirm"},
        {"utterance": "Wait, sebelum bayar tukar ke ais dulu.", "new_value": "Iced", "temperature": "Iced", "turn_id": 2, "prev_bot_prompt": "ask_confirm"},
        {"utterance": "Hold on, change to zero sugar before confirming.", "new_value": "Zero Sugar", "sweetness_level": "Zero Sugar", "turn_id": 2, "prev_bot_prompt": "ask_confirm"},
        {"utterance": "等一下先别确认，改成少冰", "new_value": "Less Ice", "temperature": "Iced", "ice_level": "Less Ice", "turn_id": 2, "prev_bot_prompt": "ask_confirm"},
        {"utterance": "Tunggu jap, tukar ke kopi c.", "base_drink": "Kopi", "milk_type": "Evaporated Milk", "new_value": "Evaporated Milk", "turn_id": 2, "prev_bot_prompt": "ask_confirm"},
        {"utterance": "Actually no ice please.", "new_value": "No Ice", "temperature": "Iced", "ice_level": "No Ice"},
        {"utterance": "Kurangkan ais sikit ya.", "new_value": "Less Ice", "temperature": "Iced", "ice_level": "Less Ice"},
        {"utterance": "不要加冰块了，帮我去冰", "new_value": "No Ice", "temperature": "Iced", "ice_level": "No Ice"},
        {"utterance": "Don't put any sugar, change to kosong.", "new_value": "Zero Sugar", "sweetness_level": "Zero Sugar"},
        {"utterance": "Tak nak susu pekat, buang susu.", "old_value": "Condensed Milk", "new_value": "None", "milk_type": "None"},
        {"utterance": "别放糖了，改走糖", "new_value": "Zero Sugar", "sweetness_level": "Zero Sugar"},
        {"utterance": "Switch to warm instead of hot.", "old_value": "Hot", "new_value": "Warm", "temperature": "Hot", "temp_detail": "Warm"},
        {"utterance": "Eh bukan Milo, saya cakap Neslo tadi.", "old_value": "Milo", "new_value": "Neslo", "base_drink": "Neslo"},
        # Shortform & STT noise modifications (20)
        {"utterance": "tkr pns ke suam pls", "old_value": "Hot", "new_value": "Warm", "temperature": "Hot", "temp_detail": "Warm"},
        {"utterance": "xnak ais bnyk, buat krg ais", "new_value": "Less Ice", "temperature": "Iced", "ice_level": "Less Ice"},
        {"utterance": "tkr jd tanpa ais pls", "new_value": "No Ice", "temperature": "Iced", "ice_level": "No Ice"},
        {"utterance": "eh tkr kp ke wc tq", "old_value": "Kopi", "new_value": "White Coffee", "base_drink": "White Coffee"},
        {"utterance": "xmo ssu pekat, tkr o je", "old_value": "Condensed Milk", "new_value": "None", "milk_type": "None"},
        {"utterance": "blh tkr jd amer sjk x?", "new_value": "Americano Iced", "base_drink": "Americano", "temperature": "Iced", "milk_type": "None"},
        {"utterance": "gula tkr ksg terus pls", "new_value": "Zero Sugar", "sweetness_level": "Zero Sugar"},
        {"utterance": "tkr krg mns sbb takut manis sgt", "new_value": "Less Sugar", "sweetness_level": "Less Sugar"},
        {"utterance": "xjadi cham, tkr teh tarik pls", "old_value": "Cham", "new_value": "Teh", "base_drink": "Teh", "milk_type": "Condensed Milk"},
        {"utterance": "tkr ssu ke ssu sejat tq", "new_value": "Evaporated Milk", "milk_type": "Evaporated Milk"},
        {"utterance": "不要咖啡瓶，我要热的", "old_value": "Iced", "new_value": "Hot", "temperature": "Hot", "temp_detail": "Hot", "stt_noise_present": "yes"},
        {"utterance": "改成走兵可以吗", "new_value": "No Ice", "temperature": "Iced", "ice_level": "No Ice", "stt_noise_present": "yes"},
        {"utterance": "换少田啦，不要太甜", "new_value": "Less Sugar", "sweetness_level": "Less Sugar", "stt_noise_present": "yes"},
        {"utterance": "不是美露，换去原样", "old_value": "Milo", "new_value": "Cham", "base_drink": "Cham", "stt_noise_present": "yes"},
        {"utterance": "change to copy all instead", "new_value": "Kopi O", "base_drink": "Kopi", "milk_type": "None", "stt_noise_present": "yes"},
        {"utterance": "switch to tea sea less sweet", "new_value": "Teh C Less Sugar", "base_drink": "Teh", "milk_type": "Evaporated Milk", "sweetness_level": "Less Sugar", "stt_noise_present": "yes"},
        {"utterance": "唔该改做冻饮少甜", "new_value": "Iced Less Sugar", "temperature": "Iced", "sweetness_level": "Less Sugar"},
        {"utterance": "老细唔好加奶，转斋啡", "new_value": "None", "milk_type": "None"},
        {"utterance": "tkr jd gd lah xpe", "new_value": "Extra Sugar", "sweetness_level": "Extra Sugar"},
        {"utterance": "换去无糖的，谢谢boss", "new_value": "Zero Sugar", "sweetness_level": "Zero Sugar"},
    ]
    append_batch([dict(intent="modify_order", **d) for d in new_modifies])

    # ------------------------------------------------------------------
    # 4. clarify_order (+80 => 300): Multi-turn fragments + domain Q&A + partial orders
    # ------------------------------------------------------------------
    new_clarifies = [
        # Fragment replies to ask_temperature (14)
        {"utterance": "ais", "temperature": "Iced", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "panas", "temperature": "Hot", "temp_detail": "Hot", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "suam", "temperature": "Hot", "temp_detail": "Warm", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "warm please", "temperature": "Hot", "temp_detail": "Warm", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "iced lah", "temperature": "Iced", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "kurang ais", "temperature": "Iced", "ice_level": "Less Ice", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "tanpa ais", "temperature": "Iced", "ice_level": "No Ice", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "no ice", "temperature": "Iced", "ice_level": "No Ice", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "少冰", "temperature": "Iced", "ice_level": "Less Ice", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "去冰", "temperature": "Iced", "ice_level": "No Ice", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "热的", "temperature": "Hot", "temp_detail": "Hot", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "温的就好", "temperature": "Hot", "temp_detail": "Warm", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "peng", "temperature": "Iced", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        {"utterance": "hot je", "temperature": "Hot", "temp_detail": "Hot", "turn_id": 2, "prev_bot_prompt": "ask_temperature"},
        # Fragment replies to ask_sweetness (14)
        {"utterance": "kurang manis", "sweetness_level": "Less Sugar", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "krg mns", "sweetness_level": "Less Sugar", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "siu dai", "sweetness_level": "Less Sugar", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "sd", "sweetness_level": "Less Sugar", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "kosong", "sweetness_level": "Zero Sugar", "milk_type": "None", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "ksg", "sweetness_level": "Zero Sugar", "milk_type": "None", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "no sugar", "sweetness_level": "Zero Sugar", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "xnak gla", "sweetness_level": "Zero Sugar", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "normal sugar", "sweetness_level": "Normal", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "biasa je", "sweetness_level": "Normal", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "gah dai", "sweetness_level": "Extra Sugar", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "少糖", "sweetness_level": "Less Sugar", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "无糖", "sweetness_level": "Zero Sugar", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        {"utterance": "正常糖", "sweetness_level": "Normal", "turn_id": 2, "prev_bot_prompt": "ask_sweetness"},
        # Fragment replies to ask_milk (12)
        {"utterance": "o", "milk_type": "None", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "c", "milk_type": "Evaporated Milk", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "si", "milk_type": "Evaporated Milk", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "tanpa susu", "milk_type": "None", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "no milk", "milk_type": "None", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "susu cair", "milk_type": "Evaporated Milk", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "susu pekat", "milk_type": "Condensed Milk", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "evaporated milk", "milk_type": "Evaporated Milk", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "condensed milk", "milk_type": "Condensed Milk", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "淡奶", "milk_type": "Evaporated Milk", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "炼奶", "milk_type": "Condensed Milk", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        {"utterance": "不加奶", "milk_type": "None", "turn_id": 2, "prev_bot_prompt": "ask_milk"},
        # Domain exploratory questions, Kopitiam terms, temperature/ice capability questions (40)
        {"utterance": "What is the difference between Kopi O and Kopi C?"},
        {"utterance": "Apa maksud siu dai dan kosong?"},
        {"utterance": "茶C和普通奶茶有什么区别？"},
        {"utterance": "Can I adjust the ice level to less ice?"},
        {"utterance": "温度可以调成温的吗？"},
        {"utterance": "Boleh buat tanpa ais tak kat mesin ni?"},
        {"utterance": "What drinks do you recommend for first time?"},
        {"utterance": "Minuman apa paling popular kat sini?"},
        {"utterance": "你们这里有什么招牌咖啡推荐？"},
        {"utterance": "Do you have non-coffee options?"},
        {"utterance": "Ada minuman selain kopi tak?"},
        {"utterance": "不喝咖啡的话有什么选择？"},
        {"utterance": "Is White Coffee sweet by default?"},
        {"utterance": "Neslo tu campur kopi dengan milo ke?"},
        {"utterance": "鸳鸯是什么味道的？"},
        {"utterance": "Does Americano come with milk?"},
        {"utterance": "Teh O Limau ada letak susu ke?"},
        {"utterance": "Kopi kosong tu maksudnya tiada gula dan tiada susu ke?"},
        {"utterance": "为什么选无糖咖啡就会自动变成 Kopi O？"},
        {"utterance": "Can I order Milo without condensed milk?"},
        {"utterance": "Which drink is less bitter?"},
        {"utterance": "Mana satu lagi manis, White Coffee atau Milo?"},
        {"utterance": "哪一款喝起来比较不苦？"},
        {"utterance": "Boleh pilih suhu suam untuk Teh Tarik?"},
        {"utterance": "有没有加淡奶的咖啡选项？"},
        {"utterance": "Got any refreshing cold drink for hot weather?", "temperature": "Iced"},
        {"utterance": "Ada air sejuk yang menyegarkan tak?", "temperature": "Iced"},
        {"utterance": "天气好热，有什么冰饮推荐？", "temperature": "Iced"},
        {"utterance": "Can I customize the sweetness level?"},
        {"utterance": "甜度可以选半糖或者无糖吗？"},
        {"utterance": "Kopi ni pakai susu pekat ke susu sejat?"},
        {"utterance": "What is Cham made of?"},
        {"utterance": "Milo Dinosaur ada jual ke kat mesin ni?"},
        {"utterance": "Can you suggest something warm for a rainy day?", "temperature": "Hot", "temp_detail": "Warm"},
        {"utterance": "tak tau nak minum apa lah, suggest sikit"},
        {"utterance": "ada jual kopi c x?"},
        {"utterance": "blh buat suam utk wc x?"},
        {"utterance": "neslo ngn milo mana lbh sedap?"},
        {"utterance": "请问有没有去冰的选项？", "temperature": "Iced", "ice_level": "No Ice"},
        {"utterance": "第一次喝本地咖啡，推荐哪一杯？"},
    ]
    append_batch([dict(intent="clarify_order", **d) for d in new_clarifies])

    # ------------------------------------------------------------------
    # 5. health_query (+60 => 280): coreference, allergens, caffeine, comparisons, shortforms
    # ------------------------------------------------------------------
    new_healths = [
        # Coreference queries (15)
        {"utterance": "这杯的卡路里高吗？", "is_coreference": "yes", "query_type": "calories"},
        {"utterance": "这杯含糖量多少克？", "is_coreference": "yes", "query_type": "sugar"},
        {"utterance": "这杯有乳糖吗？", "is_coreference": "yes", "query_type": "lactose"},
        {"utterance": "这杯含不含麸质？", "is_coreference": "yes", "query_type": "gluten"},
        {"utterance": "Is this cup high in calories?", "is_coreference": "yes", "query_type": "calories"},
        {"utterance": "Does this drink contain dairy or lactose?", "is_coreference": "yes", "query_type": "lactose"},
        {"utterance": "How much sugar is in my current drink?", "is_coreference": "yes", "query_type": "sugar"},
        {"utterance": "Will this drink make me gain weight?", "is_coreference": "yes", "query_type": "calories"},
        {"utterance": "Air ni tinggi kalori ke?", "is_coreference": "yes", "query_type": "calories"},
        {"utterance": "Minuman ni ada laktosa tak?", "is_coreference": "yes", "query_type": "lactose"},
        {"utterance": "Cawan ni ada gluten ke boss?", "is_coreference": "yes", "query_type": "gluten"},
        {"utterance": "Berapa gram gula dalam air ni?", "is_coreference": "yes", "query_type": "sugar"},
        {"utterance": "这杯会发胖吗？", "is_coreference": "yes", "query_type": "calories"},
        {"utterance": "Does this one have caffeine?", "is_coreference": "yes", "query_type": "caffeine"},
        {"utterance": "air td tinggi gla ke?", "is_coreference": "yes", "query_type": "sugar"},
        # Allergen & Dietary recommendations (18)
        {"utterance": "I am lactose intolerant, what can I drink here?", "query_type": "allergen_rec"},
        {"utterance": "Saya alahan laktosa, minuman mana selamat?", "query_type": "allergen_rec"},
        {"utterance": "我有乳糖不耐症，可以喝哪些饮品？", "query_type": "allergen_rec"},
        {"utterance": "Any gluten-free drinks on the menu?", "query_type": "allergen_rec"},
        {"utterance": "Ada minuman bebas gluten tak?", "query_type": "allergen_rec"},
        {"utterance": "对麸质过敏的人能喝什么？", "query_type": "allergen_rec"},
        {"utterance": "Does Milo contain barley malt or gluten?", "base_drink": "Milo", "query_type": "gluten"},
        {"utterance": "Neslo ada gluten ke?", "base_drink": "Neslo", "query_type": "gluten"},
        {"utterance": "Kopi O Kosong ada laktosa atau gluten tak?", "base_drink": "Kopi", "sweetness_level": "Zero Sugar", "milk_type": "None", "query_type": "lactose"},
        {"utterance": "美式咖啡是零乳糖和无麸质的吗？", "base_drink": "Americano", "milk_type": "None", "query_type": "lactose"},
        {"utterance": "Is Teh O Limau dairy free?", "base_drink": "Teh O Limau", "milk_type": "None", "query_type": "lactose"},
        {"utterance": "Which drinks contain condensed milk allergens?", "query_type": "allergen_rec"},
        {"utterance": "sy alergi susu lembu, blh mnm apa?", "query_type": "allergen_rec"},
        {"utterance": "乳糜泻患者可以点美禄或雀巢美禄吗？", "query_type": "gluten"},
        {"utterance": "What is safe for diabetics?", "query_type": "low_cal_rec"},
        {"utterance": "糖尿病可以喝无糖茶乌吗？", "base_drink": "Teh", "sweetness_level": "Zero Sugar", "milk_type": "None", "query_type": "sugar"},
        {"utterance": "Which drink has zero sugar and zero milk?", "query_type": "low_cal_rec"},
        {"utterance": "对牛奶过敏可以点哪几款咖啡？", "query_type": "allergen_rec"},
        # Caffeine, comparisons, calorie tiers (27)
        {"utterance": "How many mg of caffeine in Americano?", "base_drink": "Americano", "milk_type": "None", "query_type": "caffeine"},
        {"utterance": "Kopi dengan Teh mana lebih tinggi kafein?", "query_type": "comparison"},
        {"utterance": "美禄和柠檬茶哪个热量更低？", "query_type": "comparison"},
        {"utterance": "Compare calories of White Coffee vs Kopi O Kosong.", "query_type": "comparison"},
        {"utterance": "Milo Dinosaur berapa kalori?", "base_drink": "Milo", "query_type": "calories"},
        {"utterance": "Milo Dinosaur 会胖吗？", "base_drink": "Milo", "query_type": "calories"},
        {"utterance": "Is Teh O Ais Limau Less Sugar high in calories?", "base_drink": "Teh O Limau", "temperature": "Iced", "sweetness_level": "Less Sugar", "milk_type": "None", "query_type": "calories"},
        {"utterance": "Cham Peng 热量高不高？", "base_drink": "Cham", "temperature": "Iced", "query_type": "calories"},
        {"utterance": "How much sugar is in Kopi C Siu Dai?", "base_drink": "Kopi", "sweetness_level": "Less Sugar", "milk_type": "Evaporated Milk", "query_type": "sugar"},
        {"utterance": "白咖啡含糖量是不是很高？", "base_drink": "White Coffee", "query_type": "sugar"},
        {"utterance": "Which drink is under 50 kcal?", "query_type": "low_cal_rec"},
        {"utterance": "Ada minuman bawah 100 kalori tak?", "query_type": "low_cal_rec"},
        {"utterance": "减脂期适合喝美式咖啡还是咖啡乌无糖？", "query_type": "comparison"},
        {"utterance": "Does Milo have caffeine?", "base_drink": "Milo", "query_type": "caffeine"},
        {"utterance": "Teh O Limau ada kafein ke?", "base_drink": "Teh O Limau", "milk_type": "None", "query_type": "caffeine"},
        {"utterance": "晚上喝白咖啡会不会睡不着？咖啡因高吗？", "base_drink": "White Coffee", "query_type": "caffeine"},
        {"utterance": "Berapa kalori Teh Tarik biasa?", "base_drink": "Teh", "sweetness_level": "Normal", "milk_type": "Condensed Milk", "query_type": "calories"},
        {"utterance": "Check nutrition facts for Neslo Peng.", "base_drink": "Neslo", "temperature": "Iced", "query_type": "nutrition_general"},
        {"utterance": "Info khasiat Kopi O Panas.", "base_drink": "Kopi", "temperature": "Hot", "milk_type": "None", "query_type": "nutrition_general"},
        {"utterance": "我想看热鸳鸯的营养成分和糖分", "base_drink": "Cham", "temperature": "Hot", "query_type": "nutrition_general"},
        {"utterance": "kp c sd vs teh o ksg mana kurang gla?", "query_type": "comparison"},
        {"utterance": "wc pns ada brp g gla?", "base_drink": "White Coffee", "temperature": "Hot", "query_type": "sugar"},
        {"utterance": "milo o ksg brp kcal?", "base_drink": "Milo", "sweetness_level": "Zero Sugar", "milk_type": "None", "query_type": "calories"},
        {"utterance": "amer pns ada kafein tinggi x?", "base_drink": "Americano", "temperature": "Hot", "milk_type": "None", "query_type": "caffeine"},
        {"utterance": "cham c sd tinggi laktosa x?", "base_drink": "Cham", "sweetness_level": "Less Sugar", "milk_type": "Evaporated Milk", "query_type": "lactose"},
        {"utterance": "nk tny kalori kopi o peng", "base_drink": "Kopi", "temperature": "Iced", "milk_type": "None", "query_type": "calories"},
        {"utterance": "neslo pns ada gluten dan laktosa x?", "base_drink": "Neslo", "temperature": "Hot", "query_type": "gluten"},
    ]
    append_batch([dict(intent="health_query", **d) for d in new_healths])

    # ------------------------------------------------------------------
    # 6. price_query (100 rows)
    # ------------------------------------------------------------------
    price_templates = [
        ("How much is a Kopi O?", "Kopi", "", "None"),
        ("How much does iced White Coffee cost?", "White Coffee", "Iced", ""),
        ("What is the price of Hot Americano?", "Americano", "Hot", "None"),
        ("How much is Teh C Peng?", "Teh", "Iced", "Evaporated Milk"),
        ("How much for Cham Panas?", "Cham", "Hot", ""),
        ("What's the price for Milo Ais?", "Milo", "Iced", ""),
        ("How much is Neslo Hot?", "Neslo", "Hot", ""),
        ("How much does Teh O Ais Limau cost?", "Teh O Limau", "Iced", "None"),
        ("What is the price range of drinks here?", "", "", ""),
        ("How much extra for iced drinks?", "", "Iced", ""),
        ("Does evaporated milk cost more?", "", "", "Evaporated Milk"),
        ("How much is this cup?", "", "", ""),
        ("What is the total price of my order?", "", "", ""),
        ("Are all drinks under RM 6?", "", "", ""),
        ("How much for Milo O Hot?", "Milo", "Hot", "None"),
        ("Berapa harga Kopi O Panas?", "Kopi", "Hot", "None"),
        ("Kopi C Peng berapa ringgit?", "Kopi", "Iced", "Evaporated Milk"),
        ("Berapa harga Teh Tarik?", "Teh", "", "Condensed Milk"),
        ("Cham ais berapa?", "Cham", "Iced", ""),
        ("White Coffee panas berapa harga ya?", "White Coffee", "Hot", ""),
        ("Berapa ringgit untuk Milo ais?", "Milo", "Iced", ""),
        ("Neslo panas harga berapa?", "Neslo", "Hot", ""),
        ("Teh O Limau ais berapa sen?", "Teh O Limau", "Iced", "None"),
        ("Americano berapa ringgit kat sini?", "Americano", "", "None"),
        ("Harga minuman kat mesin ni dari berapa ke berapa?", "", "", ""),
        ("Kalau tambah ais kena tambah berapa?", "", "Iced", ""),
        ("Pakai susu cair lagi mahal ke?", "", "", "Evaporated Milk"),
        ("Air ni berapa harga dia?", "", "", ""),
        ("Jumlah pesanan saya berapa sekarang?", "", "", ""),
        ("Kopi biasa dengan Kopi O sama harga ke?", "Kopi", "", ""),
        ("咖啡乌多少钱？", "Kopi", "", "None"),
        ("冰白咖啡一杯几块钱？", "White Coffee", "Iced", ""),
        ("热美式咖啡什么价钱？", "Americano", "Hot", "None"),
        ("冻茶C多少钱一杯？", "Teh", "Iced", "Evaporated Milk"),
        ("热鸳鸯卖多少钱？", "Cham", "Hot", ""),
        ("冰美禄多少钱？", "Milo", "Iced", ""),
        ("雀巢美禄热的价格是多少？", "Neslo", "Hot", ""),
        ("冻柠檬茶几块？", "Teh O Limau", "Iced", "None"),
        ("这里的饮料大概多少钱？", "", "", ""),
        ("加冰块需要另加钱吗？", "", "Iced", ""),
        ("换淡奶会更贵吗？", "", "", "Evaporated Milk"),
        ("这杯要多少钱？", "", "", ""),
        ("我目前订单一共多少钱？", "", "", ""),
        ("热咖啡和冰咖啡差价多少？", "Kopi", "", ""),
        ("拉茶一杯多少钱？", "Teh", "", "Condensed Milk"),
        ("Boss kopi o peng brp?", "Kopi", "Iced", "None"),
        ("teh c pns brp RM?", "Teh", "Hot", "Evaporated Milk"),
        ("wc ais hrga brp ya?", "White Coffee", "Iced", ""),
        ("amer ais brp ringgit?", "Americano", "Iced", "None"),
        ("milo o pns hrga berapa?", "Milo", "Hot", "None"),
        ("cham ais brp?", "Cham", "Iced", ""),
        ("neslo peng brp RM?", "Neslo", "Iced", ""),
        ("teh o limau pns brp?", "Teh O Limau", "Hot", "None"),
        ("brp harga semua air kt sini?", "", "", ""),
        ("hrga kopi c dgn kopi biasa beza brp?", "Kopi", "", ""),
        ("brp total order sy skrg?", "", "", ""),
        ("air ni brp RM boss?", "", "", ""),
        ("kopi o ksg brp duit?", "Kopi", "", "None"),
        ("nk tau hrga white coffee peng", "White Coffee", "Iced", ""),
        ("blh check hrga cham c ais?", "Cham", "Iced", "Evaporated Milk"),
        ("这杯kp o ais多少钱？", "Kopi", "Iced", "None"),
        ("请问wc冰brp RM?", "White Coffee", "Iced", ""),
        ("amer热how much ah?", "Americano", "Hot", "None"),
        ("teh c冰多少块？", "Teh", "Iced", "Evaporated Milk"),
        ("milo ais price brp?", "Milo", "Iced", ""),
        ("Kopi O price how much boss?", "Kopi", "", "None"),
        ("White Coffee peng harga berapa ah?", "White Coffee", "Iced", ""),
        ("How much for 1 cup of Cham C?", "Cham", "", "Evaporated Milk"),
        ("Teh Tarik panas cost how much?", "Teh", "Hot", "Condensed Milk"),
        ("Neslo priceberapa?", "Neslo", "", ""),
        ("Check price for Teh O Limau please.", "Teh O Limau", "", "None"),
        ("What is the price of Kopi C Kosong?", "Kopi", "", "Evaporated Milk"),
        ("Berapa harga untuk White Coffee Less Sugar?", "White Coffee", "", ""),
        ("冰美式几钱？", "Americano", "Iced", "None"),
        ("热茶乌少糖多少钱？", "Teh", "Hot", "None"),
        ("Is Teh O cheaper than Teh Tarik?", "Teh", "", ""),
        ("Kopi O Kosong lagi murah ke?", "Kopi", "", "None"),
        ("Milo O RM berapa?", "Milo", "", "None"),
        ("How much do I need to pay for this drink?", "", "", ""),
        ("What's the unit price of Cham Peng?", "Cham", "Iced", ""),
        ("Berapa kos secawan Kopi Peng?", "Kopi", "Iced", ""),
        ("这杯需要付多少钱？", "", "", ""),
        ("请问最便宜的饮料多少钱？", "", "", ""),
        ("What is your cheapest drink?", "", "", ""),
        ("Minuman paling murah berapa ringgit?", "", "", ""),
        ("most expensive drink brp RM?", "", "", ""),
        ("最贵的那杯饮料多少钱？", "", "", ""),
        ("Does Zero Sugar cost the same?", "", "", ""),
        ("无糖和正常糖价格一样吗？", "", "", ""),
        ("Kurang manis harga sama ke?", "", "", ""),
        ("Tell me the price of Kopi C Panas.", "Kopi", "Hot", "Evaporated Milk"),
        ("Nak tanya harga Teh O Peng.", "Teh", "Iced", "None"),
        ("帮我查一下冰雀巢美禄的价格", "Neslo", "Iced", ""),
        ("How much is Kopi O Warm?", "Kopi", "Hot", "None"),
        ("White Coffee suam berapa ringgit?", "White Coffee", "Hot", ""),
        ("少冰柠檬茶多少钱？", "Teh O Limau", "Iced", "None"),
        ("Cham O ais price?", "Cham", "Iced", "None"),
        ("teh o ksg brp sen?", "Teh", "", "None"),
        ("brp kena bayar utk order ni?", "", "", ""),
        ("请问这杯amer要几多钱？", "Americano", "", "None"),
    ]
    append_batch([
        {
            "intent": "price_query",
            "utterance": u,
            "base_drink": b,
            "temperature": t,
            "milk_type": m,
            "query_type": "price",
        }
        for u, b, t, m in price_templates
    ])

    # ------------------------------------------------------------------
    # 7. order_confirm (90 rows)
    # ------------------------------------------------------------------
    confirm_utterances = [
        "Confirm my order", "Confirm order please", "Yes, confirm it", "Okay, place my order",
        "Go ahead and brew it", "Sure, proceed with this order", "Looks good, confirm!",
        "Alright, make it now", "Yes, that's everything, confirm", "Proceed to brew please",
        "Confirmed, let's pay", "Yes please, proceed", "Ok done, confirm order",
        "All correct, place order", "Hurry, confirm my drink now", "Confirm fast, need to catch LRT",
        "Yep, confirm this cup", "Looks right, go ahead", "Yes, brew this coffee now",
        "Perfect, confirm my selection", "Please finalize my order", "Ready, confirm order",
        "Sure, make this drink", "Okay proceed", "Yes, I confirm",
        "Sahkan pesanan saya", "Ya, sahkan sekarang", "Ok, teruskan pesanan",
        "Betul, buat sekarang", "Boleh, sahkan order ini", "Ngam, terus buat lah",
        "Ok confirm order boss", "Ya betul, teruskan bayar", "Semua betul, sahkan ya",
        "Cepat sikit, sahkan order sekarang", "Siap, boleh bancuh sekarang", "Jadi, sahkan yang ini",
        "Ok boleh, proceed je", "Betul semua, hantar pesanan", "Baik, buat air ini sekarang",
        "Ya saya sahkan pesanan ini", "Terus bancuh je boss", "Ok ngam, bayar dan buat",
        "Dah cukup, sahkan order", "Sahkan terus lah",
        "确认我的订单", "确认下单", "好的，现在帮我做吧", "没问题，确认这杯",
        "对的，开始冲泡吧", "就点这杯，确认", "可以了，请下单", "全部正确，确认订单",
        "快点确认，我赶时间", "好，确认付款下单", "行，就按这个做", "确定要这杯了",
        "是的，请立刻制作", "确认无误，开始吧", "好啦，下单谢谢",
        "Ok confirm lah boss", "Ya, confirm order ni pls", "Ngam lah, confirm terus",
        "Confirm 订单 please", "好，confirm order skrg", "Cepat confirm, nk catch bus",
        "Betul boss, proceed buat air", "Ok done, 确认下单 tq", "Sahkan je, 没问题",
        "Yes ngam, make it now", "Boleh, 确认这杯 pls", "Confirm skrg tq",
        "Ok 确认!", "Ya boleh, brew skrg", "Proceed lah boss",
        "cnfrm order pls", "ok byr skrg, confirm", "confirm je xpayah tunggu",
        "好嘞，confirm!", "没错，就是这杯，做吧", "确定下单咯",
        "Ok, lock this order", "Yes, send it to the machine", "Teruskan je pesanan ni",
        "Sila proses pesanan saya", "请帮我出杯吧，已确认", "OK boss, go ahead!",
        "Confirm this one, thanks!", "Beres, buat terus!", "就这样，确认下单！",
    ]
    append_batch([{"intent": "order_confirm", "utterance": u, "turn_id": 2, "prev_bot_prompt": "ask_confirm"} for u in confirm_utterances])

    # ------------------------------------------------------------------
    # 8. cancel_order (90 rows)
    # ------------------------------------------------------------------
    cancel_utterances = [
        "Cancel my order", "Cancel the current order please", "Reset everything", "Clear my cart",
        "Start over from scratch", "Never mind, cancel it", "Abort this order", "Discard my selection",
        "I don't want to order anymore", "Please void this order", "Drop the order", "Cancel all",
        "Reset the kiosk please", "Stop the order", "Erase my current drink", "No thanks, cancel it",
        "Forget it, clear the order", "Take me out, cancel everything", "I changed my mind, cancel order",
        "Remove everything from cart", "Restart my order", "Delete current selection",
        "Cancel this cup please", "Void my current selection", "Reset to blank",
        "Batalkan pesanan saya", "Batal order ini", "Reset semula semua", "Kosongkan pesanan",
        "Tak jadi order lah", "Saya tak nak beli lagi", "Padam semua pilihan", "Mula semula dari awal",
        "Batalkan semua sekarang", "Tak jadi minum, batal je", "Hapus pesanan semasa", "Berhenti, jangan buat lagi",
        "Cancel order saya sekarang", "Tutup pesanan ni", "Jangan teruskan, batal",
        "Saya nak batalkan minuman ini", "Ulang semula pesanan", "Kosongkan troli sekarang",
        "Tak perlu buat, cancel terus", "Batalkan semuanya",
        "取消我当前的订单", "取消订单", "全部重置", "清空购物车",
        "我不要点了，帮我取消", "算了，取消吧", "重新开始点单", "把刚才选的都删掉",
        "停止下单，取消全部", "不好意思，我不买了", "重置机器界面", "帮我作废这个订单",
        "先取消掉，等下再点", "不要做了，取消谢谢", "全部清零",
        "Cancel order pls", "Batal order ni tq", "xjadi order lah, cancel",
        "Cancel 当前订单 please", "Reset balik semua pls", "不要了，cancel pls",
        "xnk order dh, batal", "cncl order skrg", "cncl je boss",
        "Tak jadi lah, reset semula", "Maaf cancel order ya", "清掉这个order",
        "xmo beli dh, clear cart", "Cancel semua, start over", "Jangan buat, 取消订单",
        "Reset order sy pls", "Abort lah, tak sempat tunggu", "xjadi beli, batal semua",
        "Drop order ni please", "Clear semuanya tq", "取消算了，赶时间",
        "不喝了，帮我取消订单", "Please cancel before brewing", "Tolong batalkan sebelum bayar",
        "No need already, cancel!", "Tak payah buat dah, batal!", "重新来过，清空",
        "Cancel my drink selection", "Batalkan pesanan yang tadi", "Stop it, clear everything",
    ]
    append_batch([{"intent": "cancel_order", "utterance": u} for u in cancel_utterances])

    # ------------------------------------------------------------------
    # 9. single_cup_limit (90 rows)
    # ------------------------------------------------------------------
    multi_cup_utterances = [
        "Give me 2 cups of Kopi O", "I want two iced Americanos", "Order 3 cups of Teh Tarik please",
        "Can I get two White Coffees?", "Add another cup of Milo", "Give me one more drink",
        "Make it 2 cups instead", "I'd like three hot coffees", "Can you brew two drinks at once?",
        "Add one more Kopi C to my order", "Two cups of Cham Peng please", "Give me 3 Neslo iced",
        "Let me order two drinks together", "I want an extra cup of Teh O Limau",
        "Please prepare 2 hot White Coffees", "Can I buy 3 drinks in one go?",
        "Add another coffee for my friend", "Make two cups of Kopi Peng less sugar",
        "I need 2 cups of tea without sugar", "Order 2 iced Milo please",
        "Satu lagi cawan Kopi O boleh?", "Bagi saya dua cawan Teh Tarik", "Nak order 2 cawan sekaligus",
        "Tambah lagi satu air boleh?", "Saya nak tiga cawan White Coffee", "Buat dua cawan Kopi Peng",
        "Nak beli 2 cawan untuk kawan", "Tambah satu lagi cawan Milo ais", "Bagi 3 gelas Teh O Limau",
        "Boleh buat dua cawan sekaligus tak?", "Nak tambah satu minuman lagi", "Saya pesan dua kopi panas",
        "Tolong siapkan 2 cawan Cham ais", "Nak dua cawan Americano tanpa gula", "Tambah cawan kedua",
        "Lagi satu cawan Neslo panas", "Order dua gelas serentak boleh?", "Bagi 2 kopi c peng",
        "Nak tambah air kedua dalam troli", "Buat 3 cawan sekali gus",
        "我要两杯咖啡", "请给我两杯冰美式", "帮我点三杯拉茶", "再来一杯白咖啡",
        "可以一次点两杯吗？", "加多一杯冻鸳鸯", "我要两杯热美禄少糖", "给我三杯冻柠檬茶",
        "再加一杯咖啡乌无糖", "帮我一起做两杯咖啡", "我要买两杯，帮我一起点", "再来一碗不，再来一杯茶C",
        "可以同时出两杯饮料吗？", "我朋友也要一杯，点两杯", "加多一杯热雀巢美禄",
        "一共要三杯冰咖啡", "再点多一杯行吗？", "给我两杯冰白咖啡",
        "一次点两杯可以吗？", "我想点2杯茶乌",
        "Boss bagi 2 cups kopi o peng", "nk 2 cawan wc ais pls", "tambah lagi satu cup lah",
        "2 cups teh c sd tq", "blh order 2 drinks skrg x?", "我要2杯kopi ais",
        "加多1杯teh tarik pls", "nk 3 cawan milo peng boss", "Add 1 more cup kopi c",
        "satu lagi cup utk kawan sy", "2杯wc热，一起做pls", "boleh buat 2 cups sekaligus?",
        "nk order dua air skali gus", "extra cup kopi o pls", "我要两杯americano冰",
        "tambah satu drink lagi tq", "give me 2 teh o limau ais", "dua gelas cham peng pls",
        "3 cups kopi panas please", "bgi 2 kp c pns", "nk tambah cawan kedua",
        "1 cup kopi and 1 cup teh", "我要一杯咖啡和一杯奶茶", "Satu Kopi Peng dan satu Teh Ais",
        "Bagi Kopi O satu dengan Milo satu", "Order White Coffee plus one Americano",
        "Can I get both Kopi and Teh together?", "两杯不同口味的可以一起点吗？",
        "Nak satu Cham dan satu Neslo", "One hot Milo and one iced Kopi please",
    ]
    append_batch([{"intent": "single_cup_limit", "utterance": u} for u in multi_cup_utterances])

    # ------------------------------------------------------------------
    # 10. payment_query (90 rows)
    # ------------------------------------------------------------------
    payment_utterances = [
        "Can I pay using QR code?", "Do you accept Touch n Go eWallet?", "Can I pay with GrabPay?",
        "Does this machine accept cash?", "Can I use a credit card or debit card?",
        "Do you support DuitNow QR?", "How do I pay at this kiosk?", "Is Apple Pay supported?",
        "Can I pay by Boost or ShopeePay?", "Does the kiosk give change for cash?",
        "Do you accept RM 50 notes?", "Can I get a payment receipt?", "Is contactless card payment working?",
        "Which payment methods are accepted?", "Can I pay using MyDebit ATM card?",
        "Do you take coins?", "When do I scan the QR code?", "Is PayWave accepted here?",
        "What if my QR payment fails?", "Can I split payment?", "Do you charge SST or service tax?",
        "Is there a card surcharge?", "Can I pay after the drink is ready?",
        "Do you support Mae QR?", "Where do I tap my card?",
        "Boleh bayar guna kod QR tak?", "Terima Touch n Go eWallet ke?", "Boleh pakai GrabPay kat sini?",
        "Mesin ni terima duit tunai tak?", "Boleh guna kad kredit atau debit?",
        "Sokong DuitNow QR tak?", "Macam mana nak bayar kat mesin ni?", "Boleh bayar pakai PayWave?",
        "Mesin ni bagi duit baki ke?", "Terima duit kertas RM 50 tak?", "Ada keluar resit bayaran?",
        "Cara pembayaran apa yang diterima?", "Boleh bayar guna Boost atau ShopeePay?",
        "Boleh pakai syiling tak?", "Bila saya kena imbas kod QR?", "Kalau bayaran QR gagal macam mana?",
        "Ada caj tambahan kalau guna kad?", "Boleh bayar lepas air siap?",
        "Kat mana nak tap kad bank?", "Terima kad ATM MyDebit tak?",
        "可以用二维码付款吗？", "支持 Touch 'n Go 电子钱包吗？", "可以用 GrabPay 支付吗？",
        "这台机器收现金吗？", "能刷信用卡或借记卡吗？", "支持 DuitNow QR 吗？",
        "请问怎么付款？", "可以用 PayWave 感应刷卡吗？", "投现金的话会找零吗？",
        "收 50 令吉纸币吗？", "付款后可以拿收据吗？", "支持哪些支付方式？",
        "可以用 Boost 或 ShopeePay 吗？", "收硬币吗？", "什么时候扫码付款？",
        "如果扫码扣款失败怎么办？", "刷卡需要手续费吗？", "可以喝完再付吗？",
        "支持 Apple Pay 或 Google Pay 吗？", "卡片要在哪里感应？",
        "Can byr pakai QR tak?", "TnG eWallet blh guna x?", "terima cash ke card je?",
        "DuitNow QR ada x boss?", "blh byr guna debit card x?", "ada resit x lps byr?",
        "byr pakai ewallet blh?", "支持TnG或GrabPay吗？", "可以用QR code byr吗？",
        "cash blh masuk ke?", "PayWave blh pakai x kt mesin ni?", "nk byr guna QR mcm mne?",
        "如果用credit card有charge吗？", "mesin ni cashless je ke?", "只有无现金支付吗？",
        "Is this kiosk cashless only?", "Can I pay with Visa or Mastercard?",
        "Terima Visa dengan Mastercard tak?", "收 Visa 或 Mastercard 吗？",
        "Boleh cetak resit lepas bayar?", "Can I get an e-receipt?",
        "电子钱包哪个都可以用吗？", "Do you accept Alipay or WeChat Pay?",
        "Terima syiling 50 sen tak?", "Where is the QR scanner?",
    ]
    append_batch([{"intent": "payment_query", "utterance": u, "query_type": "payment_policy"} for u in payment_utterances])

    # ------------------------------------------------------------------
    # 11. navigate_menu (80 rows)
    # ------------------------------------------------------------------
    nav_menu_utterances = [
        "Open the menu", "Show me the drink menu", "Go to the menu screen", "Let me browse the drinks",
        "Display all available beverages", "Show menu please", "Take me to the menu page",
        "Can you open the coffee menu?", "Show me the tea options on menu", "Open full drink list",
        "Let me see the menu", "Display the menu board", "Switch to menu view",
        "Bring up the drinks menu", "I want to browse the catalog", "Show iced drinks on menu",
        "Show zero sugar options on menu", "Open beverage list please", "View menu",
        "Back to drink menu",
        "Buka menu", "Tunjuk senarai minuman", "Pergi ke skrin menu", "Saya nak tengok menu dulu",
        "Paparkan semua minuman yang ada", "Tolong buka halaman menu", "Tunjuk menu kopi dan teh",
        "Nak lihat menu penuh", "Bawa saya ke paparan menu", "Tukar ke skrin menu",
        "Tunjuk pilihan minuman sejuk dalam menu", "Buka senarai air", "Lihat menu minuman",
        "Kembali ke menu minuman", "Nak tengok katalog air", "Tunjuk senarai harga pada menu",
        "Buka menu utama minuman", "Papar menu sekarang", "Tunjuk semua pilihan minuman",
        "Saya nak pilih dari menu",
        "打开菜单", "显示所有饮品菜单", "带我去菜单页面", "我想先看看菜单",
        "展示全部可选饮料", "请打开咖啡和茶的菜单", "我想浏览饮品列表", "切换到菜单界面",
        "给我看看完整菜单", "打开饮品目录", "查看菜单页面", "回到点单菜单",
        "只看冰饮菜单", "显示无糖饮品菜单", "我想自己看菜单选", "请把菜单调出来",
        "打开饮料单", "看看有什么菜单", "进入点单列表", "显示菜单吧",
        "Boss buka menu pls", "Show menu skrg", "Open 菜单 please", "Tunjuk menu dulu lah",
        "nk tgk menu air", "buka list minuman pls", "看下menu先", "go to menu page tq",
        "switch ke skrin menu", "tunjuk menu 菜单 pls", "buka menu minuman tq",
        "open drink catalog", "menu please", "menu", "看菜单", "buka menu kopi",
        "show all drinks on screen", "tunjuk semua air kat skrin", "带我看menu", "display menu",
    ]
    append_batch([{"intent": "navigate_menu", "utterance": u} for u in nav_menu_utterances])

    # ------------------------------------------------------------------
    # 12. navigate_payment (80 rows)
    # ------------------------------------------------------------------
    nav_pay_utterances = [
        "Go to payment", "Take me to the payment screen", "Proceed to checkout", "Open payment page",
        "Show the payment QR code", "I am ready to pay now", "Let's check out", "Bring up checkout",
        "Move to payment step", "Show payment options on screen", "Open checkout page please",
        "Take me to pay", "Go to pay screen", "I want to pay right now", "Switch to payment view",
        "Display the payment screen", "Proceed to payment page", "Ready to pay, open QR",
        "Checkout now", "Payment screen please",
        "Pergi ke skrin pembayaran", "Saya nak bayar sekarang", "Terus ke halaman bayar",
        "Buka skrin pembayaran", "Tunjuk kod QR untuk bayar", "Bawa saya ke kaunter bayaran",
        "Sedia untuk bayar", "Pergi ke checkout", "Paparkan skrin bayaran", "Tukar ke halaman pembayaran",
        "Nak terus bayar lah", "Buka paparan bayar sekarang", "Tunjuk cara bayar di skrin",
        "Mulakan pembayaran", "Teruskan ke pembayaran", "Bayar sekarang", "Skrin bayaran sila",
        "Nak scan QR bayar", "Buka checkout sekarang", "Halaman bayar",
        "去付款页面", "带我去结账界面", "我要现在付款", "打开支付页面",
        "显示付款二维码", "前往结账", "调出支付界面", "我准备好了，去买单",
        "切换到付款画面", "打开付款码", "直接去结账吧", "进入支付步骤",
        "给我看付款界面", "我现在要付钱", "请跳转付款页", "结账页面",
        "买单", "扫码付款页面", "去付钱", "打开结账窗口",
        "Boss payment pls", "nk byr skrg, buka payment", "Go to 付款 screen pls",
        "open payment page tq", "tunjuk QR utk byr", "checkout skrg lah",
        "nk checkout pls", "buka 付款界面", "proceed bayar skrg", "payment screen",
        "show payment pls", "nak bayar dah", "go payment", "pay now",
        "带我去payment页面", "buka skrin QR bayar", "let's pay now", "jump to checkout",
        "ready to checkout pls", "open QR payment screen",
    ]
    append_batch([{"intent": "navigate_payment", "utterance": u} for u in nav_pay_utterances])

    # ------------------------------------------------------------------
    # 13. review_order (80 rows)
    # ------------------------------------------------------------------
    review_utterances = [
        "Show my current order", "View my cart", "Review my order", "What did I order so far?",
        "Check my selection", "Let me see my order summary", "Open the cart",
        "Show what is in my cart", "Display current drink configuration", "Check my order details",
        "Recap my order please", "What have I selected?", "Show order summary on screen",
        "Verify my drink settings", "Let me double check my order", "View cart please",
        "What's in my basket?", "Show selected items", "Can I review my order?", "Check cart",
        "Tunjuk pesanan saya sekarang", "Lihat troli saya", "Semak pesanan saya",
        "Apa yang saya pesan tadi?", "Buka ringkasan pesanan", "Tengok isi troli",
        "Boleh semak order saya?", "Paparkan pilihan minuman saya", "Saya nak semak balik pesanan",
        "Apa minuman dalam troli sekarang?", "Tunjuk butiran order saya", "Semak semula pilihan tadi",
        "Semak balik pesanan saya", "Buka bakul pesanan", "Tengok pesanan semasa",
        "Nak check balik air yang saya pilih", "Papar ringkasan order", "Apa setting air saya tadi?",
        "Tunjuk apa yang ada dalam cart", "Semak troli",
        "显示我当前的订单", "查看购物车", "检查一下我点的饮料", "我刚才点了什么？",
        "打开订单详情", "给我看看订单汇总", "我想核对一下现在的配置", "购物车里有什么？",
        "复核我的订单", "展示已选饮品", "再确认一下我选的甜度和冰量", "查看已点内容",
        "看看我目前选了啥", "打开购物车页面", "显示当前点单明细", "帮我读一遍我的订单",
        "我要检查订单", "订单摘要", "看下我选的那杯", "确认前先看下订单",
        "Show order sy skrg", "Check cart pls", "tgk order sy td", "review order tq",
        "apa sy order td?", "buka cart pls", "看下我的cart", "check 当前订单 pls",
        "show 订单详情", "nk semak order dulu", "double check order sy", "cart",
        "my order", "order summary", "看看购物车", "tunjuk cart",
        "what's in my order right now?", "buka ringkasan cart", "check order details pls", "查看订单",
    ]
    append_batch([{"intent": "review_order", "utterance": u} for u in review_utterances])

    # ------------------------------------------------------------------
    # 14. navigate_home (80 rows)
    # ------------------------------------------------------------------
    nav_home_utterances = [
        "Go back to home", "Home screen please", "Return to the main page", "Go to start screen",
        "Back to welcome screen", "Take me home", "Navigate to home page", "Go back to the beginning",
        "Return to home view", "Open home screen", "Exit to main menu", "Back to landing page",
        "Show the home screen", "Jump to home page", "I want to go back home",
        "Close this and go to home", "Back to start page", "Main screen please",
        "Go home", "Return to start",
        "Pergi ke skrin utama", "Kembali ke halaman utama", "Balik ke skrin mula",
        "Bawa saya ke halaman awal", "Tukar ke skrin utama", "Balik ke paparan depan",
        "Saya nak kembali ke muka utama", "Pergi ke laman utama", "Tutup dan balik ke home",
        "Balik ke skrin permulaan", "Buka halaman utama", "Kembali ke menu depan",
        "Nak balik ke skrin depan", "Halaman utama sila", "Papar skrin utama",
        "Keluar ke halaman utama", "Balik awal", "Skrin depan", "Ke muka utama", "Mula balik skrin utama",
        "回到首页", "返回主界面", "回到欢迎页", "带我回主页",
        "切换到首页", "回到最开始的页面", "返回开始画面", "打开主页",
        "退出到主屏幕", "我想回首页", "跳回主界面", "回到初始页面",
        "显示首页画面", "返回首页吧", "主页", "主界面",
        "回主页谢谢", "返回最前页", "关闭并返回首页", "回到起始页",
        "Go home screen pls", "Balik home page tq", "Go to 首页 please", "back home lah",
        "balik skrin utama pls", "home page", "home", "回首页 pls",
        "return to 首页", "nk balik home screen", "tutup ni balik home", "main page pls",
        "skrin utama", "back to home screen", "go back home", "去首页",
        "balik page depan", "return home", "go start screen", "main screen",
    ]
    append_batch([{"intent": "navigate_home", "utterance": u} for u in nav_home_utterances])

    # ------------------------------------------------------------------
    # 15. out_of_scope (90 rows): non-beverage + hard negative "air/cold/charge/menu" words
    # ------------------------------------------------------------------
    oos_utterances = [
        # Hard negatives containing deceptive keywords ("air", "cold", "hot", "ice", "milk", "menu", "pay", "order", "charge")
        "The air in this room is very cold today",
        "Is there an air conditioning repair shop nearby?",
        "My phone battery is dead, can I charge it here?",
        "The weather is so hot outside today",
        "Tell me tomorrow's weather forecast",
        "I need to pay my electricity and water bill",
        "Open my online bank account settings",
        "What is the menu setting on my Android phone?",
        "Can you order me a Grab ride to KLCC?",
        "Play a relaxing song on Spotify",
        "Book a flight ticket to Penang for me",
        "How do I edit a PDF document?",
        "Help me solve this math homework question",
        "Where can I repair my broken laptop screen?",
        "My parking ticket payment failed outside",
        "Order chicken rice and nasi lemak for lunch",
        "Do you sell fried noodles or roti canai?",
        "Can I buy a SIM card or phone top up here?",
        "What time does the nearest shopping mall close?",
        "Give me the football match score last night",
        "Can you translate this essay into French?",
        "Write a Python sorting algorithm for me",
        "Who won the election last year?",
        "What is the exchange rate for USD to MYR?",
        "Tell me a funny joke to cheer me up",
        "Check my flight status for AirAsia",
        "Is there a toilet near this kiosk?",
        "Can I store my luggage inside the machine?",
        "Where is the nearest pharmacy or clinic?",
        "Can you print my boarding pass?",
        "Udara hari ini sangat sejuk lah",
        "Ada kedai baiki aircond dekat sini tak?",
        "Boleh cas telefon saya kat mesin ni tak?",
        "Cuaca kat luar panas gila hari ni",
        "Ramalan cuaca esok macam mana?",
        "Saya nak bayar bil elektrik TNB",
        "Tolong tempah kereta Grab ke Mid Valley",
        "Ada jual nasi lemak dengan ayam goreng tak?",
        "Macam mana nak buat kerja rumah matematik ni?",
        "Kat mana nak baiki skrin laptop yang rosak?",
        "Tiket parking saya tak boleh bayar lah",
        "Mainkan lagu Melayu kat Spotify",
        "Boleh beli topup telefon kat sini?",
        "Pukul berapa LRT terakhir malam ni?",
        "Tandas awam kat tingkat mana ya?",
        "Boleh tolong terjemah surat ni ke bahasa Inggeris?",
        "Siapa menang bola malam tadi?",
        "Ceritakan satu lawak kelakar",
        "Berapa kadar tukaran dolar hari ini?",
        "Nak beli roti canai dua keping",
        "今天空调好冷啊，风好大",
        "附近哪里有修冷气的店？",
        "我的手机没电了，这里可以充电吗？",
        "今天外面的天气太热了",
        "告诉我明天的天气预报",
        "我要缴水电费和电话费",
        "帮我叫一辆 Grab 去机场",
        "在 Spotify 上放一首歌来听",
        "帮我订一张飞槟城的机票",
        "这道微积分作业题怎么做？",
        "哪里可以修理坏掉的笔记本电脑？",
        "这里有卖椰浆饭或炒粿条吗？",
        "请问最近的洗手间在哪里？",
        "可以帮我打印登机牌吗？",
        "昨晚的足球比赛谁赢了？",
        "现在美元兑马币的汇率是多少？",
        "给我讲一个搞笑的笑话",
        "帮我写一段 Python 爬虫代码",
        "这附近哪家超市还没关门？",
        "可以买手机充值卡吗？",
        "asdfghjkl qwertyuiop",
        "testing 123 hello world ???",
        "blah blah blah xyz",
        "The air hari ni sejuk gila",
        "kt mana nk repair laptop rosak?",
        "ada jual nasi goreng x kt sini?",
        "blh tolong charge phone sy x?",
        "can you order pizza and burger for me?",
        "tolong book hotel utk esok",
        "哪儿有卖炸鸡和薯条？",
        "请问地铁站怎么走？",
        "How do I get to the nearest MRT station?",
        "Boleh tunjuk jalan ke stesen MRT terdekat?",
        "Can I withdraw cash from this machine?",
        "Mesin ni boleh keluar duit macam ATM ke?",
        "这台机器可以当 ATM 取钱吗？",
        "Who is the Prime Minister of Malaysia?",
        "Siapa nama perdana menteri sekarang?",
        "给我查一下明天的股票走势",
        "What is the stock price of Apple today?",
    ]
    append_batch([{"intent": "out_of_scope", "utterance": u} for u in oos_utterances])

    df_out = pd.DataFrame(rows)

    # Deduplicate check (case-insensitive stripped)
    norm_u = df_out["user_utterance"].astype(str).str.lower().str.strip()
    dups = df_out[norm_u.duplicated(keep=False)]
    if not dups.empty:
        raise ValueError(f"Duplicate utterances detected:\n{dups[['id', 'intent', 'user_utterance']]}")

    # Assign stratified leak-free splits by scenario_group (70/15/15 per intent)
    rng = random.Random(42)
    for intent_name, grp_df in df_out.groupby("intent"):
        sgs = list(grp_df["scenario_group"].unique())
        rng.shuffle(sgs)
        n = len(sgs)
        n_train = round(n * 0.70)
        n_val = round(n * 0.15)
        train_sgs = set(sgs[:n_train])
        val_sgs = set(sgs[n_train:n_train + n_val])
        for sg in sgs:
            split_label = "train" if sg in train_sgs else ("validation" if sg in val_sgs else "test")
            df_out.loc[(df_out["intent"] == intent_name) & (df_out["scenario_group"] == sg), "split"] = split_label

    df_out = df_out[COLUMNS]
    df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Built {OUT_PATH}: shape={df_out.shape}")
    print("Intent counts:", df_out["intent"].value_counts().to_dict())
    print("Split counts:", df_out["split"].value_counts().to_dict())
    print("Language mix:", df_out["language_mix"].value_counts().to_dict())


if __name__ == "__main__":
    main()
