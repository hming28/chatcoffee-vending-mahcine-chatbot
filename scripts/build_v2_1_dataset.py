"""Build ChatCoffee v2.1 (1,440 rows x 26 columns, 6 core intents).

Combines the authentic Malaysian Kopitiam core of v1 (950 rows across 4 intents)
with the clean, valuable additions from v2 and v1.5:
  - 6 ML intents: order_drink (350), modify_order (300), health_query (280),
    clarify_order (270), order_confirm (120), cancel_order (120)
  - Excludes bloated UI navigation intents, mechanical word-spliced trilingual
    translations, Takeaway/serving_type, and context-free 1-word fragments that
    degrade single-turn intent accuracy.
  - Enriches schema with temp_detail, ice_level, old_value, new_value, granular
    query_type (caffeine, fat, dairy, lactose, gluten, calories, sugar, etc.),
    is_coreference, shortform_present, stt_noise_present, ui_action, target_screen,
    required_state, confirmation_required, scenario_group, and leak-free split.

Exports NLP_Datasets/chatcoffee_v2_1.csv with UTF-8 BOM (utf-8-sig).
"""

import json
import os
import random
import re
from typing import Dict, List

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "chatcoffee_gold_850.csv")
V15_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "chatcoffee_v1_5_2100.csv")
OUT_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "chatcoffee_v2_1.csv")

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
    "is_coreference",
    "shortform_present",
    "stt_noise_present",
    "ui_action",
    "target_screen",
    "required_state",
    "confirmation_required",
    "scenario_group",
    "split",
]

# (category, ui_action, target_screen, required_state, confirmation_required)
INTENT_META = {
    "order_drink": ("direct_order", "prepare_selection", "detail", "any", "no"),
    "clarify_order": ("ambiguous_exploratory", "answer_or_clarify", "", "any", "no"),
    "modify_order": ("mid_order_correction", "update_selection", "detail", "active_order", "no"),
    "health_query": ("health_qa", "lookup_nutrition", "", "any", "no"),
    "order_confirm": ("order_control", "review_and_confirm", "payment", "complete_order", "yes"),
    "cancel_order": ("order_control", "request_cancel", "home", "active_unpaid_order", "yes"),
}

SHORTFORM_TOKENS = {
    "nk", "xnak", "xnk", "xmau", "xmo", "xbo", "xleh", "xde", "xtau", "x",
    "krg", "krng", "mns", "gla", "ssu", "pns", "sjk", "byr", "hrg", "hrga",
    "brp", "sd", "siudai", "gd", "gadai", "ksg", "kp", "kpi", "wc", "amer",
    "tkr", "cncl", "cnfrm", "pls", "plz", "tq", "thx", "bgi", "bg", "bgtau",
    "blh", "utk", "yg", "dgn", "ngn", "dlm", "kt", "dkt", "skrg", "skit",
    "sgt", "tp", "tpi", "tlg", "sy", "td", "bkn", "oder", "req", "tny",
    "plg", "lbh", "xjadi", "jgn", "dpt", "org", "mnm", "stu", "dh", "aje",
    "je", "1st",
}

MS_WORDS = {
    "saya", "nak", "bagi", "tolong", "buat", "satu", "dua", "tiga", "cawan",
    "panas", "suam", "ais", "sejuk", "dingin", "kurang", "manis", "gula",
    "tanpa", "susu", "pekat", "sejat", "cair", "biasa", "lebih", "tukar",
    "ganti", "ubah", "batal", "batalkan", "pesanan", "minuman", "minum",
    "harga", "berapa", "boleh", "ada", "apa", "mana", "yang", "untuk",
    "dalam", "kalori", "kafein", "laktosa", "sihat", "rendah", "tinggi",
    "bayar", "guna", "tunai", "kad", "resit", "menu", "utama", "balik",
    "semula", "sahkan", "sekarang", "tunjuk", "lihat", "semak", "tak",
    "tidak", "tiada", "jangan", "bukan", "tadi", "jadi", "kepada", "sikit",
    "banyak", "atau", "dan", "kat", "dekat", "sini", "ni", "tu", "ke",
    "pun", "lagi", "terus", "je", "lah", "ya", "air", "teh", "kopi",
    "tarik", "limau", "kosong", "kencing", "diet", "pahit", "beza",
    "dengan", "pilihan", "cadang", "cadangan", "sesuai", "pertama", "kali",
    "mahu", "minta", "pesan", "ambil", "maaf", "tunggu", "sekejap", "jap",
    "kejap", "awak", "anda", "sila", "terima", "kasih", "betul", "ngam",
    "siap", "bancuh", "beres", "padam", "kosongkan", "hapus", "mula",
    "ulang", "troli", "semua", "jangan", "teruskan", "lemak", "bahan",
    "kandungan", "alergi", "alahan", "alatan", "selamat", "perlu", "payah",
    "takde", "tau", "kuat", "pekat", "cair", "rasa", "macam", "guna",
}

EN_WORDS = {
    "i", "id", "ill", "im", "want", "like", "would", "get", "me", "give",
    "make", "order", "one", "two", "three", "cup", "cups", "coffee", "tea",
    "white", "lemon", "hot", "warm", "iced", "ice", "cold", "less", "zero",
    "no", "normal", "extra", "sugar", "sweet", "sweetness", "milk",
    "condensed", "evaporated", "without", "black", "please", "thanks",
    "thank", "you", "can", "could", "may", "is", "are", "the", "a", "an",
    "this", "that", "it", "my", "your", "drink", "drinks", "beverage",
    "beverages", "menu", "price", "how", "much", "cost", "many", "calories",
    "kcal", "caffeine", "lactose", "gluten", "allergen", "allergens",
    "dairy", "healthy", "healthiest", "lowest", "low", "high", "fat",
    "ingredients", "nutrition", "added", "compare", "difference", "between",
    "recommend", "suggest", "suggestion", "change", "switch", "replace",
    "modify", "remove", "instead", "from", "to", "cancel", "reset", "clear",
    "confirm", "confirmed", "proceed", "checkout", "pay", "payment", "cart",
    "current", "what", "which", "any", "do", "does", "have", "available",
    "for", "in", "on", "with", "of", "and", "or", "sorry", "wait", "actually",
    "never", "mind", "only", "just", "now", "first", "time", "visit",
    "bitter", "strong", "diabetic", "friendly", "weight", "loss", "option",
    "options", "contains", "contain", "free", "there", "here", "ready",
    "go", "ahead", "place", "done", "stop", "abort", "start", "over",
    "again", "everything", "selection", "void", "discard", "drop", "erase",
    "forget", "restart", "delete", "finalize", "brew", "looks", "good",
    "right", "all", "correct", "hurry", "fast", "catch", "lrt", "bus",
    "sure", "ok", "okay", "yes", "yep", "yeah", "alright", "perfect",
}


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
    if intent != "health_query":
        return ""
    t = text.lower()
    if re.search(r"lactose|laktosa|乳糖|dairy|tenusu|奶制品", t):
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
    if re.search(r"\b(?:fat|lemak)\b|脂肪", t):
        return "fat"
    if re.search(r"\b(?:ingredients?|bahan|kandungan)\b|成分|配料|原料", t):
        return "ingredients"
    if re.search(r"\b(?:vs|atau|or|compare|banding|mana\s+lebih|mana\s+kurang)\b|还是|哪个更|对比", t):
        return "comparison"
    if re.search(r"sugar|gula|糖|diabetic|kencing\s+manis|糖尿", t):
        if re.search(r"recommend|suggest|cadang|ada|which|mana|apa|推荐|什么|适合", t) and not re.search(r"\b(?:kopi|teh|cham|milo|neslo|americano|white coffee)\b|咖啡|茶|美禄|白咖啡", t):
            return "low_cal_rec"
        return "sugar"
    if re.search(r"calori|kalori|kcal|卡路里|热量|卡|gemuk|胖", t):
        if re.search(r"recommend|suggest|cadang|ada|which|mana|apa|推荐|什么|最健康|最低|fewest|lowest", t) and not re.search(r"\b(?:kopi|teh|cham|milo|neslo|americano|white coffee|this|ni|这杯)\b|咖啡|茶|美禄|白咖啡|这杯", t):
            return "low_cal_rec"
        return "calories"
    if re.search(r"diet|weight|sihat|healthy|减肥|减脂|健康|控糖|养生", t):
        return "low_cal_rec"
    return "nutrition_general"


def infer_coref(text: str, intent: str) -> str:
    if intent not in ("health_query", "modify_order"):
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
    is_coreference: str = "",
    stt_noise_present: str = "no",
    scenario_group: str = "",
) -> Dict[str, object]:
    cat, ui_act, tgt_scr, req_state, conf_req = INTENT_META[intent]
    if temperature == "Hot" and not temp_detail:
        temp_detail = "Hot"
    if temperature == "Iced" and not ice_level:
        ice_level = "Normal Ice"
    row = {
        "id": f"CCV21_{idx:04d}",
        "category": cat,
        "user_utterance": utterance.strip(),
        "language_mix": infer_language_mix(utterance),
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
        "is_coreference": is_coreference or infer_coref(utterance, intent),
        "shortform_present": detect_shortform(utterance),
        "stt_noise_present": stt_noise_present,
        "ui_action": ui_act,
        "target_screen": tgt_scr,
        "required_state": req_state,
        "confirmation_required": conf_req,
        "scenario_group": scenario_group or f"{intent}_v21_{idx:04d}",
        "split": "",
    }
    row["slots"] = build_slots_json(row)
    return row


def main() -> None:
    v1 = pd.read_csv(V1_PATH, keep_default_na=False)
    v15 = pd.read_csv(V15_PATH, keep_default_na=False)
    assert len(v1) == 950, f"Expected 950 v1 rows, got {len(v1)}"

    rows: List[Dict[str, object]] = []
    seen = set()

    def add_row(row: Dict[str, object]) -> None:
        key = row["user_utterance"].lower().strip()
        if key in seen:
            raise ValueError(f"Duplicate utterance: {row['user_utterance']}")
        seen.add(key)
        rows.append(row)

    # 1. All 950 v1 rows (preserving IDs CC_0001..CC_0950)
    for idx, r in v1.iterrows():
        utt = str(r["user_utterance"]).strip()
        intent = str(r["intent"]).strip()
        cat, ui_act, tgt_scr, req_state, conf_req = INTENT_META[intent]
        base = str(r["base_drink"]).strip()
        temp = str(r["temperature"]).strip()
        sweet = str(r["sweetness_level"]).strip()
        milk = str(r["milk_type"]).strip()

        t_low = utt.lower()
        temp_detail = ""
        ice_level = ""
        if temp == "Hot":
            temp_detail = "Warm" if ("温" in utt or re.search(r"\b(?:suam|warm)\b", t_low)) else "Hot"
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
            "is_coreference": infer_coref(utt, intent),
            "shortform_present": detect_shortform(utt),
            "stt_noise_present": "no",
            "ui_action": ui_act,
            "target_screen": tgt_scr,
            "required_state": req_state,
            "confirmation_required": conf_req,
            "scenario_group": f"v1_{intent}_{idx+1:04d}",
            "split": "",
        }
        row["slots"] = build_slots_json(row)
        add_row(row)

    # 2. Import high-value v1.5 additions (ONLY clean non-fragment rows for the 4 core intents)
    #    - order_drink (+60 => 350)
    #    - modify_order (+80 => 300)
    #    - clarify_order (+40 domain/exploratory Q&A only, skipping 40 1-word fragments!)
    #    - health_query (+60 => 280)
    next_idx = 951
    v15_new = v15[v15["id"].str.startswith("CCV15_")].copy()

    for _, r in v15_new.iterrows():
        intent = r["intent"]
        if intent not in ("order_drink", "modify_order", "health_query", "clarify_order"):
            continue
        if intent == "clarify_order" and r.get("prev_bot_prompt", "none") != "none":
            # Skip context-free 1-2 word fragments (handled by DST pending_slot in runtime)
            continue
        add_row(make_row(
            next_idx,
            utterance=r["user_utterance"],
            intent=intent,
            base_drink=r["base_drink"],
            temperature=r["temperature"],
            temp_detail=r["temp_detail"],
            ice_level=r["ice_level"],
            sweetness_level=r["sweetness_level"],
            milk_type=r["milk_type"],
            old_value=r["old_value"],
            new_value=r["new_value"],
            query_type=r["query_type"],
            turn_id=int(r["turn_id"]),
            is_coreference=r["is_coreference"],
            stt_noise_present=r["stt_noise_present"],
        ))
        next_idx += 1

    # 3. Add +10 v2-inspired natural exploratory clarify_order questions (bringing clarify_order 260 -> 270)
    extra_clarify_v2 = [
        {"utterance": "What does siew dai mean?", "intent": "clarify_order"},
        {"utterance": "Apa maksud siew dai?", "intent": "clarify_order"},
        {"utterance": "少甜和Siu Dai是一样的吗？", "intent": "clarify_order"},
        {"utterance": "Any coffee that is not too bitter?", "intent": "clarify_order"},
        {"utterance": "Ada kopi yang tak pahit sangat tak?", "intent": "clarify_order"},
        {"utterance": "Which drink is best for a first visit?", "intent": "clarify_order"},
        {"utterance": "Minuman apa sesuai untuk kali pertama?", "intent": "clarify_order"},
        {"utterance": "Is there an option without milk?", "intent": "clarify_order", "milk_type": "None"},
        {"utterance": "Ada pilihan minuman tanpa susu?", "intent": "clarify_order", "milk_type": "None"},
        {"utterance": "请问有不加奶的咖啡或茶吗？", "intent": "clarify_order", "milk_type": "None"},
    ]
    for item in extra_clarify_v2:
        add_row(make_row(next_idx, **item))
        next_idx += 1

    # 4. Add 120 natural order_confirm rows (curated EN / MS / ZH / Rojak / shortforms — no mechanical word-splicing)
    confirm_utterances = [
        # EN (28)
        "Confirm my order", "Confirm order please", "Yes, confirm it", "Okay, place my order",
        "Go ahead and brew it", "Sure, proceed with this order", "Looks good, confirm!",
        "Alright, make it now", "Yes, that's everything, confirm", "Proceed to brew please",
        "Confirmed, let's pay", "Yes please, proceed", "Ok done, confirm order",
        "All correct, place order", "Hurry, confirm my drink now", "Confirm fast, need to catch LRT",
        "Yep, confirm this cup", "Looks right, go ahead", "Yes, brew this coffee now",
        "Perfect, confirm my selection", "Please finalize my order", "Ready, confirm order",
        "Sure, make this drink", "Okay proceed", "Yes, I confirm",
        "Lock this order and brew it", "Confirm this one, thanks!", "Yes, send it to the machine",
        # MS (30)
        "Sahkan pesanan saya", "Ya, sahkan sekarang", "Ok, teruskan pesanan",
        "Betul, buat sekarang", "Boleh, sahkan order ini", "Ngam, terus buat lah",
        "Ok confirm order boss", "Ya betul, teruskan bayar", "Semua betul, sahkan ya",
        "Cepat sikit, sahkan order sekarang", "Siap, boleh bancuh sekarang", "Jadi, sahkan yang ini",
        "Ok boleh, proceed je", "Betul semua, hantar pesanan", "Baik, buat air ini sekarang",
        "Ya saya sahkan pesanan ini", "Terus bancuh je boss", "Ok ngam, bayar dan buat",
        "Dah cukup, sahkan order", "Sahkan terus lah", "Teruskan je pesanan ni",
        "Sila proses pesanan saya", "Beres, buat terus!", "Ya boleh sahkan",
        "Ok mantap, sahkan sekarang", "Terus sahkan ya", "Saya setuju, buat sekarang",
        "Sahkan pesanan tadi", "Boleh terus bancuh", "Ok siap, sahkan pesanan",
        # ZH / Cantonese (28)
        "确认我的订单", "确认下单", "好的，现在帮我做吧", "没问题，确认这杯",
        "对的，开始冲泡吧", "就点这杯，确认", "可以了，请下单", "全部正确，确认订单",
        "快点确认，我赶时间", "好，确认付款下单", "行，就按这个做", "确定要这杯了",
        "是的，请立刻制作", "确认无误，开始吧", "好啦，下单谢谢", "帮我出杯吧，已确认",
        "好嘞，确定下单咯", "没错，就是这杯，做吧", "就这样，确认下单！", "可以，立刻冲泡吧",
        "唔该确认落单", "好呀，帮我做呢杯", "确认啦，唔该晒", "无问题，照呢个做",
        "行了，开始制作吧", "确定这杯，谢谢老板", "搞定，直接下单", "好，确认这杯饮品",
        # Rojak & Shortforms (34)
        "Ok confirm lah boss", "Ya, confirm order ni pls", "Ngam lah, confirm terus",
        "Confirm 订单 please", "好，confirm order skrg", "Cepat confirm, nk catch bus",
        "Betul boss, proceed buat air", "Ok done, 确认下单 tq", "Sahkan je, 没问题",
        "Yes ngam, make it now", "Boleh, 确认这杯 pls", "Confirm skrg tq",
        "Ok 确认!", "Ya boleh, brew skrg", "Proceed lah boss",
        "cnfrm order pls", "ok byr skrg, confirm", "confirm je xpayah tunggu",
        "ok boss cnfrm", "cnfrm skrg tq", "ya ngam cnfrm lah",
        "boleh boss, proceed order skrg", "ok confirm order sy pls", "confirm lah xde perubahan",
        "ok terus bancuh lah boss", "done, cnfrm order ni", "confirm terus, sy nk bayar",
        "sahkan order sy skrg tq", "yes confirm lah pls", "ngam boss, cnfrm terus",
        "可以了 boss, confirm!", "好嘞，confirm skrg pls", "对，cnfrm 这杯 tq",
        "ok boss, confirm & brew now",
    ]
    for u in confirm_utterances:
        add_row(make_row(next_idx, utterance=u, intent="order_confirm", turn_id=2))
        next_idx += 1

    # 5. Add 120 natural cancel_order rows (curated EN / MS / ZH / Rojak / shortforms)
    cancel_utterances = [
        # EN (30)
        "Cancel my order", "Cancel the current order please", "Reset everything", "Clear my cart",
        "Start over from scratch", "Never mind, cancel it", "Abort this order", "Discard my selection",
        "I don't want to order anymore", "Please void this order", "Drop the order", "Cancel all",
        "Reset the kiosk please", "Stop the order", "Erase my current drink", "No thanks, cancel it",
        "Forget it, clear the order", "Take me out, cancel everything", "I changed my mind, cancel order",
        "Remove everything from cart", "Restart my order", "Delete current selection",
        "Cancel this cup please", "Void my current selection", "Reset to blank",
        "Please cancel before brewing", "No need already, cancel!", "Cancel my drink selection",
        "Stop it, clear everything", "Clear current selection please",
        # MS (30)
        "Batalkan pesanan saya", "Batal order ini", "Reset semula semua", "Kosongkan pesanan",
        "Tak jadi order lah", "Saya tak nak beli lagi", "Padam semua pilihan", "Mula semula dari awal",
        "Batalkan semua sekarang", "Tak jadi minum, batal je", "Hapus pesanan semasa", "Berhenti, jangan buat lagi",
        "Cancel order saya sekarang", "Tutup pesanan ni", "Jangan teruskan, batal",
        "Saya nak batalkan minuman ini", "Ulang semula pesanan", "Kosongkan troli sekarang",
        "Tak perlu buat, cancel terus", "Batalkan semuanya", "Tolong batalkan sebelum bayar",
        "Tak payah buat dah, batal!", "Batalkan pesanan yang tadi", "Saya ubah fikiran, batal",
        "Buang pesanan ni semua", "Reset balik ke mula", "Jangan bancuh, batalkan",
        "Hentikan pesanan sekarang", "Tak jadi lah boss, batal", "Kosongkan semua dalam troli",
        # ZH / Cantonese (28)
        "取消我当前的订单", "取消订单", "全部重置", "清空购物车",
        "我不要点了，帮我取消", "算了，取消吧", "重新开始点单", "把刚才选的都删掉",
        "停止下单，取消全部", "不好意思，我不买了", "重置机器界面", "帮我作废这个订单",
        "先取消掉，等下再点", "不要做了，取消谢谢", "全部清零", "不喝了，帮我取消订单",
        "重新来过，清空", "不要了，帮我取消掉", "取消刚才那杯", "唔好意思，取消订单",
        "唔买啦，帮我取消", "取消晒全部先", "重头再来过", "删掉所有选择",
        "不用做了，取消吧", "麻烦取消我现在的订单", "停下来，取消这杯", "清空刚才选的饮料",
        # Rojak & Shortforms (32)
        "Cancel order pls", "Batal order ni tq", "xjadi order lah, cancel",
        "Cancel 当前订单 please", "Reset balik semua pls", "不要了，cancel pls",
        "xnk order dh, batal", "cncl order skrg", "cncl je boss",
        "Tak jadi lah, reset semula", "Maaf cancel order ya", "清掉这个order",
        "xmo beli dh, clear cart", "Cancel semua, start over", "Jangan buat, 取消订单",
        "Reset order sy pls", "Abort lah, tak sempat tunggu", "xjadi beli, batal semua",
        "Drop order ni please", "Clear semuanya tq", "取消算了，赶时间",
        "cncl semua skrg pls", "xnk teruskan, cncl je", "batal terus boss tq",
        "sy xnak order dah, reset", "padam order td pls", "xjadi lah, clear cart",
        "cncl dulu lah boss", "jangan bancuh, cncl tq", "不要了 boss, cncl pls",
        "reset je lah semua", "xmo order skrg, batal",
    ]
    for u in cancel_utterances:
        add_row(make_row(next_idx, utterance=u, intent="cancel_order", turn_id=2))
        next_idx += 1

    df_out = pd.DataFrame(rows)

    # Leak-free stratified split by scenario_group (70/15/15 per intent)
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
