"""Regression & parity tests verifying audit fixes across KB, ML artifacts, FastAPI, and index.html."""

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app
from tests.verify_html_node import run_node_verification

client = TestClient(app)


def test_serve_root_index_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "ChatCoffee" in r.text
    assert "KOPITIAM_KB" in r.text


def test_cleaned_datasets_integrity():
    v15_path = os.path.join(ROOT_DIR, "NLP_Datasets", "coffee-rojak.csv")
    with open(v15_path, "rb") as f:
        assert f.read(3) == b"\xef\xbb\xbf", "v1.5 CSV must include UTF-8 BOM for Excel Chinese display"
    v15 = pd.read_csv(v15_path, keep_default_na=False)
    assert len(v15) == 2100
    assert v15["intent"].nunique() == 14
    assert "serving_type" not in v15.columns
    assert set(v15["split"].unique()) == {"train", "validation", "test"}

    kb = pd.read_csv(os.path.join(ROOT_DIR, "NLP_Datasets", "unified_coffee_kb.csv"), keep_default_na=False)
    assert len(kb) == 137
    assert kb["drink_name"].duplicated().sum() == 0
    assert (kb["milk_type"] == "").sum() == 0
    for forbidden in ("Beer Gao", "Orh Gao", "Ice kosong", "Bandung", "Horlick"):
        assert not kb["drink_name"].str.contains(forbidden, case=False).any()
    for required in ("Teh O Peng", "Milo", "Milo O", "Milo O Peng", "Milo O Kosong Peng", "Teh O Ais Limau Siu Dai"):
        assert required in set(kb["drink_name"]), f"Missing {required} from KB"


def test_model_parity_joblib_json_and_index_html():
    jb = joblib.load(os.path.join(ROOT_DIR, "models", "chatcoffee_intent_classifier.joblib"))
    with open(os.path.join(ROOT_DIR, "models", "chatcoffee_model_web.json"), "r", encoding="utf-8") as f:
        wj = json.load(f)
    with open(os.path.join(ROOT_DIR, "index.html"), "r", encoding="utf-8") as f:
        html = f.read()

    m = re.search(r"(?:const|let) ROJAK_ML_MODEL = (\{[^\n]*\});", html)
    assert m is not None
    embedded = json.loads(m.group(1))

    jb_vocab_size = len(jb["vectorizer"].vocabulary_)
    assert jb_vocab_size == len(wj["vocabulary"]) == len(embedded["vocabulary"]) >= 842
    assert len(jb["classes"]) == 8
    assert list(jb["classes"]) == wj["classes"] == embedded["classes"]

    r_model = client.get("/models/chatcoffee_model_web.json")
    assert r_model.status_code == 200
    served_model = r_model.json()
    assert len(served_model["vocabulary"]) == len(wj["vocabulary"])


def test_audit_bugfixes_ner_dst_pricing_and_ood():
    # 1. "brp?" -> price_query
    r_brp = client.post("/api/chat", json={"utterance": "brp?", "session_id": uuid.uuid4().hex}).json()
    assert r_brp["intent"] == "price_query"

    # 2. "温度可以改吗" must NOT extract Warm/Hot
    r_temp = client.post("/api/chat", json={"utterance": "温度可以改吗", "session_id": uuid.uuid4().hex}).json()
    assert "temperature" not in r_temp["slots"]

    # 3. "RM 4.0" must NOT extract Zero Sugar
    r_rm = client.post("/api/chat", json={"utterance": "RM 4.0", "session_id": uuid.uuid4().hex}).json()
    assert "sweetness_level" not in r_rm["slots"]

    # 4. Gibberish OOD returns fallback or out_of_scope
    r_ood = client.post("/api/chat", json={"utterance": "asdfghjkl qwerty", "session_id": uuid.uuid4().hex}).json()
    assert r_ood["intent"] in ("fallback", "out_of_scope")

    # 5. "Kopi panas no sugar" auto-couples milk_type="None" (Kopi O Kosong, RM 3.20)
    r_kopi = client.post("/api/chat", json={"utterance": "Kopi panas no sugar", "session_id": uuid.uuid4().hex}).json()
    assert r_kopi["state"]["milk_type"] == "None"
    assert r_kopi["state"]["sweetness_level"] == "Zero Sugar"
    assert r_kopi["state"]["price_rm"] == 3.20

    # 6. Cham O panas exact price RM 3.70, Teh O Peng exact price RM 4.00
    r_cham = client.post("/api/chat", json={"utterance": "Cham O panas", "session_id": uuid.uuid4().hex}).json()
    assert r_cham["state"]["price_rm"] == 3.70
    r_teho = client.post("/api/chat", json={"utterance": "Teh O Peng", "session_id": uuid.uuid4().hex}).json()
    assert r_teho["state"]["price_rm"] == 4.00

    # 7. Reserved keyword 'kopi' rejected in /api/learn_slang
    r_bad_slang = client.post("/api/learn_slang", json={"slang": "kopi", "meaning": "air"})
    assert r_bad_slang.status_code == 400

    # 8. Shortform normalization & slot extraction (v1 upgrades)
    r_sf1 = client.post("/api/predict_intent", json={"utterance": "Boss nk 1 kp o pns krg mns pls?"}).json()
    assert "kurang manis" in r_sf1["clean_text"]
    assert "kopi o" in r_sf1["clean_text"]
    assert r_sf1["predicted_intent"] == "order_drink"
    assert r_sf1["slots"]["base_drink"] == "Kopi"
    assert r_sf1["slots"]["temperature"] == "Hot"
    assert r_sf1["slots"]["sweetness_level"] == "Less Sugar"
    assert r_sf1["slots"]["milk_type"] == "None"

    r_sf2 = client.post("/api/predict_intent", json={"utterance": "nk teh c ais sd 1 tq"}).json()
    assert r_sf2["predicted_intent"] == "order_drink"
    assert r_sf2["slots"]["base_drink"] == "Teh"
    assert r_sf2["slots"]["temperature"] == "Iced"
    assert r_sf2["slots"]["sweetness_level"] == "Less Sugar"
    assert r_sf2["slots"]["milk_type"] == "Evaporated Milk"

    r_sf3 = client.post("/api/predict_intent", json={"utterance": "少糖pls, nk amer ais xnak gla!"}).json()
    assert "please" in r_sf3["clean_text"]
    assert "tak nak" in r_sf3["clean_text"]
    assert r_sf3["slots"]["base_drink"] == "Americano"
    assert r_sf3["slots"]["temperature"] == "Iced"
    assert r_sf3["slots"]["sweetness_level"] == "Zero Sugar"
    assert r_sf3["slots"]["milk_type"] == "None"

    # 9. 8-intent core kiosk ice/warm slots & OOS
    sid = uuid.uuid4().hex
    r_ice = client.post("/api/chat", json={"utterance": "Kopi peng kurang ais siu dai", "session_id": sid}).json()
    assert r_ice["slots"]["ice_level"] == "Less Ice"

    r_oos2 = client.post("/api/chat", json={"utterance": "The air in this room is very cold today", "session_id": uuid.uuid4().hex}).json()
    assert r_oos2["intent"] in ("fallback", "out_of_scope")


def test_concatenated_kopitiam_and_mixed_ice():
    r_cham = client.post(
        "/api/chat",
        json={"utterance": "I want a champeng kurang manis kurang ice", "session_id": uuid.uuid4().hex},
    ).json()
    assert r_cham["slots"]["base_drink"] == "Cham"
    assert r_cham["slots"]["temperature"] == "Iced"
    assert r_cham["slots"]["sweetness_level"] == "Less Sugar"
    assert r_cham["slots"]["ice_level"] == "Less Ice"

    from app import extract_slots

    for utt, expected in [
        ("kopipeng less ais", {"base_drink": "Kopi", "temperature": "Iced", "ice_level": "Less Ice"}),
        ("tehc tanpa ice", {"base_drink": "Teh", "milk_type": "Evaporated Milk", "temperature": "Iced", "ice_level": "No Ice"}),
        ("kopio no ais", {"base_drink": "Kopi", "milk_type": "None", "temperature": "Iced", "ice_level": "No Ice"}),
        ("teholimau", {"base_drink": "Teh O Limau", "milk_type": "None"}),
    ]:
        slots = extract_slots(utt)
        for k, v in expected.items():
            assert slots.get(k) == v, f"{utt} -> {slots}"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js required for JS engine checks")
def test_frontend_js_suites():
    run_node_verification()
    for script_name in ("test_kiosk_rules.js", "test_lock_milk.js", "test_all_items_matrix.js"):
        path = os.path.join(ROOT_DIR, "tests", script_name)
        res = subprocess.run(["node", path], cwd=ROOT_DIR, capture_output=True, text=True)
        assert res.returncode == 0, f"{script_name} failed:\n{res.stdout}\n{res.stderr}"


def test_conversation_fixes_and_table_issues():
    # 1. "帅哥，我是美式男孩，所以我要一杯冷咖啡少冰" -> Kopi Iced Less Ice (not Americano)
    r_boy = client.post(
        "/api/chat",
        json={"utterance": "帅哥，我是美式男孩，所以我要一杯冷咖啡少冰", "session_id": uuid.uuid4().hex},
    ).json()
    assert r_boy["state"]["base_drink"] == "Kopi"
    assert r_boy["state"]["temperature"] == "Iced"
    assert r_boy["state"]["ice_level"] == "Less Ice"
    assert r_boy["state"]["temp_detail"] is None

    # 2. "bro, kopi o satu da bao, ikat tepi" -> Kopi O Iced directly
    r_ikat = client.post(
        "/api/chat",
        json={"utterance": "bro, kopi o satu da bao, ikat tepi", "session_id": uuid.uuid4().hex},
    ).json()
    assert r_ikat["state"]["base_drink"] == "Kopi"
    assert r_ikat["state"]["milk_type"] == "None"
    assert r_ikat["state"]["temperature"] == "Iced"

    # 3. Multi-turn hot negation: "kopi o" -> "tak mau panas" -> Iced (NOT Hot), and "tukar mana, bro???" asks what to change
    sid_neg = uuid.uuid4().hex
    r_ko = client.post("/api/chat", json={"utterance": "kopi o", "session_id": sid_neg}).json()
    assert "Kopi O" in r_ko["response"]
    r_neg = client.post("/api/chat", json={"utterance": "tak mau panas", "session_id": sid_neg}).json()
    assert r_neg["state"]["temperature"] == "Iced"
    assert r_neg["state"]["temp_detail"] is None
    r_tukar = client.post("/api/chat", json={"utterance": "tukar mana, bro???", "session_id": sid_neg}).json()
    assert "tukar" in r_tukar["response"].lower()
    assert "kepada" not in r_tukar["response"].lower()

    # 4. Table Issue 1: health_query must not ignore base_drink for milk/caffeine
    r_cham_q = client.post(
        "/api/chat",
        json={"utterance": "Does cham contain milk or caffeine?", "session_id": uuid.uuid4().hex},
    ).json()
    assert r_cham_q["intent"] == "health_query"
    assert "cham" in r_cham_q["response"].lower()
    assert "110" in r_cham_q["response"]
    assert "milk" in r_cham_q["response"].lower() or "lactose" in r_cham_q["response"].lower()

    # 5. Table Issue 2: ice_level & Warm (Suam) in reply, and temp_detail cleared when Iced
    r_cp = client.post(
        "/api/chat",
        json={"utterance": "I want a champeng kurang manis kurang ice", "session_id": uuid.uuid4().hex},
    ).json()
    assert r_cp["state"]["temp_detail"] is None
    assert "Less Ice" in r_cp["response"]

    r_suam = client.post(
        "/api/chat",
        json={"utterance": "Teh suam kurang manis", "session_id": uuid.uuid4().hex},
    ).json()
    assert r_suam["state"]["temp_detail"] == "Warm"
    assert "Warm" in r_suam["response"] and "Suam" in r_suam["response"]

    # 6. Table Issue 3: Condensed milk + Kosong conflict explanation
    r_conflict = client.post(
        "/api/chat",
        json={"utterance": "Kopi susu pekat kosong", "session_id": uuid.uuid4().hex},
    ).json()
    assert r_conflict["state"]["sweetness_level"] == "Less Sugar"
    assert "condensed milk" in r_conflict["response"].lower() or "susu pekat" in r_conflict["response"].lower() or "炼奶" in r_conflict["response"]

    # 7. Table Issue 4: Menu browse & QR payment lightweight keyword routing
    r_menu = client.post(
        "/api/chat",
        json={"utterance": "Open the drink menu please", "session_id": uuid.uuid4().hex},
    ).json()
    assert r_menu["intent"] == "navigate_menu"
    assert r_menu["target_screen"] == "menu"

    r_pay = client.post(
        "/api/chat",
        json={"utterance": "Can I pay using QR code?", "session_id": uuid.uuid4().hex},
    ).json()
    assert r_pay["intent"] == "payment_query"
    assert "qr" in r_pay["response"].lower()
