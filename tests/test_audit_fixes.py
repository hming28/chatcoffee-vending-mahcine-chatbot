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
    csv_path = os.path.join(ROOT_DIR, "NLP_Datasets", "chatcoffee_gold_850.csv")
    with open(csv_path, "rb") as f:
        assert f.read(3) == b"\xef\xbb\xbf", "v1 CSV must include UTF-8 BOM for Excel Chinese display"
    gold = pd.read_csv(csv_path)
    assert len(gold) == 950
    for col in ("target_response", "serving_type", "old_value", "new_value", "customer_id"):
        assert col not in gold.columns

    v21_path = os.path.join(ROOT_DIR, "NLP_Datasets", "chatcoffee_v2_1.csv")
    with open(v21_path, "rb") as f:
        assert f.read(3) == b"\xef\xbb\xbf", "v2.1 CSV must include UTF-8 BOM for Excel Chinese display"
    v21 = pd.read_csv(v21_path, keep_default_na=False)
    assert len(v21) == 1440
    assert v21["intent"].nunique() == 6
    assert "serving_type" not in v21.columns
    assert set(v21["split"].unique()) == {"train", "validation", "test"}

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
    assert len(jb["classes"]) == 6
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

    # 4. Gibberish OOD returns fallback
    r_ood = client.post("/api/chat", json={"utterance": "asdfghjkl qwerty", "session_id": uuid.uuid4().hex}).json()
    assert r_ood["intent"] == "fallback"

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

    # 9. v2.1 6-intent ML confirm/cancel expressions, ice_level, and OOD fallback
    sid = uuid.uuid4().hex
    r_ice = client.post("/api/chat", json={"utterance": "Kopi peng kurang ais siu dai", "session_id": sid}).json()
    assert r_ice["slots"]["ice_level"] == "Less Ice"
    assert r_ice["state"]["status"] == "AWAITING_CONFIRMATION"

    r_cnfrm = client.post("/api/chat", json={"utterance": "ok ngam cnfrm lah", "session_id": sid}).json()
    assert r_cnfrm["intent"] == "order_confirm"
    assert r_cnfrm["state"]["status"] == "DISPENSING"

    sid2 = uuid.uuid4().hex
    client.post("/api/chat", json={"utterance": "Teh O panas kosong", "session_id": sid2})
    r_cncl = client.post("/api/chat", json={"utterance": "xjadi order lah, cncl dulu", "session_id": sid2}).json()
    assert r_cncl["intent"] == "system_reset"
    assert r_cncl["state"]["base_drink"] is None

    r_oos2 = client.post("/api/chat", json={"utterance": "Book a flight ticket to Penang", "session_id": uuid.uuid4().hex}).json()
    assert r_oos2["intent"] == "fallback"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js required for JS engine checks")
def test_frontend_js_suites():
    run_node_verification()
    for script_name in ("test_kiosk_rules.js", "test_lock_milk.js", "test_all_items_matrix.js"):
        path = os.path.join(ROOT_DIR, "tests", script_name)
        res = subprocess.run(["node", path], cwd=ROOT_DIR, capture_output=True, text=True)
        assert res.returncode == 0, f"{script_name} failed:\n{res.stdout}\n{res.stderr}"
