"""Regression suite exercising FastAPI /api/chat multi-turn dialogue state tracking."""

import os
import sys
import uuid
from fastapi.testclient import TestClient

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

client = TestClient(app)


def chat_turn(utterance: str, sid: str):
    r = client.post("/api/chat", json={"utterance": utterance, "session_id": sid})
    assert r.status_code == 200
    return r.json()


def test_multiturn_english_order_and_modification():
    sid = f"reg-{uuid.uuid4().hex[:8]}"
    r1 = chat_turn("I want coffee", sid)
    assert r1["state"]["base_drink"] == "Kopi"
    assert r1["state"]["pending_slot"] == "temperature"

    r2 = chat_turn("Iced please", sid)
    assert r2["state"]["temperature"] == "Iced"
    assert r2["state"]["status"] == "AWAITING_CONFIRMATION"
    assert r2["state"]["price_rm"] == 4.50

    r3 = chat_turn("Make it less sugar", sid)
    assert r3["state"]["sweetness_level"] == "Less Sugar"
    assert r3["state"]["status"] == "AWAITING_CONFIRMATION"

    r4 = chat_turn("Confirm", sid)
    assert r4["intent"] == "order_confirm"
    assert r4["state"]["status"] == "DISPENSING"


def test_multiturn_chinese_and_no_milk_switch():
    sid = f"reg-{uuid.uuid4().hex[:8]}"
    r1 = chat_turn("我要一杯茶", sid)
    assert r1["state"]["base_drink"] == "Teh"
    assert r1["state"]["pending_slot"] == "temperature"

    r2 = chat_turn("热的不要加奶少甜", sid)
    assert r2["state"]["temperature"] == "Hot"
    assert r2["state"]["milk_type"] == "None"
    assert r2["state"]["sweetness_level"] == "Less Sugar"
    assert r2["state"]["price_rm"] == 3.20

    r3 = chat_turn("Reset", sid)
    assert r3["intent"] == "system_reset"
    assert r3["state"]["status"] == "IDLE"
    assert r3["state"]["base_drink"] is None


def test_rojak_single_shot_and_health_qa_no_state_mutation():
    sid = f"reg-{uuid.uuid4().hex[:8]}"
    r1 = chat_turn("Boss, bagi satu Cham Peng Siu Dai", sid)
    assert r1["state"]["base_drink"] == "Cham"
    assert r1["state"]["temperature"] == "Iced"
    assert r1["state"]["sweetness_level"] == "Less Sugar"
    assert r1["state"]["price_rm"] == 5.00

    # Asking about calories/lactose mid-order should NOT clobber cart
    r2 = chat_turn("Does this cup contain lactose?", sid)
    assert r2["intent"] == "health_query"
    assert "lactose" in r2["response"].lower()
    assert r2["state"]["base_drink"] == "Cham"
    assert r2["state"]["status"] == "AWAITING_CONFIRMATION"
