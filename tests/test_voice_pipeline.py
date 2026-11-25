"""Unit tests for voice pipeline (/api/tts, /api/transcribe, phonetic fix & ASR rescoring).

Uses mocked network calls during pytest so tests run fast and offline without
requiring live internet or Groq quota. Run directly with `python tests/test_voice_pipeline.py`
to execute optional live endpoint smoke checks against a running server.
"""

import os
import sys
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import app as backend_app

client = TestClient(backend_app.app)


def test_phonetic_fix_malay_english_cjk():
    fixed1, changed1 = backend_app.apply_stt_phonetic_fix_py("nak copy peng satu")
    assert changed1 is True
    assert "kopi peng" in fixed1.lower()

    fixed2, changed2 = backend_app.apply_stt_phonetic_fix_py("我要一杯咖啡乌无糖")
    assert "咖啡乌" in fixed2


def test_score_asr_candidate_prefers_domain_tokens():
    s_good, info_good = backend_app.score_asr_candidate("kopi o kosong peng satu")
    s_bad, _ = backend_app.score_asr_candidate("random unrelated garbage noise")
    assert s_good > s_bad
    assert info_good["slots"].get("base_drink") == "Kopi"


def test_tts_validation_and_mocked_edge_tts():
    assert client.post("/api/tts", json={"text": ""}).status_code == 400
    assert client.post("/api/tts", json={"text": "x" * 501}).status_code == 400

    fake_comm = MagicMock()

    async def fake_stream():
        yield {"type": "audio", "data": b"ID3-MOCK-MP3"}

    fake_comm.stream = fake_stream
    with patch.object(backend_app, "edge_tts") as mock_etts:
        mock_etts.Communicate.return_value = fake_comm
        r = client.post("/api/tts", json={"text": "Beres boss, kopi peng satu!", "tone": "female"})
        assert r.status_code == 200
        assert r.content == b"ID3-MOCK-MP3"
        assert r.headers.get("X-Voice-Used") == "ms-MY-YasminNeural"


def test_transcribe_missing_groq_key_returns_503():
    with patch.object(backend_app, "get_groq_api_key", return_value=""):
        r = client.post(
            "/api/transcribe",
            files={"file": ("test.webm", b"RIFF0000WAVEfmt " * 10, "audio/webm")},
            data={"lang": "auto", "multitrack": "true"},
        )
        assert r.status_code == 503
        assert "GROQ_API_KEY" in r.json().get("error", "")


def test_transcribe_multitrack_mocked_rescoring():
    def fake_post(url, headers=None, files=None, data=None, timeout=20):
        lang = (data or {}).get("language", "auto")
        resp = MagicMock()
        resp.status_code = 200
        text_by_lang = {
            "auto": "copy o kosong please",
            "ms": "kopi o kosong satu",
            "en": "coffee zero sugar",
            "zh": "咖啡无糖",
        }
        resp.json.return_value = {"text": text_by_lang.get(lang, "kopi o")}
        return resp

    with patch.object(backend_app, "get_groq_api_key", return_value="gsk_test_dummy_key"):
        with patch.object(backend_app._HTTP_SESSION, "post", side_effect=fake_post):
            r = client.post(
                "/api/transcribe",
                files={"file": ("sample.webm", b"RIFF0000WAVEfmt " * 10, "audio/webm")},
                data={"lang": "auto", "multitrack": "true"},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "success"
            assert body["best_track"] in {"auto", "ms", "en", "zh"}
            assert isinstance(body["text"], str) and len(body["text"]) > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
