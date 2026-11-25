# ChatCoffee — Malaysian Smart Coffee Kiosk Chatbot

A multilingual Kopitiam coffee ordering chatbot supporting **English, Bahasa Melayu, Mandarin (中文), and Bahasa Rojak / Manglish** for a single-cup unmanned kiosk workflow.

---

## Quick Start

### Option 1: Full-Stack Server (Recommended — UI + Voice STT/TTS)
```bash
pip install -r requirements.txt
python app.py
```
Then open `http://localhost:8000` in your browser.

> **Optional Voice Setup:** Add `GROQ_API_KEY` to a `.env` file to enable Groq Whisper multilingual voice transcription. Without it, the kiosk automatically falls back to browser Web Speech.

### Option 2: Frontend Only (Static / Offline)
- **GitHub Pages / Local HTTP server:** Serve `index.html` — it automatically fetches the latest `models/chatcoffee_model_web.json` on startup.
- **Offline:** Double-click `index.html` directly in your browser — it falls back to its embedded model automatically.

---

## What It Does

- **14-Intent NLU & Slang Understanding:** Uses a `LogisticRegression` classifier (`1,954` TF-IDF n-gram features) paired with Malaysian Kopitiam slang normalization (`NLP_Datasets/shortform-dict.json`) and runtime slang learning (`/api/learn_slang`).
- **Drink & Customization Extraction:** Understands drink names, temperature (`Hot` / `Iced` / `Warm`), sweetness (`Kosong`, `Siu Dai`, `Normal`, `Gah Dai`), milk type, and ice level while enforcing kiosk rules (single-cup limit, milk locking on `Teh O Limau` / `Americano`, and sugar guards).
- **137-Item Coffee Knowledge Base:** `NLP_Datasets/unified_coffee_kb.csv` stores verified prices (RM 3.20–RM 6.80), calories, sugar, caffeine, lactose, and gluten details across 14 drink families.
- **Voice Input & Output:** `POST /api/transcribe` (Groq Whisper + domain rescoring) and `POST /api/tts` (Microsoft Edge-TTS for Malay, Mandarin, and English).

---

## Project Structure

```text
.
├── app.py                                              # FastAPI backend server & REST APIs
├── index.html                                          # Kiosk web UI (auto-loads models/chatcoffee_model_web.json)
├── V2_ChatCoffee_NLP_Model_Training_and_Evaluation.ipynb  # V2 model training & evaluation notebook
├── models/
│   ├── chatcoffee_intent_classifier.joblib             # Scikit-learn TF-IDF + LogisticRegression model
│   └── chatcoffee_model_web.json                       # Web-exported model weights & slang dictionary
├── NLP_Datasets/
│   ├── chatcoffee_v1_5_2100.csv                        # Primary 2,100-row intent & slot dataset
│   ├── chatcoffee_gold_850.csv                         # Baseline gold dataset
│   ├── unified_coffee_kb.csv                           # 137-item Kopitiam beverage knowledge base
│   └── shortform-dict.json                             # Malaysian slang / shortform dictionary
├── scripts/
│   ├── train_and_export.py                             # Re-train models & sync index.html
│   ├── build_v1_5_dataset.py                           # Dataset builder script
│   └── validate_v1_dataset.py                          # Dataset validation script
├── tests/                                              # Pytest & Node.js automated test suite
├── Dockerfile / Procfile / requirements.txt            # Deployment configuration
└── README.md
```

---

## Re-train & Test

```bash
# Re-train models and sync models/ + index.html
python scripts/train_and_export.py

# Run test suite
python -m pytest -v
```
