# ChatCoffee — Multilingual Malaysian Smart Coffee Kiosk Chatbot

Interactive multi-turn voice and text ordering kiosk for Malaysian Kopitiam beverages supporting **English, Bahasa Melayu, Mandarin (中文), and colloquial Bahasa Rojak / Manglish** (single-cup unmanned dispenser workflow).

---

## Key Features

- **Dual-Script ML Intent Classification (`842` TF-IDF n-gram features)**:
  - Stratified 80/20 split (`680` train / `170` held-out test) on `NLP_Datasets/chatcoffee_gold_850.csv` (`850` utterances across 4 intents: `order_drink`, `clarify_order`, `modify_order`, `health_query`).
  - **LinearSVC Benchmark**: `98.24%` Test Accuracy (`0.9812` Macro F1).
  - **Deployed LogisticRegression** (synced 1:1 across `models/chatcoffee_intent_classifier.joblib`, `models/chatcoffee_model_web.json`, and `index.html`): `97.06%` Test Accuracy (`0.9695` Macro F1).
- **Dual-Script NER & Kopitiam DST**:
  - Extracts `base_drink`, `temperature`, `sweetness_level`, `milk_type`, `temp_detail`, `ice_level` across Malay, English, Manglish, and Chinese (`318` flattened normalization rules from `NLP_Datasets/shortform-dict.json` + runtime human-in-the-loop slang teaching via `/api/learn_slang`).
  - Enforces physical unmanned kiosk rules: single-cup dispenser limit, auto milk-locking (`Teh O Limau`, `Americano`), Zero Sugar / `Kosong` auto-coupling on `Kopi`/`Teh`/`Milo`/`Cham`/`Neslo`, and condensed-milk zero-sugar guard.
- **Grounded Coffee Knowledge Base (`137` Kopitiam Records)**:
  - `NLP_Datasets/unified_coffee_kb.csv` covers Hot & Iced variations for `Kopi`, `Kopi O`, `Kopi C`, `Teh Tarik`, `Teh O`, `Teh C`, `Teh O Limau`, `Cham`, `White Coffee`, `Milo`, `Milo O`, `Milo C`, `Neslo`, and `Americano` with verified prices (RM 3.20–RM 6.80), calories, sugar, caffeine, lactose, and gluten metadata.
- **4-Track Multilingual ASR + Neural TTS**:
  - `POST /api/transcribe` runs parallel Groq Whisper (`auto`, `ms`, `en`, `zh`) with domain NLU rescoring (`score_asr_candidate`) and phonetic correction (`apply_stt_phonetic_fix_py`), with Web Speech fallback in the browser (`index.html`).
  - `POST /api/tts` synthesizes neural Malay (`ms-MY-*`), Mandarin (`zh-CN-*`), and Singapore/Malaysian English (`en-SG-*`) speech via Microsoft Edge-TTS.

---

## Project Structure

```text
.
├── app.py                                           # FastAPI backend server (serves index.html & REST APIs)
├── index.html                                       # Single-file standalone/connected Kiosk UI + synced JS NLU/DST
├── ChatCoffee_NLP_Model_Training_and_Evaluation.ipynb # End-to-end training, NER benchmark, Figure 2 & simulation notebook
├── Dockerfile / Procfile / requirements.txt         # Container & PaaS deployment configuration
├── scripts/
│   └── train_and_export.py                          # Unified training + model export + index.html KB/model sync
├── models/
│   ├── chatcoffee_intent_classifier.joblib          # Sklearn TF-IDF + LogisticRegression artifact (842 features)
│   └── chatcoffee_model_web.json                    # Parity JSON weights + 318 slang rules
├── NLP_Datasets/
│   ├── chatcoffee_gold_850.csv                      # 850-row x 13-col annotated intent & slot corpus
│   ├── unified_coffee_kb.csv                        # 137-row x 19-col cleaned Kopitiam knowledge base
│   └── shortform-dict.json                          # Manglish/BM/EN slang dictionary
├── docs/                                            # Assignment specs, STT evaluation report & generated figures
├── hf_dataset/                                      # Hugging Face dataset card
└── tests/                                           # Pytest API/unit tests + Node.js frontend engine tests
```

---

## Quick Start

### 1. Run Full-Stack Server (FastAPI + Kiosk UI + Voice APIs)
```bash
pip install -r requirements.txt
# Optional: set GROQ_API_KEY in .env for Groq Whisper voice transcription
python app.py
```
Open `http://localhost:8000` in your browser.

### 2. Re-Train & Sync Models + Knowledge Base
```bash
python scripts/train_and_export.py
```

### 3. Run Verification Suite (Python + Node.js)
```bash
python -m pytest -v
```
