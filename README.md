# ChatCoffee (v3.0) — Malaysian Smart Coffee Kiosk Chatbot

An interactive, multi-turn **voice and text ordering kiosk** for Malaysian Kopitiam beverages.  
It understands **English, Bahasa Melayu, Mandarin (中文), and colloquial Bahasa Rojak / Manglish** (for example: *"I want a champeng kurang manis kurang ice"* or *"Nak kopi o peng siu dai"*).

---

## Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python app.py
```
*(Optional: create a `.env` file with `GROQ_API_KEY=...` to enable Groq Whisper multilingual voice transcription.)*

### 3. Open the Kiosk UI
Open **`http://localhost:8000`** in your browser and order via text or microphone.

---

## v3.0 Architecture & Key Capabilities

### 1. 8-Intent ML Classifier (1:1 Backend + Browser Parity)
The deployed **TF-IDF (1–2 n-gram, Dual-Script CJK) + LogisticRegression** model (`1,494` vocabulary features, `356` slang normalization rules) is synchronized across:
- `models/chatcoffee_intent_classifier.joblib` (scikit-learn pipeline)
- `models/chatcoffee_model_web.json` (lightweight JSON weights)
- `ROJAK_ML_MODEL` embedded in `index.html` (offline/standalone browser inference)

Trained on the **8 core kiosk intents** (`1,600` filtered utterances from `NLP_Datasets/coffee-rojak.csv`):

| Intent | Training Rows | Purpose | Example Utterance |
| :--- | :---: | :--- | :--- |
| `order_drink` | 350 | Start or place a drink order | *"I want a champeng kurang manis kurang ice"* |
| `clarify_order` | 300 | Answer follow-up slot prompts | *"ais kurang manis"* |
| `modify_order` | 300 | Change an existing drink/modifier | *"tukar jadi teh c"* |
| `health_query` | 280 | Allergen, nutrition, or caffeine QA | *"Does cham contain milk or caffeine?"* |
| `price_query` | 100 | Drink price & surcharge lookup | *"How much is kopi o peng?"* / *"brp?"* |
| `order_confirm` | 90 | Confirm order and proceed to checkout | *"ok confirm order"* |
| `cancel_order` | 90 | Cancel and reset the current order | *"cancel order lah"* |
| `out_of_scope` | 90 | Reject unrelated / off-topic input | *"The air in this room is very cold today"* |

---

### 2. Dual-Script NER, Compound Splitting & Kopitiam DST
Implemented identically in `app.py` (`clean_text` / `extract_slots`) and `index.html` (`preprocessRojakText` / `extractDrinkSlots`):

- **Concatenated Kopitiam Compound Splitting** (`split_kopitiam_compounds` / `splitKopitiamCompounds`):
  - Splits fused local terms before slang expansion and slot matching:
    `champeng` → `cham peng`, `kopipeng` → `kopi peng`, `tehc` → `teh c`, `kopio` → `kopi o`, `teholimau` → `teh o limau`, `siudai` → `siu dai`, `siewdai` → `siew dai`, `gahdai` → `gah dai`.
- **Extracted Slots**:
  - `base_drink`: `Kopi`, `Teh`, `Cham`, `White Coffee`, `Milo`, `Neslo`, `Teh O Limau`, `Americano`
  - `temperature` & `temp_detail`: `Hot` (`Hot` / `Warm` / `suam`) or `Iced`
  - `ice_level`: `Less Ice` / `No Ice` — supports English, Malay, Chinese, and **mixed Malay-English** modifiers (`kurang ice`, `less ais`, `tanpa ice`, `no ais`, `kosong ice`, `sikit ice`, `少冰`, `去冰`)
  - `sweetness_level`: `Zero Sugar`, `Less Sugar`, `Normal`, `Extra Sugar`
  - `milk_type`: `Condensed Milk`, `Evaporated Milk`, `None`
- **Unmanned Vending Machine Rules**:
  - Enforces a **1-cup-per-order** dispenser limit.
  - Auto-locks `milk_type = "None"` for `Teh O Limau` and `Americano`.
  - Auto-couples `Kosong` / `Zero Sugar` milk rules and prevents invalid condensed-milk zero-sugar combinations.

---

### 3. Grounded Knowledge Base & Voice Pipeline
- **Knowledge Base** (`NLP_Datasets/unified_coffee_kb.csv`, `137` rows × `19` columns):
  - Verified Malaysian market pricing (`RM`), calories, fat, carbohydrates, sugar, caffeine, and allergen flags (`contains_lactose`, `contains_gluten`).
- **4-Track Multilingual ASR** (`POST /api/transcribe`):
  - Runs parallel Groq Whisper (`auto`, `ms`, `en`, `zh`) with domain NLU rescoring (`score_asr_candidate`) and phonetic correction (`apply_stt_phonetic_fix_py`), plus browser Web Speech fallback.
- **Neural TTS** (`GET / POST /api/tts`):
  - Synthesizes Malay (`ms-MY`), Mandarin (`zh-CN`), and English (`en-SG`) audio via Microsoft Edge-TTS.

---

## REST API Endpoints (`app.py`)

| Method & Path | Description |
| :--- | :--- |
| `GET /` | Serves the interactive Kiosk UI (`index.html`) |
| `GET /api/health` | Service health and loaded model/KB status |
| `POST /api/chat` | Multi-turn NLU + Dialogue State Tracking endpoint |
| `POST /api/predict_intent` | Inspect raw normalized text, predicted intent, and extracted slots |
| `POST /api/transcribe` | Multilingual audio transcription with domain rescoring |
| `GET / POST /api/tts` | Neural text-to-speech audio generation |
| `GET /api/slang` | List current Manglish/Rojak slang normalization rules |
| `POST /api/learn_slang` | Teach new runtime slang mappings |
| `GET /models/chatcoffee_model_web.json` | Serve the web-parity JSON model |

---

## Project Structure

```text
.
├── app.py                                           # FastAPI backend, NLU/NER & Dialogue State Tracker
├── index.html                                       # Single-file Kiosk UI + synced JS NLU/DST engine
├── ChatCoffee_NLP_Model_Training_and_Evaluation.ipynb # Training, evaluation & export notebook
├── models/
│   ├── chatcoffee_intent_classifier.joblib          # Sklearn TF-IDF + LogisticRegression (8 intents)
│   └── chatcoffee_model_web.json                    # Web-parity JSON weights + 356 slang rules
├── NLP_Datasets/
│   ├── coffee-rojak.csv                             # Referenced 2,100-row x 24-col Kopitiam NLU dataset
│   ├── chatcoffee_gold_850.csv                      # Baseline 950-row corpus for dataset tests/scripts
│   ├── unified_coffee_kb.csv                        # 137-row x 19-col beverage & nutrition KB
│   └── shortform-dict.json                          # Manglish / BM / EN shortform dictionary
├── scripts/
│   ├── build_v1_5_dataset.py                        # Dataset builder utility
│   ├── train_and_export.py                          # Unified training + model export + index.html sync
│   └── validate_v1_dataset.py                       # Dataset schema & integrity validator
├── tests/                                           # Pytest API/unit tests + Node.js frontend suites
├── Dockerfile / Procfile / requirements.txt         # Container & PaaS deployment files
└── LICENSE                                          # MIT License
```

---

## Running Tests & Training

### Run Full Verification Suite (Python + Node.js)
```bash
python -m pytest -q
```

### Re-Train & Sync Models + Knowledge Base
```bash
python scripts/train_and_export.py
```

---

## Dataset Reference

This application uses the third-party **Coffee-Rojak** Malaysian Kopitiam dataset (`NLP_Datasets/coffee-rojak.csv`, `2,100` rows × `24` columns across `14` annotated intents, filtered to the `8` core kiosk intents for v3.0 training) together with `NLP_Datasets/unified_coffee_kb.csv` and `NLP_Datasets/shortform-dict.json`.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
