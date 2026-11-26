# ChatCoffee — Malaysian Smart Coffee Kiosk Chatbot

An interactive voice and text ordering kiosk for Malaysian Kopitiam drinks.  
It understands **English, Bahasa Melayu, Mandarin (中文), and Bahasa Rojak / Manglish** (e.g., *"I want a champeng kurang manis kurang ice"* or *"Nak kopi o peng siu dai"*).

---

## Quick Start (3 Steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python app.py
```
*(Optional: create a `.env` file with `GROQ_API_KEY=...` to enable Groq Whisper voice transcription.)*

### 3. Open the Kiosk UI
Visit **`http://localhost:8000`** in your browser and start ordering by text or microphone.

---

## What It Can Do

- **8 Core Conversational Intents**:
  1. `order_drink` — Order a drink (e.g., *"Nak kopi peng satu"*)
  2. `clarify_order` — Answer follow-up questions (e.g., *"ais kurang manis"*)
  3. `modify_order` — Change an existing order (e.g., *"tukar jadi teh c"*)
  4. `order_confirm` — Confirm and proceed to checkout (e.g., *"ok confirm"*)
  5. `cancel_order` — Cancel and reset the order (e.g., *"cancel order lah"*)
  6. `health_query` — Ask about calories, sugar, caffeine, or allergens
  7. `price_query` — Ask about drink prices (e.g., *"berapa?"*)
  8. `out_of_scope` — Politely reject unrelated requests
- **Malaysian Kopitiam Slot Extraction (NER)**:
  - Extracts `base_drink`, `temperature` (`Hot`/`Iced`), `temp_detail` (`Hot`/`Warm`), `ice_level` (`Less Ice`/`No Ice`), `sweetness_level` (`Zero Sugar`/`Less Sugar`/`Normal`/`Extra Sugar`), and `milk_type`.
  - Automatically splits concatenated Kopitiam terms (`champeng`, `kopipeng`, `tehc`, `kopio`, `teholimau`) and handles mixed Malay-English ice modifiers (`kurang ice`, `less ais`, `tanpa ice`, `no ais`).
- **Voice Input & Speech Output**:
  - Multilingual speech-to-text (`POST /api/transcribe`) and neural text-to-speech (`POST /api/tts`).

---

## Project Structure

```text
.
├── app.py                                           # FastAPI backend & NLU/DST engine
├── index.html                                       # Kiosk web UI + synced JS NLU engine
├── ChatCoffee_NLP_Model_Training_and_Evaluation.ipynb # Model training & evaluation notebook
├── models/
│   ├── chatcoffee_intent_classifier.joblib          # Trained TF-IDF + LogisticRegression model
│   └── chatcoffee_model_web.json                    # Lightweight JSON weights for web/JS parity
├── NLP_Datasets/
│   ├── coffee-rojak.csv                             # Referenced Kopitiam NLU dataset
│   ├── unified_coffee_kb.csv                        # Beverage menu, nutrition & allergen table
│   └── shortform-dict.json                          # Manglish / BM / EN shortform dictionary
├── scripts/                                         # Dataset & model export utilities
├── tests/                                           # Pytest + Node.js verification suite
├── Dockerfile / Procfile / requirements.txt         # Deployment configuration
└── LICENSE                                          # MIT License
```

---

## Run Tests

```bash
python -m pytest -q
```

---

## Dataset Reference

This application uses the third-party **Coffee-Rojak** Malaysian Kopitiam dataset (`NLP_Datasets/coffee-rojak.csv`) together with `NLP_Datasets/unified_coffee_kb.csv` and `NLP_Datasets/shortform-dict.json` for training and evaluation.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
