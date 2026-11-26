# Project Memory — ChatCoffee Unmanned Smart Kiosk

## Canonical Architecture & Files
- **Backend (`app.py`)**: FastAPI server serving `index.html` (`GET /`), `/api/health`, `/api/chat`, `/api/predict_intent`, `/api/transcribe` (4-track Groq Whisper + domain NLU rescoring + phonetic post-processing), `/api/tts` (Edge-TTS Neural), `/api/slang`, and `/api/learn_slang` (thread-safe JSON persistence to `NLP_Datasets/learned_slang.json`, gitignored).
- **Frontend (`index.html`)**: Single canonical HTML UI & synced client-side NLU/DST engine (`ROJAK_ML_MODEL` + `KOPITIAM_KB`), plus MediaRecorder + VAD + Web Speech fallback.
- **Training Pipeline (`scripts/train_and_export.py`)**: Single source of truth that trains TF-IDF (1,2, `min_df=2`, `sublinear_tf=True`) + LogisticRegression on the 80/20 stratified split (`random_state=42`) of `NLP_Datasets/chatcoffee_gold_850.csv`, logs `LinearSVC` benchmark metrics, exports parity artifacts to `models/chatcoffee_intent_classifier.joblib` and `models/chatcoffee_model_web.json`, and injects synced `ROJAK_ML_MODEL` and `KOPITIAM_KB` into `index.html`.
- **Notebook (`ChatCoffee_NLP_Model_Training_and_Evaluation.ipynb`)**: Fully executed evaluation notebook generating `docs/figures/figure2_model_eval.png` from real held-out test predictions and programmatic NER evaluation on the 250 `direct_order` samples.

## Verified Metrics & Dataset Counts
- **Gold NLU Corpus (`NLP_Datasets/chatcoffee_gold_850.csv`)**: `850` rows × `13` populated columns (all-NaN legacy columns removed; 4 intents: `order_drink` 250, `clarify_order` 200, `modify_order` 200, `health_query` 200).
- **Coffee Knowledge Base (`NLP_Datasets/unified_coffee_kb.csv`)**: `137` deduplicated Kopitiam beverage records × `19` columns (`milk_type` normalized to `"None"`, non-coffee/beer/soda noise purged, Hot & Iced coverage across all 13 kiosk menu bases).
- **Slang Normalization (`NLP_Datasets/shortform-dict.json`)**: `318` clean flattened single-word rules (`shortforms` + `slang_positive` + `slang_negative`, excluding `metadata`/`intensifiers` and protecting `"kau"`). Punctuation is stripped before word expansion (`"brp?"` -> `"brp"` -> `"berapa"`).
- **Intent Classification (80/20 split, `680` train / `170` test, `842` vocab features)**:
  - `LinearSVC`: `98.24%` test accuracy (`0.9812` Macro F1); `99.65%` (`847/850`) full-corpus reference.
  - `LogisticRegression` (deployed): `97.06%` test accuracy (`0.9695` Macro F1); `99.29%` full-corpus reference.
- **Multilingual Voice Pipeline (`docs/multilingual_stt_evaluation_report.md`)**: 36-sample benchmark Model C (4-track parallel + NLU rescoring + phonetic fix) achieved `34.72%` WER and `67.87%` Slot F1.

## Domain Rules & Guardrails
- **Single-Cup Dispenser Limit**: Multi-cup requests (`2 cups`, `dua cawan`, `两杯`, `tambah satu`) are politely blocked until the current cup is confirmed/dispensed; ingredient modifiers (`tambah gula`/`kurang`) are allowed.
- **Milk & Zero-Sugar Constraints**: `Teh O Limau` and `Americano` lock `milk_type="None"`. Requesting `Zero Sugar` / `Kosong` on `Kopi`/`Teh`/`Milo`/`Cham`/`Neslo` without explicit milk auto-couples to `milk_type="None"` (`O Kosong`); explicit condensed-milk drinks clamp `Zero Sugar` -> `Less Sugar`.
- **NER & OOD Precision Guards**: `"温度"` does not trigger `Warm`; `"RM 4.0"` does not trigger `Zero Sugar`; low-confidence utterances without slots (`conf < 0.35`) return fallback instead of misrouting to `health_query`.
- **Verification**: `python -m pytest -v` executes API/unit/parity/audit suites plus Node.js headless checks against `index.html`.
