# ChatCoffee NLP Assignment: Comprehensive Project Memory & Architecture Ledger

**Course**: TAR UMT — BMDS2123 Natural Language Processing  
**Topic**: Topic 4 — Rojak/Mixed Language Chatbot for Unmanned Coffee Vending Machine (*ChatCoffee*, formerly *RojakBrew*)  
**Evaluation Rubric**: 90% Group Report & Prototype + 10% Live Presentation (100% Total)  
**Last Updated**: 2026-11-22  

---

## 1. Project Overview & Scenario Identity

### 1.1 The Business & NLP Problem
In Malaysia, traditional food & beverage ordering takes place in multi-lingual, multi-dialectal environments (Kopitiam, Mamak stalls, food courts). Customers naturally code-mix English, Bahasa Melayu, and Chinese dialects (Cantonese, Hokkien) alongside distinctive discourse particles (`lah`, `meh`, `lor`, `leh`) and non-standard chat abbreviations (`nk`, `brp`, `xnak`, `tapau`).

Off-the-shelf single-language NLP models (such as standard English BERT or Indonesian IndoBERT) suffer severe Out-Of-Vocabulary (OOV) failure and grammatical parsing breakdown when handling Malaysian **Bahasa Rojak / Manglish**.

### 1.2 Proposed Intelligent Solution: ChatCoffee
**ChatCoffee** is an autonomous, unmanned smart coffee vending kiosk featuring:
1. **Dual-Script Orthographic Normalization**: Unifies Latin slang abbreviations and preserves Chinese characters without substring collision.
2. **Supervised Intent Classification**: Multi-class machine learning models (Naive Bayes, Logistic Regression, Linear SVM, Random Forest) trained on TF-IDF word unigrams and bigrams ($N = 850$).
3. **Dual-Script Named Entity Recognition (NER)**: Extracts beverage base, temperature/ice level, sweetness, and milk type from mixed colloquial speech.
4. **Deterministic Dialogue State Tracker (DST)**: Robust state machine managing proactive slot gathering, mid-order human corrections, dynamic temperature switching, and Kopitiam zero-sugar auto-coupling.
5. **Grounded Local Knowledge Base & RAG Engine**: 144 authentic Malaysian Kopitiam classic beverages (`unified_coffee_kb.csv`) with laboratory-verified nutrition (calories, sugars, fats, caffeine, allergens: lactose, gluten) from the Malaysian Food Composition Database (MyFCD, MOH Malaysia) and live Malaysian Ringgit (RM) pricing.

---

## 2. Dataset Ecosystem & Engineering Ledger (Consolidated 5-Dataset Suite)

To ensure 100% authentic local relevance and eliminate foreign coffee chain noise, the dataset architecture was streamlined into a focused **5-Dataset Suite** (removing the external 242-item Starbucks Kaggle dataset).

| Index | Dataset Name | Scale & Files | Upstream Source & License | Role in System |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **`ChatCoffee-Gold-850`** | 850 instances<br>• `chatcoffee_gold_850.csv`<br>• `rojak_brew_gold_850.csv`<br>• `chatcoffee_gold_850.jsonl` | Self-Curated Gold Standard<br>Hugging Face: [`HM28/malaysian-coffee-ordering-nlu`](https://huggingface.co/datasets/HM28/malaysian-coffee-ordering-nlu)<br>License: CC BY 4.0 | Primary NLU training & evaluation benchmark. Balanced across 4 categories: direct order (250), ambiguous (200), mid-order correction (200), health/nutrition QA (200). |
| **2** | **`Unified Coffee Domain KB`** | 144 drinks (106 Hot, 38 Iced) × 19 features<br>• `unified_coffee_kb.csv` | Harmonized from GitHub [`alvations/kopitiam`](https://github.com/alvations/kopitiam) + Malaysian Food Composition Database (MyFCD 2015, Ministry of Health Malaysia / KKM)<br>License: CC BY 4.0 | RAG knowledge base for health queries, calorie 4-tier evaluation, allergen safety checks (lactose & gluten), and live kiosk RM checkout billing. |
| **3** | **`Manglish Shortform Dict`** | 343 rules<br>• `shortform-dict.json`<br>• `shortform_dict.csv` | GitHub: [`ZafranYusof/malaysia-manglish-nlp`](https://github.com/ZafranYusof/malaysia-manglish-nlp) & [`huseinzol05/Malaya`](https://github.com/huseinzol05/Malaya)<br>License: MIT | Slang normalization dictionary in Step 4. Expands `nk` $\rightarrow$ `nak`, `brp` $\rightarrow$ `berapa`, `x` $\rightarrow$ `tak`, `xnak` $\rightarrow$ `tak nak`. |
| **4** | **`Kopitiam Lexicon`** | 168 entries<br>• `kopitiam.tsv`<br>• `kopitiam.csv` | GitHub: [`alvations/kopitiam`](https://github.com/alvations/kopitiam) (Liling Tan)<br>License: MIT | Domain ontology defining physical beverage properties (`Peng`, `Kosong`, `Siu Dai`, `C`, `O`). |
| **5** | **`Barista Dialogue Corpus`** | 7,764 turns<br>• `barista_task7_corrections_sample.txt`<br>• `barista_task7_dialogues.csv` | GitHub: [`birfan/BaristaDatasets`](https://github.com/birfan/BaristaDatasets) (ACM/IEEE HRI 2020)<br>License: CC BY-NC-SA 4.0 | Architectural benchmark for multi-turn dialogue state machine, mid-order correction, and customer conversational flow. |

### Consolidated Clean Package
All 5 datasets are organized in the directory:
- **Local Path**: `NLP_Datasets/` (Total footprint: **< 1.0 MB**).
- **Google Drive Ready**: Can be uploaded directly to `My Drive/NLP_Datasets/` or `My Drive/NLP Ass/`.

---

## 3. Machine Learning & Experimental Results

### 3.1 Google Colab Execution Log
- **Environment**: Python 3.10+ (Google Colab CPU/GPU).
- **Vectorization Pipeline**: CountVectorizer + TfidfVectorizer (Word unigrams + bigrams, sublinear TF scaling).
- **Dataset Partitioning**: 80% Train (680 samples) / 20% Test (170 samples), Stratified split across all 4 intent classes.

### 3.2 Intent Classifier Benchmark Leaderboard (N = 850, Test Set = 170)
| Rank | Classifier Model | Vectorizer Configuration | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Linear Support Vector Classifier (`LinearSVC`)** | TF-IDF (1,2) N-Grams | **98.24%** | **0.9832** | **0.9825** | **0.9828** | Champion Model |
| 🥈 | **Logistic Regression (`LogisticRegression`)** | TF-IDF (1,2) N-Grams | **97.65%** | **0.9778** | **0.9762** | **0.9770** | Deployed Web Engine |
| 🥉 | **Multinomial Naive Bayes (`MultinomialNB`)** | CountVectorizer (1,2) | 94.71% | 0.9510 | 0.9462 | 0.9481 | Baseline |
| 4 | **Random Forest (`RandomForestClassifier`, 100 Trees)** | TF-IDF (1,2) N-Grams | 93.53% | 0.9380 | 0.9345 | 0.9358 | Ensemble |

### 3.3 Classification Performance by Intent Class (Balanced 4-Class Architecture)
- **`order_drink`** (Direct order): Precision = 0.98, Recall = 0.98, F1 = 0.98 (Support = 50)
- **`clarify_order`** (Ambiguous multi-turn): Precision = 0.97, Recall = 0.97, F1 = 0.97 (Support = 40)
- **`modify_order`** (Mid-order human correction): Precision = 0.97, Recall = 0.97, F1 = 0.97 (Support = 40)
- **`health_query`** (Nutritional & allergen QA): Precision = 1.00, Recall = 0.97, F1 = 0.99 (Support = 40)
- **Overall Macro F1**: **0.9770**; **Weighted F1**: **0.9765**.

---

## 4. Key Architectural Decisions & Strict Constraints Implemented

1. **Zero Inline Data in Code Cells**:
   - Replaced all hardcoded synthetic datasets with external loaders (`pd.read_csv()` / `pd.read_json()`).
   - Slang dictionary dynamically reads `shortform-dict.json` instead of inline Python dictionaries.
2. **Strict Academic Cleanliness**:
   - Exactly zero occurrences of the words `"practical"` or `"syllabus"` in all notebooks and scripts.
3. **Storage & Cloud Persistence**:
   - Configured Colab auto-search paths for Google Drive persistence (`/content/drive/MyDrive/NLP_Datasets/`).
4. **Removal of Starbucks Dataset & Knowledge Base Grounding (378 $\rightarrow$ 144 items)**:
   - Purged all 242 Specialty Cafe (Starbucks) records to eliminate irrelevant US cafe noise.
   - Retained 136 authentic Kopitiam Classic items and supplemented 8 missing core permutations:
     * `Teh O Limau Panas` (Normal, Siu Dai, Kosong — 3 Hot items)
     * `Teh O Ais Limau` (Normal, Siu Dai, Kosong — 3 Iced items)
     * `Americano` (Hot Americano, Iced Americano — 2 items)
   - Total: exactly **144 beverages** (106 Hot, 38 Iced), fully attributed to `alvations/kopitiam` + Malaysian Food Composition Database (MyFCD 2015, Ministry of Health Malaysia / KKM).
5. **Removal of Kopi Gau / Kau / Po (Strength) Feature**:
   - Commercial automated coffee vending machines operate on fixed powder-to-water brewing extraction formulas.
   - Removing Strength eliminated unnecessary complexity and streamlined the DST inspector to 4 core slots (`base_drink`, `temperature`, `sweetness_level`, `milk_type`) plus contextual ice/warm modifiers.
6. **Removal of Redundant `price_comparison` Feature**:
   - Removed Starbucks/Zus price comparison from NLU classification and dialogue logic.
   - Converted all category 4 instances into pure health & nutritional RAG queries (`health_query`) for a clean, perfectly balanced 4-class intent architecture (`order_drink`: 250, `clarify_order`: 200, `modify_order`: 200, `health_query`: 200). Local Kopitiam price queries are handled cleanly without cross-chain comparisons.
7. **Removal of Dine-In / Takeaway (`serving_type` / `bungkus` / `tapau`)**:
   - Unmanned vending kiosks strictly dispense single cups into a physical pickup bay. Dine-in seating does not apply.
8. **Single-Cup Physical Vending Machine Limit**:
   - Replaced multi-cup quantity accumulation on `tambah` / `tambah lagi` with a polite notification enforcing single-cup dispensing per transaction (*"Mesin ini bancuh 1 cawan setiap pesanan boss! Sila sahkan & ambil cawan ini dulu ya"*).
9. **Context-Aware Dynamic Temp & Ice Slider (Plan C)**:
   - Replaced contradictory sliders with a single context-sensitive vertical modifier bar:
     * Hot selected: toggles between **Hot** (standard hot) and **Warm** (`Suam`).
     * Iced selected: toggles between **Normal Ice**, **Less Ice** (`Kurang Ais`), and **No Ice** (`Tanpa Ais`).
   - Synchronized with SVG animated cup graphics (steam opacity for hot/warm, 0-3 ice cubes for cold).
10. **Multilingual Language Mirroring NLG**:
    - The chatbot automatically identifies the customer's language (English, Bahasa Melayu, or Chinese) and replies cleanly in that language, eliminating awkward bilingual parentheticals like *"Mau Hot (Panas) atau Iced (Sejuk)?"*.
11. **Smart Kopitiam Zero-Sugar Auto-Coupling**:
    - Condensed milk contains ~55% sucrose; true zero-sugar is physically impossible with condensed milk.
    - When a customer requests Zero Sugar (`kosong` / `no sugar`) on `Kopi` or `Teh`, the DST automatically couples `milk_type` to `None` (`Kopi O Kosong` / `Teh O Kosong`), avoiding dead-end sugar lockouts.
12. **Scientific 4-Tier Calorie Classification & Coreference Resolution**:
    - Contextual coreference: When the customer asks *"这杯卡路里高不高？"*, *"is this cup high calorie?"*, or *"air ni tinggi kalori ke?"*, the system automatically resolves *"这杯"* to the currently selected drink in the cart (`state.base_drink`).
    - Calorie 4-tier evaluation classifier:
      * **Tier 1 (< 50 kcal)**: Very Low / 极低卡 / Sangat Rendah (e.g. Kopi O Kosong 24 kcal, Americano 10 kcal).
      * **Tier 2 (50 - 120 kcal)**: Low-to-Moderate / 适中微甜 / Rendah-Sederhana (e.g. Kopi C Siew Dai 86 kcal, Teh O Ais Limau 88 kcal).
      * **Tier 3 (120 - 180 kcal)**: Standard Kopitiam / 常规偏高 / Sedang-Tinggi (e.g. Kopi 131 kcal, Teh Tarik 131 kcal, Cham 131 kcal). Explains condensed milk calories and suggests Siew Dai.
      * **Tier 4 (> 180 kcal)**: High / 偏高能量 / Tinggi (e.g. Milo Dinosaur 280 kcal).
13. **Food Science Allergen RAG (Lactose & Gluten)**:
    - Grounded allergen attributes across all 144 items (`contains_lactose`, `contains_gluten`, `allergens_summary`):
      * **Lactose**: Condensed milk & evaporated milk contain lactose; black coffee/tea (`milk_type: None`) are 100% dairy & lactose-free.
      * **Gluten**: Milo and Neslo contain Protomalt (barley malt extract) and contain gluten; pure coffee and tea are 100% gluten-free.
    - Supports specific drink inquiries as well as general allergen-safe recommendations (e.g. recommending Kopi O Kosong, Teh O, and Americano for lactose intolerance).
14. **Clean Menu Card Consistency & Parity**:
    - Renamed `Cham (鸳鸯)` to `Cham` for consistent English/Malay tile labels.
    - Fixed Milo O base price calculation so it accurately reflects RM 3.20 (Hot) / RM 4.00 (Iced).
    - Fixed-recipe drinks (White Coffee, Neslo, Cham) automatically inherit `Condensed Milk` without redundant milk prompts.

---

## 5. File Synchronization & Deliverables Matrix

All modifications are kept in 100% byte-for-byte synchronization across workspace mirrors:

| File Path | Description | Status |
| :--- | :--- | :---: |
| `NLP_Datasets/unified_coffee_kb.csv` | 144-drink grounded Kopitiam knowledge base (106 Hot, 38 Iced) | ✅ Verified (144 Rows) |
| `app.py` | FastAPI backend server with 4-tier calorie evaluation & allergen RAG | ✅ 18/18 Tests Passed |
| `chatcoffee_malaysian_unmanned_smart_kiosk.html` | Standalone interactive kiosk frontend with live ML telemetry | ✅ Verified |
| `github/chatcoffee_malaysian_unmanned_smart_kiosk.html` | GitHub repository mirror (origin/main) | ✅ Synced |
| `github/index.html` | GitHub Pages live deployment mirror | ✅ Synced |
| `d:\This PC\Downloads\chatcoffee_malaysian_unmanned_smart_kiosk_bmds2123_nlp (10).html` | Downloads folder mirror | ✅ Synced |
| `README.md` & `github/README.md` | Overhauled project documentation (144 items, 5 datasets) | ✅ Synced |
| `ChatCoffee_NLP_Model_Training_and_Evaluation.ipynb` | Academic deliverable notebook (N=850 intact) | ✅ Ready |
