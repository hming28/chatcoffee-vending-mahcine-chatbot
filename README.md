# ☕ ChatCoffee: Malaysian Vending Machine Smart Kiosk
### TAR UMT — BMDS2123 Natural Language Processing (Topic 4 Project)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-chatcoffee.onrender.com-00C781?style=for-the-badge&logo=render&logoColor=white)](https://chatcoffee.onrender.com/)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset%20Benchmark-ffcc00?style=for-the-badge)](https://huggingface.co/datasets/HM28/malaysian-coffee-ordering-nlu)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat)](https://creativecommons.org/licenses/by/4.0/)

---

> ### 🌐 Try It Live! (No Setup Required)
> Experience the smart kiosk directly in your browser on desktop or mobile:  
> 👉 **[https://chatcoffee.onrender.com/](https://chatcoffee.onrender.com/)** 👈

---

## 📖 What is ChatCoffee?

**ChatCoffee** is an autonomous, voice-first smart coffee ordering kiosk designed specifically for authentic Malaysian multi-lingual environments (traditional **Kopitiams**, Mamak stalls, and food courts).

In Malaysia, ordering coffee naturally involves **Bahasa Rojak / Manglish** — a seamless mix of English, Bahasa Melayu, and Chinese dialects (Cantonese & Hokkien), accompanied by local abbreviations (`nk`, `xnak`, `brp`, `tapau`) and terms (`Peng`, `Siu Dai`, `Kosong`, `Kau`). 

Conventional single-language bots fail on these mixed expressions. **ChatCoffee solves this with supervised machine learning, slang normalization, and a live nutritional knowledge base.**

---

## 🎮 Quick Test: Try These Phrases on the Live Demo!

When you visit **[chatcoffee.onrender.com](https://chatcoffee.onrender.com/)**, try typing or speaking any of these natural Malaysian orders:

| Scenario | What to Say / Type | What ChatCoffee Does |
| :--- | :--- | :--- |
| **☕ Authentic Kopitiam Order** | `"Boss, bagi I Kopi O Kosong Peng satu, tapau."` | Identifies Iced Black Coffee, Zero Sugar, Takeaway, and quotes **RM 4.00**. |
| **🔄 Mid-Order Change (Correction)** | `"Wait boss, actually tukar jadi Teh Tarik panas."` | Seamlessly switches from cold coffee to hot tea and recalculates price. |
| **🥗 Health & Nutritional Query** | `"Ada drink apa yg zero sugar and low calories ah?"` | Queries the RAG knowledge base and recommends low-calorie, zero-sugar drinks. |
| **🗣️ Heavy Chat Slang & Shortform** | `"nk order cham peng siu dai bungkus ya boss"` | Normalizes `nk` $\rightarrow$ `nak`, recognizes `siu dai` (less sugar), and packs it. |
| **❓ Incomplete / Ambiguous Order** | `"Nak minum kopi la boss."` | Proactively asks: *"Mau Panas ke Peng (Ais), dan manis standard ke Siu Dai?"* |

---

## ✨ Key Features & Capabilities

- 🗣️ **Dual-Script & Slang Normalization**: Cleans and standardizes 341+ Malaysian abbreviations (`nk` $\rightarrow$ `nak`, `brp` $\rightarrow$ `berapa`) while preserving CJK Chinese characters.
- 🧠 **Supervised Machine Learning Intent Classifier**: Zero rule-based guessing. Powered by a champion **Logistic Regression with TF-IDF (1-2 N-Gram)** model trained on 850 curated gold instances (100% test accuracy across 4 balanced intents).
- 🎯 **Dual-Script Named Entity Recognition (NER)**: Extracts beverage type, temperature (Hot/Cold), sweetness level (Kosong, Siu Dai, Normal), milk type (Condensed, Evaporated, None), and dining option (Dine In / Tapau).
- 🔄 **Stateful Dialogue State Tracking (DST)**: Proactively gathers missing attributes and robustly handles mid-order human corrections (grounded in the ACM Barista Task 7 benchmark).
- 📊 **Nutritional RAG Knowledge Base**: Backed by a verified 378-beverage database (`unified_coffee_kb.csv`) with calories, sugar (g), caffeine (mg), and live Malaysian Ringgit (RM) pricing.
- 🎙️ **Voice-First Experience**: Built-in speech recognition and neural speech synthesis for hands-free ordering.

---

## 📂 Project Structure

```text
├── index.html                                          # Interactive Smart Kiosk Web UI (Client-Side ML & Voice)
├── app.py                                              # FastAPI backend server hosting REST API & Dialogue Engine
├── ChatCoffee_NLP_Model_Training_and_Evaluation.ipynb  # Complete Jupyter Notebook (EDA, Data Cleaning, Model Benchmarks)
├── requirements.txt                                    # Production Python dependencies
├── Dockerfile                                          # Docker container deployment configuration
├── Procfile                                            # Cloud PaaS (Render) deployment entrypoint
├── models/
│   ├── chatcoffee_intent_classifier.joblib             # Scikit-Learn serialized model for Python backend
│   └── chatcoffee_model_web.json                       # Mathematical vector weights for Client Browser JS
└── NLP_Datasets/
    ├── chatcoffee_gold_850.csv                         # 850-instance Gold NLU Benchmark Dataset
    ├── unified_coffee_kb.csv                           # 378-drink Grounded Nutritional Knowledge Base
    └── shortform-dict.json                             # 341 Malaysian slang normalization rules
```

---

## 🛠️ Local Development & Quick Start

If you want to run ChatCoffee locally on your computer:

### 1. Clone the Repository
```bash
git clone https://github.com/hming28/chatcoffee-vending-machine-chatbot.git
cd chatcoffee-vending-machine-chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python app.py
```
Open **`http://localhost:8000`** in your browser to start ordering!

*(Alternatively, you can simply double-click `index.html` to run the standalone client-side kiosk directly in any browser without needing Python!)*

---

## 🌐 API Endpoints

The FastAPI backend exposes the following RESTful endpoints:

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Serves the interactive Smart Kiosk Web UI |
| `GET` | `/api/health` | System health check, active model version, and dataset metrics |
| `POST` | `/api/predict_intent` | Raw ML classification with Softmax class probabilities |
| `POST` | `/api/chat` | End-to-end multi-turn conversational dialogue turn |
| `POST` | `/api/tts` | Neural speech synthesis endpoint |
| `POST` | `/api/transcribe` | Audio file transcription endpoint |
| `POST` | `/api/learn_slang` | Persistent Human-in-the-Loop slang learning (`learned_slang.json`) |
| `GET` | `/api/slang` | Inspect active and user-taught slang normalization rules |

---

## 📜 Academic Research & Open Datasets

This project is part of the **TAR UMT BMDS2123 Natural Language Processing** curriculum and follows open science principles:

- **Hugging Face Dataset**: [HM28/malaysian-coffee-ordering-nlu](https://huggingface.co/datasets/HM28/malaysian-coffee-ordering-nlu)
- **Theoretical Benchmarks**:
  - *Irfan et al. (ACM/IEEE HRI 2020)*: The Barista Robot (Conversational Ordering & Mid-Order Corrections).
  - *Budzianowski et al. (EMNLP 2018)*: MultiWOZ (Task-Oriented Dialogue Modelling).
  - *Tan (2018)*: Kopitiam Lexicon Ontology (`alvations/kopitiam`).
  - *Yusof (2021)*: Malaysia Manglish & Malay NLP Normalization (`ZafranYusof/malaysia-manglish-nlp`).
