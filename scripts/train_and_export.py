"""Unified training, evaluation, and export pipeline for ChatCoffee NLU (v1.5, 14 intents).

Trains the dual-script (Malay/English/Chinese) TF-IDF (1,2) + LogisticRegression
classifier on NLP_Datasets/chatcoffee_v1_5_2100.csv using its leak-free scenario_group
split (train+validation -> test evaluation, with fallback to chatcoffee_gold_850.csv),
verifies LinearSVC benchmark comparison on the same split, evaluates shortform & STT-noise
subsets, and exports identical model artifacts to:
  - models/chatcoffee_intent_classifier.joblib
  - models/chatcoffee_model_web.json
Also synchronizes ROJAK_ML_MODEL and KOPITIAM_KB inside index.html.
"""

import json
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V15_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "chatcoffee_v1_5_2100.csv")
GOLD_PATH = V15_PATH if os.path.exists(V15_PATH) else os.path.join(BASE_DIR, "NLP_Datasets", "chatcoffee_gold_850.csv")
KB_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "unified_coffee_kb.csv")
SHORTFORM_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "shortform-dict.json")
JOBLIB_PATH = os.path.join(BASE_DIR, "models", "chatcoffee_intent_classifier.joblib")
WEB_JSON_PATH = os.path.join(BASE_DIR, "models", "chatcoffee_model_web.json")
HTML_PATH = os.path.join(BASE_DIR, "index.html")


def load_flat_slang_dict(path: str) -> dict[str, str]:
    """Flatten shortform-dict.json (shortforms + slang_positive + slang_negative)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    flat: dict[str, str] = {}
    for section in ("shortforms", "slang_positive", "slang_negative"):
        block = raw.get(section, {})
        if isinstance(block, dict):
            for k, v in block.items():
                if isinstance(k, str) and isinstance(v, str):
                    kl = k.strip().lower()
                    vl = v.strip().lower()
                    # Protect kopitiam modifier "kau" and English keywords ("on", "confirm")
                    if kl and vl and kl not in ("kau", "on", "confirm"):
                        flat[kl] = vl
    return flat


def preprocess_text(text: str, slang_dict: dict[str, str]) -> str:
    """Lowercase, separate CJK-Latin boundaries, strip punctuation BEFORE word expansion, expand Manglish, split CJK."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"([\u4e00-\u9fff])([a-z0-9])", r"\1 \2", t)
    t = re.sub(r"([a-z0-9])([\u4e00-\u9fff])", r"\1 \2", t)
    t = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", t)
    for phrase in sorted((k for k in slang_dict if " " in k), key=len, reverse=True):
        t = re.sub(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", slang_dict[phrase], t)
    words = [slang_dict.get(w, w) for w in t.split()]
    t = " ".join(words)
    t = re.sub(r"([\u4e00-\u9fff])", r" \1 ", t)
    return re.sub(r"\s+", " ", t).strip()


def derive_kb_base(drink_name: str) -> str:
    """Map full KB drink_name to the canonical base used by frontend KOPITIAM_KB."""
    n = drink_name.lower()
    if "limau" in n or "lemon" in n:
        return "Teh O Limau"
    if "white coffee" in n:
        return "White Coffee"
    if "americano" in n:
        return "Americano"
    if "neslo" in n:
        return "Neslo"
    if "cham" in n:
        return "Cham"
    if "milo" in n or "tak kiu" in n:
        return "Milo"
    if "teh" in n:
        return "Teh"
    return "Kopi"


def build_frontend_kb(kb_df: pd.DataFrame) -> list[dict]:
    records = []
    for _, r in kb_df.iterrows():
        lac = str(r["contains_lactose"]).strip().lower() == "true"
        glu = str(r["contains_gluten"]).strip().lower() == "true"
        milk = str(r["milk_type"]).strip() or "None"
        records.append({
            "id": str(r["drink_id"]).strip(),
            "name": str(r["drink_name"]).strip(),
            "base": derive_kb_base(str(r["drink_name"])),
            "temp": str(r["temperature"]).strip(),
            "sweet": str(r["sweetness_level"]).strip(),
            "milk": milk,
            "price": round(float(r["price_rm"]), 2),
            "cal": int(float(r["calories"])),
            "sugar": float(r["sugars_g"]),
            "caf": float(r["caffeine_mg"]),
            "contains_lactose": lac,
            "contains_gluten": glu,
            "allergens_summary": str(r["allergens_summary"]).strip(),
            "lactose": lac,
            "gluten": glu,
        })
    return records


def sync_index_html(web_payload: dict, kb_records: list[dict]) -> None:
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Replace const KOPITIAM_KB = [ ... ];
    kb_lines = ["    const KOPITIAM_KB = ["]
    for i, rec in enumerate(kb_records):
        comma = "," if i < len(kb_records) - 1 else ""
        kb_lines.append("      " + json.dumps(rec, ensure_ascii=False) + comma)
    kb_lines.append("    ];")
    kb_block = "\n".join(kb_lines)

    kb_pattern = re.compile(r"    const KOPITIAM_KB = \[[\s\S]*?\n    \];")
    if not kb_pattern.search(html):
        raise RuntimeError("Could not locate KOPITIAM_KB block in index.html")
    html = kb_pattern.sub(lambda _: kb_block, html, count=1)

    # 2. Replace let/const ROJAK_ML_MODEL = { ... };
    model_line = "    let ROJAK_ML_MODEL = " + json.dumps(web_payload, ensure_ascii=False) + ";"
    model_pattern = re.compile(r"    (?:const|let) ROJAK_ML_MODEL = \{[^\n]*\};")
    if not model_pattern.search(html):
        raise RuntimeError("Could not locate ROJAK_ML_MODEL line in index.html")
    html = model_pattern.sub(lambda _: model_line, html, count=1)

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    slang_dict = load_flat_slang_dict(SHORTFORM_PATH)
    df = pd.read_csv(GOLD_PATH, keep_default_na=False)
    df["clean_text"] = df["user_utterance"].apply(lambda x: preprocess_text(x, slang_dict))

    if "split" in df.columns and set(df["split"].unique()).issuperset({"train", "test"}):
        train_mask = df["split"].isin(["train", "validation"])
        test_mask = df["split"] == "test"
        X_train, y_train = df.loc[train_mask, "clean_text"], df.loc[train_mask, "intent"]
        X_test, y_test = df.loc[test_mask, "clean_text"], df.loc[test_mask, "intent"]
        test_df = df.loc[test_mask].copy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df["clean_text"],
            df["intent"],
            test_size=0.2,
            stratify=df["intent"],
            random_state=42,
        )
        test_df = df.loc[X_test.index].copy()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Deployable probabilistic classifier (calibrated probabilities for FastAPI & JS)
    lr_model = LogisticRegression(
        max_iter=1000,
        C=4.0,
        class_weight="balanced",
        random_state=42,
    )
    lr_model.fit(X_train_vec, y_train)
    lr_preds = lr_model.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average="macro")

    # Non-probabilistic margin benchmark
    svc_model = LinearSVC(class_weight="balanced", random_state=42)
    svc_model.fit(X_train_vec, y_train)
    svc_preds = svc_model.predict(X_test_vec)
    svc_acc = accuracy_score(y_test, svc_preds)
    svc_f1 = f1_score(y_test, svc_preds, average="macro")

    print(f"Dataset: {GOLD_PATH} ({len(df)} rows, {df['intent'].nunique()} intents) | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"TF-IDF Vocabulary Size: {len(vectorizer.vocabulary_)} features | Slang rules: {len(slang_dict)}")
    print(f"LogisticRegression Test Accuracy: {lr_acc:.4f} | Macro F1: {lr_f1:.4f}")
    print(f"LinearSVC          Test Accuracy: {svc_acc:.4f} | Macro F1: {svc_f1:.4f}")

    if "shortform_present" in test_df.columns:
        sf_mask = (test_df["shortform_present"] == "yes").values
        if sf_mask.any():
            print(f"Shortform Subset (n={sf_mask.sum()}) -> LR Acc: {accuracy_score(y_test.values[sf_mask], lr_preds[sf_mask]):.4f} | SVC Acc: {accuracy_score(y_test.values[sf_mask], svc_preds[sf_mask]):.4f}")
    if "stt_noise_present" in test_df.columns:
        stt_mask = (test_df["stt_noise_present"] == "yes").values
        if stt_mask.any():
            print(f"STT Noise Subset (n={stt_mask.sum()}) -> LR Acc: {accuracy_score(y_test.values[stt_mask], lr_preds[stt_mask]):.4f} | SVC Acc: {accuracy_score(y_test.values[stt_mask], svc_preds[stt_mask]):.4f}")

    print("\nLogisticRegression Classification Report:")
    print(classification_report(y_test, lr_preds, digits=4))

    classes = [str(c) for c in lr_model.classes_]
    model_name = "LogisticRegression (TF-IDF N-Gram 1-2, Dual-Script CJK, 14-Intent v1.5)"

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": lr_model,
            "classes": classes,
            "model_name": model_name,
            "slang_rules_count": len(slang_dict),
            "vocab_size": len(vectorizer.vocabulary_),
        },
        JOBLIB_PATH,
    )

    vocab_sorted = {str(k): int(v) for k, v in sorted(vectorizer.vocabulary_.items(), key=lambda kv: kv[1])}
    web_payload = {
        "model_name": model_name,
        "classes": classes,
        "vocabulary": vocab_sorted,
        "idf": [round(float(x), 6) for x in vectorizer.idf_],
        "coef": [[round(float(c), 6) for c in row] for row in lr_model.coef_],
        "intercept": [round(float(b), 6) for b in lr_model.intercept_],
        "ngram_range": list(vectorizer.ngram_range),
        "sublinear_tf": bool(vectorizer.sublinear_tf),
        "slang_dict": slang_dict,
    }
    with open(WEB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(web_payload, f, ensure_ascii=False)

    kb_df = pd.read_csv(KB_PATH, keep_default_na=False)
    kb_records = build_frontend_kb(kb_df)
    sync_index_html(web_payload, kb_records)
    print(f"Exported {JOBLIB_PATH} & {WEB_JSON_PATH} (exact 1:1 parity)")
    print(f"Synced index.html: ROJAK_ML_MODEL ({len(vocab_sorted)} features) + KOPITIAM_KB ({len(kb_records)} items)")


if __name__ == "__main__":
    main()
