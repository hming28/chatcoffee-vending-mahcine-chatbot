"""Automated 7-point QA validator for NLP_Datasets/chatcoffee_v1_5_2100.csv."""

import json
import os
import re
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(BASE_DIR, "NLP_Datasets", "chatcoffee_v2_1.csv")
DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH


def main() -> int:
    errors = []

    # 1. UTF-8 BOM check
    with open(DATASET_PATH, "rb") as f:
        bom = f.read(3)
    if bom != b"\xef\xbb\xbf":
        errors.append(f"[BOM] Expected UTF-8 BOM (ef bb bf), got {bom!r}")

    # 2. Load with keep_default_na=False so "None" is preserved as string
    df = pd.read_csv(DATASET_PATH, keep_default_na=False)
    if len(df) < 1000:
        errors.append(f"[RowCount] Expected >= 1000 rows, got {len(df)}")

    # 3. JSON slots vs flat columns 100% parity
    slot_cols = ["base_drink", "temperature", "temp_detail", "ice_level", "sweetness_level", "milk_type"]
    for idx, row in df.iterrows():
        try:
            d = json.loads(row["slots"]) if row["slots"].strip() else {}
        except Exception as exc:
            errors.append(f"[JSON] Row {row['id']} invalid JSON slots: {exc}")
            continue
        for k in slot_cols:
            fv = str(row[k]).strip()
            sv = "" if k not in d or d[k] is None else str(d[k]).strip()
            if fv != sv:
                errors.append(f"[Parity] Row {row['id']} col '{k}': flat={fv!r} != json={sv!r}")

    # 4. Uniqueness check
    norm_u = df["user_utterance"].astype(str).str.lower().str.strip()
    if norm_u.nunique() != len(df):
        dups = df[norm_u.duplicated(keep=False)]["user_utterance"].tolist()
        errors.append(f"[Duplicates] Found {len(dups)} duplicate utterances: {dups[:5]}")

    if df["id"].nunique() != len(df):
        errors.append("[ID] Duplicate IDs detected")

    # 5. Domain constraints check (no Takeaway, no Starbucks, old_value != new_value)
    if "serving_type" in df.columns:
        errors.append("[Constraint] serving_type column should not exist")
    for forbidden in ("starbucks", "takeaway", "dine-in"):
        hits = df[df["user_utterance"].str.lower().str.contains(forbidden, na=False)]
        if not hits.empty:
            errors.append(f"[Forbidden] Term {forbidden!r} found in rows: {hits['id'].tolist()}")

    same_mod = df[(df["old_value"] != "") & (df["old_value"] == df["new_value"])]
    if not same_mod.empty:
        errors.append(f"[Modify] old_value == new_value in rows: {same_mod['id'].tolist()}")

    # 6. Zero scenario_group leakage across splits
    splits = df.groupby("scenario_group")["split"].nunique()
    leaked = splits[splits > 1]
    if not leaked.empty:
        errors.append(f"[Leakage] scenario_group spans multiple splits: {leaked.index.tolist()[:5]}")
    if not set(df["split"].unique()).issubset({"train", "validation", "test"}):
        errors.append(f"[Split] Unexpected split values: {df['split'].unique().tolist()}")

    # 7. Language tag sanity check
    for _, row in df.iterrows():
        has_zh = bool(re.search(r"[\u4e00-\u9fff]", row["user_utterance"]))
        if has_zh and "zh-cmn" not in row["language_mix"]:
            errors.append(f"[Lang] Row {row['id']} has CJK characters but language_mix={row['language_mix']!r}")

    if errors:
        print(f"QA VALIDATION FAILED ({len(errors)} errors):")
        for e in errors[:20]:
            print(" ", e)
        return 1

    print("ALL 7 QA CHECKS PASSED!")
    print(f"Rows: {len(df)} | Intents: {df['intent'].nunique()} | Splits: {df['split'].value_counts().to_dict()}")
    print("Intent distribution:", df["intent"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
