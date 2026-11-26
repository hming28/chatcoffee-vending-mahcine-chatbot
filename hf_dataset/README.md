---
language:
- en
- ms
- zh
license: mit
task_categories:
- text-classification
- token-classification
- conversational
pretty_name: Coffee-Rojak Malaysian Kopitiam Multilingual NLU Dataset
size_categories:
- 1K<n<10K
---

# Coffee-Rojak — Malaysian Kopitiam Multilingual NLU Dataset

A beginner-friendly, multilingual dialogue and slot-filling dataset for **Malaysian Kopitiam beverage ordering**.  
It covers **English, Bahasa Melayu, Mandarin (中文), and colloquial Bahasa Rojak / Manglish** (for example: *"I want a champeng kurang manis kurang ice"* or *"Nak kopi o peng siu dai"*).

---

## Quick Start

### Load with `datasets`
```python
from datasets import load_dataset

ds = load_dataset("csv", data_files="coffee-rojak.csv")
print(ds["train"][0])
```

### Load with `pandas`
```python
import pandas as pd

df = pd.read_csv("coffee-rojak.csv")
print(df.shape)        # (2100, 24)
print(df["intent"].value_counts())
```

---

## What Is Inside

| File | Size | Description |
| :--- | :--- | :--- |
| **`coffee-rojak.csv`** | 2,100 rows × 24 columns | Multilingual customer utterances with intent, slot, language-mix, and dialogue-turn labels |
| **`unified_coffee_kb.csv`** | 137 rows × 19 columns | Kopitiam beverage reference table (prices in RM, calories, sugar, caffeine, lactose/gluten flags) |
| **`shortform-dict.json`** | 356 rules | Malaysian Manglish / Rojak shortform and slang normalization dictionary |

---

## Key Columns in `coffee-rojak.csv`

- **Text & Language**: `id`, `category`, `user_utterance`, `language_mix`
- **Intent**: `intent` (14 annotated intents including `order_drink`, `clarify_order`, `modify_order`, `order_confirm`, `cancel_order`, `health_query`, `price_query`, `out_of_scope`, etc.)
- **Drink Slots**: `base_drink`, `temperature`, `temp_detail`, `ice_level`, `sweetness_level`, `milk_type`
- **Dialogue & Metadata**: `slots`, `context`, `old_value`, `new_value`, `query_type`, `turn_id`, `prev_bot_prompt`, `is_coreference`, `shortform_present`, `stt_noise_present`, `ui_action`, `scenario_group`, `split`

---

## License

This dataset is released under the **MIT License**.
