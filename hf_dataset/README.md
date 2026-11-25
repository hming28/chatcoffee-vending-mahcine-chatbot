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
pretty_name: ChatCoffee Malaysian Kopitiam Bilingual Dialogue & Knowledge Base
size_categories:
- n<1K
---

# ChatCoffee Bilingual Malaysian Kopitiam Dialogue & Knowledge Base

Domain-specific multilingual (Bahasa Melayu, English, Mandarin 中文, and colloquial Bahasa Rojak / Manglish) NLU corpus and Kopitiam beverage knowledge base built for an unmanned smart coffee vending kiosk (**ChatCoffee**).

## Dataset Summary

1. **`chatcoffee_gold_850.csv`** (`850` rows × `13` columns)
   - Annotated customer utterances across 4 task-oriented dialogue intents:
     - `order_drink` (`250` direct single-shot orders with full slot annotations)
     - `clarify_order` (`200` ambiguous / multi-turn clarification queries)
     - `modify_order` (`200` mid-order human corrections)
     - `health_query` (`200` nutrition, calorie, sugar, caffeine, and lactose/gluten allergen QAs)
   - Slot columns: `base_drink`, `temperature`, `sweetness_level`, `milk_type`, `query_type`.

2. **`unified_coffee_kb.csv`** (`137` rows × `19` columns)
   - Grounded Malaysian Kopitiam menu knowledge base covering Hot and Iced beverages across `Kopi`, `Kopi O`, `Kopi C`, `Teh Tarik`, `Teh O`, `Teh C`, `Teh O Limau`, `Cham`, `White Coffee`, `Milo`, `Milo O`, `Milo C`, `Neslo`, and `Americano`.
   - Includes Malaysian market pricing (`price_rm`), nutrition (`calories`, `total_fat_g`, `carbohydrates_g`, `sugars_g`, `caffeine_mg`), and allergen flags (`contains_lactose`, `contains_gluten`, `allergens_summary`).

3. **`shortform-dict.json`**
   - Bilingual Malaysian Manglish / Rojak shortform & slang normalization lexicon (`318` flattened runtime rules across `shortforms`, `slang_positive`, `slang_negative`).
