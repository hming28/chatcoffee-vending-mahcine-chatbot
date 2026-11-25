import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_feature_2_and_3():
    print("=== Testing Feature 2: Calorie Assessment & Coreference ===")
    
    # 1. No drink in cart, asking "这杯卡路里高不高"
    res1 = client.post("/api/chat", json={"utterance": "这杯卡路里高不高", "state": {}})
    data1 = res1.json()
    print("1. No drink in cart (ZH):", data1["response"])
    assert "您目前还没有选择饮品哦" in data1["response"], "Failed no drink prompt ZH"

    # 2. No drink in cart, asking in English: "is this cup high calorie?"
    res2 = client.post("/api/chat", json={"utterance": "is this cup high calorie?", "state": {}})
    data2 = res2.json()
    print("2. No drink in cart (EN):", data2["response"])
    assert "haven't selected a drink yet" in data2["response"], "Failed no drink prompt EN"

    # 3. No drink in cart, asking in BM: "air ni tinggi kalori ke?"
    res3 = client.post("/api/chat", json={"utterance": "air ni tinggi kalori ke?", "state": {}})
    data3 = res3.json()
    print("3. No drink in cart (BM):", data3["response"])
    assert "belum pilih sebarang minuman" in data3["response"], "Failed no drink prompt BM"

    # 4. Cart has Kopi O Kosong (Tier 1: < 50 kcal), user asks "这杯卡路里高不高"
    res4 = client.post("/api/chat", json={
        "utterance": "这杯卡路里高不高",
        "state": {"base_drink": "Kopi", "sweetness_level": "Zero Sugar", "milk_type": "None", "calories": 24}
    })
    data4 = res4.json()
    print("4. Tier 1 Coref (ZH):", data4["response"])
    assert "第 1 级：极低卡" in data4["response"] or "极低卡" in data4["response"]
    assert "不会发胖" in data4["response"]

    # 5. Cart has Teh O Ais Limau Less Sugar (Tier 2: 50-120 kcal), user asks "is this drink high calorie?"
    res5 = client.post("/api/chat", json={
        "utterance": "is this drink high calorie?",
        "state": {"base_drink": "Teh O Limau", "sweetness_level": "Less Sugar", "temperature": "Iced"}
    })
    data5 = res5.json()
    print("5. Tier 2 Coref (EN):", data5["response"])
    assert "Tier 2: Low-to-Moderate" in data5["response"]

    # 6. Cart has Teh Tarik (Tier 3: 120-180 kcal), user asks "air ni tinggi kalori ke?"
    res6 = client.post("/api/chat", json={
        "utterance": "air ni tinggi kalori ke?",
        "state": {"base_drink": "Teh", "milk_type": "Condensed Milk", "sweetness_level": "Normal"}
    })
    data6 = res6.json()
    print("6. Tier 3 Coref (BM):", data6["response"])
    assert "Tahap 3" in data6["response"]
    assert "susu pekat" in data6["response"]

    # 7. Explicit mention Milo Dinosaur (Tier 4: > 180 kcal), user asks "Milo Dinosaur 会胖吗"
    res7 = client.post("/api/chat", json={
        "utterance": "Milo Dinosaur 会胖吗",
        "state": {}
    })
    data7 = res7.json()
    print("7. Tier 4 Explicit (ZH):", data7["response"])
    assert "第 4 级：偏高能量" in data7["response"]
    assert "增重发胖" in data7["response"]

    print("\n=== Testing Feature 3: Allergen Inquiry & RAG ===")

    # 8. Cart has Kopi (condensed milk), user asks "这杯有乳糖吗？"
    res8 = client.post("/api/chat", json={
        "utterance": "这杯有乳糖吗？",
        "state": {"base_drink": "Kopi", "milk_type": "Condensed Milk"}
    })
    data8 = res8.json()
    print("8. Lactose in Kopi (ZH):", data8["response"])
    assert "含有乳糖" in data8["response"]
    assert "乳糖不耐受" in data8["response"]

    # 9. Cart has Americano, user asks "does this cup have lactose?"
    res9 = client.post("/api/chat", json={
        "utterance": "does this cup have lactose?",
        "state": {"base_drink": "Americano", "milk_type": "None"}
    })
    data9 = res9.json()
    print("9. Lactose free in Americano (EN):", data9["response"])
    assert "lactose-free" in data9["response"]

    # 10. Cart has Milo, user asks "ada gluten ke air ni?"
    res10 = client.post("/api/chat", json={
        "utterance": "ada gluten ke air ni?",
        "state": {"base_drink": "Milo"}
    })
    data10 = res10.json()
    print("10. Gluten in Milo (BM):", data10["response"])
    assert "mengandungi gluten" in data10["response"]
    assert "Protomalt" in data10["response"]

    # 11. General lactose intolerance recommendation (BM): "Saya lactose intolerant, nak minum apa?"
    res11 = client.post("/api/chat", json={
        "utterance": "Saya lactose intolerant, nak minum apa?",
        "state": {}
    })
    data11 = res11.json()
    print("11. Lactose recommendation (BM):", data11["response"])
    assert "bebas laktosa" in data11["response"]
    assert "Kopi O Kosong" in data11["response"]

    # 12. General gluten recommendation (EN): "Any gluten-free drinks?"
    res12 = client.post("/api/chat", json={
        "utterance": "Any gluten-free drinks?",
        "state": {}
    })
    data12 = res12.json()
    print("12. Gluten recommendation (EN):", data12["response"])
    assert "Gluten-Free" in data12["response"]
    assert "Milo" in data12["response"]

    # 13. General lactose recommendation (ZH): "我乳糖不耐受有什么推荐？"
    res13 = client.post("/api/chat", json={
        "utterance": "我乳糖不耐受有什么推荐？",
        "state": {}
    })
    data13 = res13.json()
    print("13. Lactose recommendation (ZH):", data13["response"])
    assert "不含乳糖" in data13["response"]
    assert "Kopi O Kosong" in data13["response"]

    # 14. Recommendation query with active drink in cart (ZH)
    res14 = client.post("/api/chat", json={
        "utterance": "我乳糖不耐受有什么推荐？",
        "state": {"base_drink": "Kopi", "milk_type": "Condensed Milk", "temperature": "Hot", "sweetness_level": "Normal"}
    })
    data14 = res14.json()
    print("14. Lactose rec with Kopi in cart (ZH):", data14["response"])
    assert "为您推荐 100% 不含乳糖" in data14["response"]

    # 15. Recommendation query with active drink in cart (BM)
    res15 = client.post("/api/chat", json={
        "utterance": "Saya lactose intolerant, nak minum apa?",
        "state": {"base_drink": "Kopi", "milk_type": "Condensed Milk", "temperature": "Hot", "sweetness_level": "Normal"}
    })
    data15 = res15.json()
    print("15. Lactose rec with Kopi in cart (BM):", data15["response"])
    assert "ChatCoffee syorkan pilihan 100% bebas laktosa" in data15["response"]

    # 16. Recommendation query with active drink in cart (EN)
    res16 = client.post("/api/chat", json={
        "utterance": "Any gluten-free drinks?",
        "state": {"base_drink": "Kopi", "milk_type": "Condensed Milk", "temperature": "Hot", "sweetness_level": "Normal"}
    })
    data16 = res16.json()
    print("16. Gluten rec with Kopi in cart (EN):", data16["response"])
    assert "ChatCoffee recommends naturally Gluten-Free" in data16["response"]

    # 17. Low calorie recommendation with active drink in cart
    res17 = client.post("/api/chat", json={
        "utterance": "有什么低卡推荐？",
        "state": {"base_drink": "Kopi", "milk_type": "Condensed Milk", "temperature": "Hot", "sweetness_level": "Normal"}
    })
    data17 = res17.json()
    print("17. Low-calorie rec with Kopi in cart:", data17["response"])
    assert "健康低卡推荐" in data17["response"] or "推荐" in data17["response"]

    # 18. Calorie assessment without '高不高' (Chinese keyword detection)
    res18 = client.post("/api/chat", json={
        "utterance": "这杯卡路里多少",
        "state": {"base_drink": "Kopi", "milk_type": "Condensed Milk", "temperature": "Hot", "sweetness_level": "Normal"}
    })
    data18 = res18.json()
    print("18. Calorie query without 高不高 with Kopi in cart:", data18["response"])
    assert "第 3 级" in data18["response"]
    assert "131" in data18["response"]

    print("\nALL BACKEND API TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_feature_2_and_3()
