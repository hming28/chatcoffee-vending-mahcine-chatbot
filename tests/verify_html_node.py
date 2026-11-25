"""Executes index.html NLU/DST JavaScript harness inside Node without leaving stray temp files."""

import os
import subprocess
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_HTML = os.path.join(ROOT_DIR, "index.html")


def run_node_verification():
    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        src = f.read()

    start_idx = src.index("const KOPITIAM_KB =")
    end_idx = src.index("function checkMissingSlotsAndReply(")
    core_engine_src = src[start_idx:end_idx]

    harness = f"""
const assert = require('assert');
global.window = {{ speechSynthesis: {{ getVoices: () => [] }} }};
global.document = {{
  getElementById: () => null,
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: () => {{}},
  documentElement: {{ setAttribute: () => {{}} }}
}};
global.localStorage = {{ getItem: () => null, setItem: () => {{}} }};
global.navigator = {{}};

{core_engine_src}

assert(Array.isArray(KOPITIAM_KB) && KOPITIAM_KB.length === 137, 'Expected 137 KB items, got ' + KOPITIAM_KB.length);
assert.strictEqual(preprocessRojakText('brp?'), 'berapa', 'Punctuation stripped before expansion');
assert.strictEqual(extractDrinkSlots('温度可以改吗').temperature, undefined, '温度 must not trigger Warm');
assert.strictEqual(extractDrinkSlots('RM 4.0').sweetness_level, undefined, 'RM 4.0 must not trigger Zero Sugar');
assert.strictEqual(extractDrinkSlots('i want kopi o 0').sweetness_level, 'Zero Sugar', 'Standalone 0 must trigger Zero Sugar');
assert.strictEqual(extractDrinkSlots('热的不要加奶少甜').milk_type, 'None', '不要加奶 must set None milk');
assert.strictEqual(detectUtteranceLang('nak kopi o kosong satu'), 'bm');
assert.strictEqual(detectUtteranceLang('how many calories in kopi o'), 'en');
assert.strictEqual(detectUtteranceLang('请问美式咖啡多少钱'), 'zh');
const mlRes = predictIntentML(preprocessRojakText('Boss bagi satu kopi peng'));
assert.strictEqual(mlRes.intent, 'order_drink');
console.log('verify_html_node PASS:', KOPITIAM_KB.length, 'KB items |', Object.keys(ROJAK_ML_MODEL.vocabulary).length, 'vocab features');
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        js_file = os.path.join(tmpdir, "verify_bundle.js")
        with open(js_file, "w", encoding="utf-8") as f:
            f.write(harness)
        proc = subprocess.run(["node", js_file], capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(f"Node verification failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        print(proc.stdout.strip())


if __name__ == "__main__":
    run_node_verification()
