"""Tests for train_lora.build_training_set Stage-2 enhancements (offline, GPU-free)."""
import json
import os
import tempfile

import train_lora
from girp import GIRP_PII_LABELS


def _hard(rows):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "hard.jsonl")
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_explicit_negative_gets_full_vocab_empty():
    # a confirmed false positive (no real PII) teaches "none of these labels apply here"
    p = _hard([{"id": "n", "text": "order ref 0412345678 shipped", "spans": [],
                "negative": True, "gold_level": "Public"}])
    ex = train_lora.build_training_set(p, augment_n=0, out_path=None, explicit_negatives=True)
    assert len(ex) == 1
    ents = ex[0]["entities"]
    assert set(ents) == set(GIRP_PII_LABELS) and all(v == [] for v in ents.values())


def test_positive_row_keeps_its_entities():
    p = _hard([{"id": "p", "text": "Anna Smith", "gold_level": "Private",
                "spans": [{"label": "person", "start": 0, "end": 10}]}])
    ex = train_lora.build_training_set(p, augment_n=0, out_path=None)
    assert ex[0]["entities"] == {"person": ["Anna Smith"]}


def test_max_chars_skips_long_rows():
    long_text = "x " * 3000      # ~6000 chars
    p = _hard([{"id": "long", "text": long_text, "spans": [], "negative": True, "gold_level": "Public"},
               {"id": "short", "text": "Anna Smith", "gold_level": "Private",
                "spans": [{"label": "person", "start": 0, "end": 10}]}])
    ex = train_lora.build_training_set(p, augment_n=0, out_path=None, max_chars=2000)
    texts = [e["text"] for e in ex]
    assert "Anna Smith" in texts and long_text not in texts   # the long negative is dropped


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"OK  {fn.__name__}")
        except Exception as e:
            print(f"XX  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\ntrain_lora: {passed}/{len(fns)} passed.")
    if passed != len(fns):
        raise SystemExit(1)
    print("All train_lora tests passed.")


if __name__ == "__main__":
    _run()
