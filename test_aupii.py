"""Tests for the Australian hybrid PII system (aupii).

Tiered so the base suite stays green without the hybrid extras:
  * recognizer tests need `presidio_analyzer` + a spaCy model (requirements-hybrid.txt);
  * the end-to-end test additionally needs gliner2 + the local `model.safetensors`.
Missing deps are SKIPPED, never failed.

Run:  python test_aupii.py        (or)   pytest test_aupii.py
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
HAS_PRESIDIO = importlib.util.find_spec("presidio_analyzer") is not None
HAS_MODEL = bool(importlib.util.find_spec("gliner2") and importlib.util.find_spec("torch")
                 and os.path.exists(os.path.join(_HERE, "model.safetensors")))

try:
    import pytest
    if not HAS_PRESIDIO:   # under pytest, skip the whole module cleanly
        pytest.skip("requires hybrid extras (presidio_analyzer); see requirements-hybrid.txt",
                    allow_module_level=True)
except ImportError:
    pytest = None


def test_au_recognizers_detect_valid_ids():
    from aupii import build_analyzer
    a = build_analyzer()
    text = "ABN 51 824 753 556, ACN 004 085 616, tax file number 123 456 782."
    got = {x.entity_type for x in a.analyze(text, language="en", score_threshold=0.5)}
    assert "AU_ABN" in got and "AU_ACN" in got and "AU_TFN" in got, got


def test_presidio_elements_map_to_girp():
    from aupii import build_analyzer, presidio_elements
    a = build_analyzer()
    els = presidio_elements(a, "Visa 4111 1111 1111 1111, email x@y.com, TFN 123 456 782.")
    assert "credit card number" in els and "email address" in els and "tax file number" in els
    assert "credit card number" not in presidio_elements(a, "reference number 4111 1111 1111 1112")


def test_regex_phone_backstop():
    from aupii import _regex_phone
    assert _regex_phone("ph +61 413 394 313 today") == {"phone number"}
    assert _regex_phone("call +1 (415) 555 0132") == {"phone number"}
    assert _regex_phone("order number 12345 shipped") == set()


def test_hybrid_classifies_au_example():
    if not HAS_MODEL:
        msg = "requires gliner2 + local model.safetensors"
        if pytest:
            pytest.skip(msg)
        raise _Skip(msg)
    from aupii import load_hybrid, detect_and_classify_hybrid
    model, analyzer, _ = load_hybrid()
    o = detect_and_classify_hybrid(
        model, analyzer,
        "Patient Mary Citizen, phone 0412 345 678, tax file number 123 456 782, has asthma.")
    assert o["level"] == "Highly Confidential", o
    assert "tax file number" in o["elements"] and "tax file number" in o["structured"]


class _Skip(Exception):
    pass


def _run():
    if not HAS_PRESIDIO:
        print("SKIP aupii tests: presidio_analyzer not installed (pip install -r requirements-hybrid.txt)")
        return
    test_au_recognizers_detect_valid_ids()
    print("OK  Australian recognizers (ABN/ACN/TFN) detect valid IDs")
    test_presidio_elements_map_to_girp()
    print("OK  Presidio elements map to GIRP (Luhn card, email, TFN; non-Luhn rejected)")
    test_regex_phone_backstop()
    print("OK  phone regex backstop")
    try:
        test_hybrid_classifies_au_example()
        print("OK  hybrid classifies an AU example as Highly Confidential")
    except _Skip as e:
        print(f"SKIP hybrid model test: {e}")
    print("\naupii tests passed (skips where optional deps absent).")


if __name__ == "__main__":
    _run()
