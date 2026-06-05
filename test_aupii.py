"""Tests for the Australian hybrid PII system (aupii).

Requires the hybrid extras installed: presidio-analyzer + spaCy model (+ gliner2 for the
model-based test). Run:  python test_aupii.py
"""
from aupii import build_analyzer, presidio_elements, PRESIDIO2GIRP, _regex_phone


def test_au_recognizers_detect_valid_ids():
    a = build_analyzer()
    text = "ABN 51 824 753 556, ACN 004 085 616, tax file number 123 456 782."
    got = {x.entity_type for x in a.analyze(text, language="en", score_threshold=0.5)}
    assert "AU_ABN" in got, got
    assert "AU_ACN" in got, got
    assert "AU_TFN" in got, got


def test_presidio_elements_map_to_girp():
    a = build_analyzer()
    els = presidio_elements(a, "Visa 4111 1111 1111 1111, email x@y.com, TFN 123 456 782.")
    assert "credit card number" in els     # Luhn-valid
    assert "email address" in els
    assert "tax file number" in els
    # a non-Luhn 16-digit number is NOT a card (checksum rejects it)
    els2 = presidio_elements(a, "reference number 4111 1111 1111 1112")
    assert "credit card number" not in els2


def test_regex_phone_backstop():
    assert _regex_phone("ph +61 413 394 313 today") == {"phone number"}
    assert _regex_phone("call +1 (415) 555 0132") == {"phone number"}
    assert _regex_phone("order number 12345 shipped") == set()


def test_hybrid_classifies_au_example():
    from aupii import load_hybrid, detect_and_classify_hybrid
    model, analyzer, _ = load_hybrid()
    o = detect_and_classify_hybrid(
        model, analyzer,
        "Patient Mary Citizen, phone 0412 345 678, tax file number 123 456 782, has asthma.")
    assert o["level"] == "Highly Confidential", o
    assert "tax file number" in o["elements"]
    assert "tax file number" in o["structured"]   # came from Presidio's checksum recognizer


def _run():
    test_au_recognizers_detect_valid_ids()
    print("OK  Australian recognizers (ABN/ACN/TFN) detect valid IDs")
    test_presidio_elements_map_to_girp()
    print("OK  Presidio elements map to GIRP (Luhn card, email, TFN; non-Luhn rejected)")
    test_regex_phone_backstop()
    print("OK  phone regex backstop")
    test_hybrid_classifies_au_example()
    print("OK  hybrid classifies an AU example as Highly Confidential")
    print("\nAll aupii tests passed.")


if __name__ == "__main__":
    _run()
