"""Deterministic tests for the gold-record schema (offline, no model).

Run directly (no pytest needed):   python test_gold_schema.py
Or under pytest:                    pytest test_gold_schema.py
"""
from gold_data.schema import (
    GoldSpan, GoldRecord, build_record, to_jsonl, from_jsonl, validate, make_id,
    SCHEMA_VERSION,
)


def _phone_record():
    text = "Call Jane Smith on +61 412 345 678."
    raw = [
        ("person", 5, 15, "FIRSTNAME"),          # "Jane Smith"
        ("phone number", 19, 34, "PHONENUMBER"),  # "+61 412 345 678"
    ]
    return text, raw


def test_build_record_derives_level_and_elements():
    text, raw = _phone_record()
    rec = build_record(text, raw, source="unit", split="test")
    # name + phone (a combination element) -> Confidential
    assert rec.gold_level == "Confidential", rec.gold_level
    assert rec.gold_elements == ("person", "phone number"), rec.gold_elements
    # span.text convenience matches the slice
    assert rec.spans[0].text == "Jane Smith"
    assert rec.spans[1].text == "+61 412 345 678"
    assert rec.source == "unit" and rec.split == "test"
    assert rec.id == make_id("unit", text)
    assert rec.schema_version == SCHEMA_VERSION


def test_lone_name_is_private():
    text = "Regards, Jane Smith"
    rec = build_record(text, [("person", 9, 19, "NAME")], source="unit", split="test")
    assert rec.gold_level == "Private", rec.gold_level


def test_public_when_no_spans():
    rec = build_record("The office is closed Monday.", [], source="unit", split="test")
    assert rec.gold_level == "Public"
    assert rec.gold_elements == ()


def test_roundtrip_jsonl():
    text, raw = _phone_record()
    rec = build_record(text, raw, source="unit", split="test")
    line = to_jsonl(rec)
    assert "\n" not in line
    back = from_jsonl(line)
    assert back == rec, (back, rec)


def test_validate_passes_for_wellformed():
    text, raw = _phone_record()
    rec = build_record(text, raw, source="unit", split="test")
    assert validate(rec) == []


def test_validate_catches_out_of_bounds_offset():
    text, raw = _phone_record()
    rec = build_record(text, raw, source="unit", split="test")
    bad = GoldRecord(
        id=rec.id, text=rec.text,
        spans=(GoldSpan("phone number", 19, 9999, "x", ""),),
        gold_elements=("phone number",), gold_level="Private",
        source="unit", split="test",
    )
    problems = validate(bad)
    assert any("offset" in p.lower() for p in problems), problems


def test_validate_catches_unknown_label():
    bad = GoldRecord(
        id="x", text="hello world",
        spans=(GoldSpan("not_a_real_label", 0, 5, "x", "hello"),),
        gold_elements=("not_a_real_label",), gold_level="Public",
        source="unit", split="test",
    )
    problems = validate(bad)
    assert any("label" in p.lower() for p in problems), problems


def test_validate_catches_tampered_level():
    text, raw = _phone_record()
    rec = build_record(text, raw, source="unit", split="test")
    tampered = GoldRecord(
        id=rec.id, text=rec.text, spans=rec.spans,
        gold_elements=rec.gold_elements,
        gold_level="Public",  # wrong: should be Confidential
        source=rec.source, split=rec.split,
    )
    problems = validate(tampered)
    assert any("gold_level" in p for p in problems), problems


def test_train_lora_compatible_spans():
    """from_jsonl spans must expose label/start/end for train_lora.load_examples."""
    text, raw = _phone_record()
    rec = build_record(text, raw, source="unit", split="test")
    d = __import__("json").loads(to_jsonl(rec))
    assert "text" in d and "spans" in d
    for s in d["spans"]:
        assert {"label", "start", "end"} <= set(s.keys()), s


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
    print(f"\ngold schema: {passed}/{len(fns)} passed.")
    if passed != len(fns):
        raise SystemExit(1)
    print("All gold-schema tests passed.")


if __name__ == "__main__":
    _run()
