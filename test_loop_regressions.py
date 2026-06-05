"""Regression tests grown by the recursive-improvement loop (offline, no model).

Every Stage-1 fix the loop makes lands here FIRST as a failing case, then the minimal edit to
girp.py / aupii.py makes it green. Pure-function and deterministic, like test_girp.py /
test_failures.py. Also guards the Stage-2 training pool schema so the trainer never chokes.

Run directly:   python test_loop_regressions.py
Or pytest:      pytest test_loop_regressions.py
"""
import json
import os

from girp import GIRP_PII_LABELS, classify_elements, is_valid_entity

# (elements set, expected GIRP level, why) — GIRP rule regressions found by the loop.
GIRP_CASES = []

# (label, value, expected_is_valid, why) — validation regressions found by the loop.
VALIDATION_CASES = [
    # iter-001: "Citizen" is a real surname (AU placeholder "John Citizen"); the person filter
    # wrongly rejected any name containing it, causing dangerous person under-classification.
    ("person", "Raj Citizen", True, "iter1: Citizen is a valid surname; must not be filtered"),
    ("person", "Jane Citizen", True, "iter1: AU placeholder name must validate as person"),
    ("person", "senior citizen", False, "iter1: generic role phrase still filtered via 'senior'"),
    ("person", "the manager", False, "iter1: role phrase still filtered (no regression)"),
]


def test_low_precision_birthplace_maiden_suppressed_not_removed():
    """iter-001: birthplace / mother's maiden name have ~0 zero-shot precision (they fire on any
    place name in real ai4privacy/gretel text) and drove Public over-classification. They are kept
    in the EXTRACTION label set (removing them perturbs zero-shot person detection — measured
    -1.7pp) but SUPPRESSED from the final element set, and kept in the GIRP rule engine."""
    import aupii
    # still extracted (so the zero-shot label-set context is unchanged)...
    assert "birthplace" in aupii.GLINER_FUZZY_LABELS
    assert "mother's maiden name" in aupii.GLINER_FUZZY_LABELS
    # ...but suppressed from the decision, and still in the rule vocabulary
    assert aupii.SUPPRESSED_FUZZY_LABELS == frozenset({"birthplace", "mother's maiden name"})
    assert "birthplace" in GIRP_PII_LABELS and "mother's maiden name" in GIRP_PII_LABELS


def test_girp_cases():
    for elements, expected, why in GIRP_CASES:
        got = classify_elements(elements)
        assert got == expected, f"{sorted(elements)} -> {got!r}, expected {expected!r} ({why})"


def test_validation_cases():
    for label, value, expected, why in VALIDATION_CASES:
        got = is_valid_entity(label, value)
        assert got == expected, f"is_valid_entity({label!r}, {value!r}) = {got}, expected {expected} ({why})"


def test_hard_examples_schema():
    """Every row in the Stage-2 training pool must be valid for train_lora.load_examples."""
    path = "data/hard_examples.jsonl"
    if not os.path.exists(path):
        return  # nothing accumulated yet
    vocab = set(GIRP_PII_LABELS)
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert "text" in row and "spans" in row, f"line {ln}: missing text/spans"
            text = row["text"]
            for s in row["spans"]:
                assert {"label", "start", "end"} <= set(s), f"line {ln}: span missing keys"
                assert 0 <= s["start"] < s["end"] <= len(text), f"line {ln}: bad offsets {s}"
                # negatives may carry an empty/other label; positives must be in vocabulary
                if not row.get("negative"):
                    assert s["label"] in vocab, f"line {ln}: label {s['label']!r} not in GIRP vocab"


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
    print(f"\nloop regressions: {passed}/{len(fns)} passed "
          f"({len(GIRP_CASES)} girp cases, {len(VALIDATION_CASES)} validation cases).")
    if passed != len(fns):
        raise SystemExit(1)
    print("All loop-regression tests passed.")


if __name__ == "__main__":
    _run()
