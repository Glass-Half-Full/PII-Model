"""Tests for synthetic.py's additive span-labelled path (offline, no model).

Run directly:   python test_synthetic_spans.py
Or pytest:      pytest test_synthetic_spans.py
"""
from girp import classify_elements, GIRP_PII_LABELS
from synthetic import generate_synthetic_dataset


def test_unlabeled_path_unchanged():
    """return_spans defaults to False and yields the original DataFrame[text, expected]."""
    df = generate_synthetic_dataset(12, seed=0)
    assert list(df.columns) == ["text", "expected"], list(df.columns)
    # deterministic for a seed
    df2 = generate_synthetic_dataset(12, seed=0)
    assert df["text"].tolist() == df2["text"].tolist()


def test_spans_path_returns_rows_with_spans():
    rows = generate_synthetic_dataset(40, seed=1, return_spans=True)
    assert isinstance(rows, list) and rows
    for row in rows:
        assert set(row) >= {"text", "expected", "spans"}, row.keys()


def test_span_slices_match_and_labels_valid():
    rows = generate_synthetic_dataset(80, seed=2, return_spans=True)
    for row in rows:
        text = row["text"]
        for label, start, end in row["spans"]:
            assert 0 <= start < end <= len(text), (label, start, end, text)
            assert text[start:end].strip(), (label, start, end, text)
            assert label in GIRP_PII_LABELS, label


def test_span_labels_yield_expected_tier():
    """The GIRP tier derived from the span labels must equal the row's intended tier."""
    rows = generate_synthetic_dataset(120, seed=3, return_spans=True)
    for row in rows:
        labels = {label for label, _, _ in row["spans"]}
        assert classify_elements(labels) == row["expected"], (row["expected"], sorted(labels), row["text"])


def test_au_specific_coverage_present():
    """AU-critical elements that public datasets lack must all appear in a reasonable batch."""
    rows = generate_synthetic_dataset(200, seed=4, return_spans=True)
    seen = {label for row in rows for label, _, _ in row["spans"]}
    for required in ["tax file number", "medicare number", "bank account number",
                     "phone number", "address", "person", "health condition"]:
        assert required in seen, f"missing AU/required element: {required}; saw {sorted(seen)}"


def test_spans_deterministic():
    a = generate_synthetic_dataset(30, seed=7, return_spans=True)
    b = generate_synthetic_dataset(30, seed=7, return_spans=True)
    assert [r["text"] for r in a] == [r["text"] for r in b]
    assert [r["spans"] for r in a] == [r["spans"] for r in b]


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
    print(f"\nsynthetic spans: {passed}/{len(fns)} passed.")
    if passed != len(fns):
        raise SystemExit(1)
    print("All synthetic-span tests passed.")


if __name__ == "__main__":
    _run()
