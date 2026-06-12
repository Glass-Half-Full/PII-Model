"""Tests for real-free-text ingestion (Text Anonymization Benchmark) + build_real.

Offline, no model, no network (the one ingester that needs the network is exercised with a
stubbed ``_iter_raw``). Run directly:  /usr/bin/python3 test_real_data.py   ·   or: pytest
"""
import gold_data.ingest as ingest
from gold_data import mappings


# --- TAB annotation flattening (TAB ships ~12 parallel annotations per doc) ----------------
def test_tab_mentions_uses_quality_checked_annotator():
    annotations = {
        "annotator1": {"entity_mentions": [
            {"entity_type": "PERSON", "start_offset": 0, "end_offset": 4},
            {"entity_type": "ORG", "start_offset": 10, "end_offset": 20},
        ]},
        "annotator2": {"entity_mentions": [
            {"entity_type": "PERSON", "start_offset": 0, "end_offset": 9},   # different boundary
        ]},
    }
    out = ingest._tab_mentions(annotations, quality_checked=["annotator1"])
    assert out == [{"label": "PERSON", "start": 0, "end": 4},
                   {"label": "ORG", "start": 10, "end": 20}]


def test_tab_mentions_deterministic_fallback_first_annotator():
    annotations = {
        "annotator2": {"entity_mentions": [{"entity_type": "PERSON", "start_offset": 5, "end_offset": 9}]},
        "annotator1": {"entity_mentions": [{"entity_type": "PERSON", "start_offset": 0, "end_offset": 4}]},
    }
    out = ingest._tab_mentions(annotations, quality_checked=None)   # -> first by sorted name
    assert out == [{"label": "PERSON", "start": 0, "end": 4}]


def test_tab_mentions_empty():
    assert ingest._tab_mentions({}, None) == []


# --- TAB label mapping (conservative: only PERSON is a GIRP element) -----------------------
def test_tab_mapping_keeps_person_drops_coarse():
    assert mappings.map_label("tab", "PERSON") == "person"
    for coarse in ("ORG", "LOC", "DEM", "DATETIME", "QUANTITY", "MISC", "CODE"):
        assert mappings.is_known("tab", coarse), coarse
        assert mappings.map_label("tab", coarse) is None, coarse


def test_tab_spans_from_maps_person_and_drops_rest():
    text = "Anna Smith sued the Court on 2004-01-01."
    entries = [
        {"label": "PERSON", "start": 0, "end": 10},      # Anna Smith -> person
        {"label": "ORG", "start": 20, "end": 25},        # Court -> dropped
        {"label": "DATETIME", "start": 29, "end": 39},   # date -> dropped
    ]
    raw_spans, unknown = ingest._spans_from(text, entries, "tab")
    assert unknown == []
    assert raw_spans == [("person", 0, 10, "PERSON")]


def test_build_real_is_test_only_and_counts_binary():
    """build_real writes ONLY test.jsonl, forces split='test', and counts PII present/absent."""
    import os
    import tempfile
    import gold_data.build_real as build_real
    from gold_data.schema import build_record, from_jsonl

    recs = [
        build_record("Anna Smith v. The State — a long judgment naming the applicant.",
                     [("person", 0, 10)], "tab", ""),
        build_record("an essay about climate change with no personal information at all", [], "tab", ""),
        build_record("you can reach me at a@b.com any time", [("email address", 20, 27)], "tab", ""),
    ]
    saved = ingest.ingest_tab
    ingest.ingest_tab = lambda n, split="test", max_scan=None: ((r, []) for r in recs)
    out = tempfile.mkdtemp()
    try:
        manifest = build_real.build("real-test", kaggle_path=None, n_kaggle=0, n_tab=10, out_dir=out)
    finally:
        ingest.ingest_tab = saved

    vdir = os.path.join(out, "real-test")
    assert os.path.exists(os.path.join(vdir, "test.jsonl"))
    assert not os.path.exists(os.path.join(vdir, "train.jsonl"))   # never trainable
    assert not os.path.exists(os.path.join(vdir, "dev.jsonl"))
    loaded = [from_jsonl(l) for l in open(os.path.join(vdir, "test.jsonl"))]
    assert loaded and all(r.split == "test" for r in loaded)
    assert manifest["counts"]["total"] == 3
    assert manifest["counts"]["pii_present"] == 2   # person + email
    assert manifest["counts"]["pii_absent"] == 1    # the no-PII essay (the precision-measuring row)


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
    print(f"\nreal-data ingest: {passed}/{len(fns)} passed.")
    if passed != len(fns):
        raise SystemExit(1)
    print("All real-data ingest tests passed.")


if __name__ == "__main__":
    _run()
