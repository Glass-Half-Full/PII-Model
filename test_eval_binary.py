"""Tests for eval_binary.py — binary PII-present precision curve + headline (offline, no model).

The orchestration is exercised model-free by monkeypatching evaluate.predict_rich / load_gold.
Run directly:  /usr/bin/python3 test_eval_binary.py   ·   or: pytest test_eval_binary.py
"""
import eval_binary as eb


# --- pure helpers --------------------------------------------------------------------------
def test_present_flag_modes():
    # mode A = any PII element present; mode B = level != "Public" (the shipped tier decision)
    assert eb.present_flag({"date of birth"}, "Public", "A") is True    # lone DOB IS a PII element
    assert eb.present_flag({"date of birth"}, "Public", "B") is False   # ...but GIRP keeps it Public
    assert eb.present_flag(set(), "Public", "A") is False
    assert eb.present_flag({"person"}, "Private", "B") is True


def test_flag_prf_counts():
    gold = [True, True, False, False]
    pred = [True, False, True, False]
    m = eb.flag_prf(gold, pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert abs(m["false_flag_rate"] - 0.5) < 1e-9


def test_pick_precision_at_recall_prefers_precision_above_floor():
    curve = [
        {"threshold": 0.3, "precision": 0.70, "recall": 0.99, "f1": 0.82},
        {"threshold": 0.7, "precision": 0.88, "recall": 0.97, "f1": 0.92},
        {"threshold": 0.9, "precision": 0.95, "recall": 0.80, "f1": 0.87},   # recall below floor
    ]
    point, ok = eb.pick_precision_at_recall(curve, 0.95)
    assert ok and point["threshold"] == 0.7      # highest precision among recall >= 0.95


def test_pick_precision_at_recall_floor_unreachable():
    curve = [{"threshold": 0.9, "precision": 0.95, "recall": 0.50, "f1": 0.66},
             {"threshold": 0.3, "precision": 0.60, "recall": 0.80, "f1": 0.69}]
    point, ok = eb.pick_precision_at_recall(curve, 0.97)
    assert not ok and point["threshold"] == 0.3   # floor unreachable -> highest-recall point


def test_best_f1_point():
    curve = [{"threshold": 0.3, "precision": 0.7, "recall": 0.99, "f1": 0.82},
             {"threshold": 0.7, "precision": 0.9, "recall": 0.95, "f1": 0.92}]
    assert eb.best_f1_point(curve)["threshold"] == 0.7


# --- model-free integration ----------------------------------------------------------------
def _fixture():
    import types
    recs = [
        types.SimpleNamespace(id="a", source="kaggle-pii", text="Anna Smith wrote this essay",
                              gold_level="Private", gold_elements=["person"],
                              spans=[types.SimpleNamespace(label="person", start=0, end=10)]),
        # gold-ABSENT row: an order number the model mistakes for a phone (a real false-positive mode)
        types.SimpleNamespace(id="b", source="kaggle-pii", text="order ref 0412345678 shipped",
                              gold_level="Public", gold_elements=[], spans=[]),
        types.SimpleNamespace(id="c", source="tab", text="ring 0412345678 now",
                              gold_level="Private", gold_elements=["phone number"],
                              spans=[types.SimpleNamespace(label="phone number", start=5, end=15)]),
    ]
    gliner = [[("person", 0, 10, 0.9)],
              [("phone number", 10, 20, 0.55)],     # fires at t<=0.5 (FP), gone at t>=0.7
              [("phone number", 5, 15, 0.95)]]
    return recs, gliner


def _run_fixture(**kw):
    recs, gliner = _fixture()
    saved_pr, saved_lg = eb.ev.predict_rich, eb.ev.load_gold
    eb.ev.predict_rich = lambda engine, texts, *a, **k: ({"engine": "hybrid", "device": "cpu"},
                                                         gliner, [[] for _ in texts])
    eb.ev.load_gold = lambda path, limit=None: recs
    try:
        return eb.run("ignored", engine="hybrid", sweep=[0.5, 0.7, 0.9],
                      operating_threshold=0.7, recall_floor=0.9, mode="A",
                      n_boot=50, progress=False, **kw)
    finally:
        eb.ev.predict_rich, eb.ev.load_gold = saved_pr, saved_lg


def test_run_curve_is_threshold_dependent():
    result, _ = _run_fixture()
    curve = {c["threshold"]: c for c in result["curve"]}
    assert sorted(curve) == [0.5, 0.7, 0.9]
    # t=0.5: phone FP on the order-ref row -> precision 2/3; t>=0.7: FP gone -> precision 1.0
    assert abs(curve[0.5]["precision"] - 2 / 3) < 1e-3 and curve[0.5]["recall"] == 1.0
    assert curve[0.7]["precision"] == 1.0 and curve[0.7]["recall"] == 1.0


def test_run_headline_and_breakdowns():
    result, mm = _run_fixture()
    h = result["headline"]
    for k in ("precision_at_recall_floor", "precision_at_operating", "best_f1"):
        assert k in h, k
    assert h["precision_at_operating"]["precision"] == 1.0
    assert set(result["by_source"]) == {"kaggle-pii", "tab"}
    assert "secondary_balanced_accuracy" in result
    assert "binary_precision_ci95" in h["precision_at_operating"]
    # the order-ref FP at t=0.5 is the only binary mismatch direction we expect at operating t? none at 0.7
    assert isinstance(mm, list)


def test_write_report_renders():
    import os
    import tempfile
    result, mm = _run_fixture()
    out = tempfile.mkdtemp()
    p = eb.write_report(result, out, mismatches=mm)
    assert os.path.exists(p) and os.path.exists(os.path.join(out, "metrics.json"))
    txt = open(p).read()
    assert "Binary PII-present" in txt and "by source" in txt.lower()


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
    print(f"\neval_binary: {passed}/{len(fns)} passed.")
    if passed != len(fns):
        raise SystemExit(1)
    print("All eval_binary tests passed.")


if __name__ == "__main__":
    _run()
