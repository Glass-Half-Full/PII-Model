"""Deterministic tests for evaluate.py metric math (offline, no model).

Run directly:   python test_evaluate.py
Or pytest:      pytest test_evaluate.py
"""
import evaluate as ev


def test_confusion_and_balanced_accuracy():
    gold = ["Public", "Public", "Private", "Confidential", "Highly Confidential"]
    pred = ["Public", "Private", "Private", "Confidential", "Confidential"]
    cm = ev.confusion(gold, pred)
    assert cm[0][0] == 1 and cm[0][1] == 1          # one Public correct, one leaked to Private
    assert cm[1][1] == 1                            # Private correct
    assert cm[2][2] == 1                            # Confidential correct
    assert cm[3][2] == 1                            # Highly under-classified to Confidential
    bal, per_tier = ev.balanced_accuracy(cm)
    # recalls: Public 1/2, Private 1/1, Confidential 1/1, Highly 0/1 -> mean = 0.625
    assert abs(bal - 0.625) < 1e-9, bal
    assert per_tier["Highly Confidential"] == 0.0


def test_rates_over_under_health():
    gold = ["Highly Confidential", "Private", "Public"]
    pred = ["Confidential", "Confidential", "Public"]   # 1 under (health), 1 over, 1 exact
    r = ev.rates(gold, pred)
    assert abs(r["under"] - 1/3) < 1e-9
    assert abs(r["over"] - 1/3) < 1e-9
    assert r["health_total"] == 1 and abs(r["health_under"] - 1.0) < 1e-9


def test_presence_prf():
    gold = [{"person", "phone number"}, {"email address"}]
    pred = [{"person"}, {"email address", "person"}]
    m = ev.presence_prf(gold, pred)
    # person: tp=1 (row0), fp=1 (row1), fn=0 -> P=.5 R=1
    assert m["person"]["tp"] == 1 and m["person"]["fp"] == 1 and m["person"]["fn"] == 0
    # phone number: fn=1
    assert m["phone number"]["fn"] == 1 and m["phone number"]["tp"] == 0


def test_span_prf_iou_matching():
    gold = [[("person", 0, 10)], [("phone number", 5, 20)]]
    pred = [[("person", 0, 9)],  [("phone number", 50, 60)]]   # row0 overlaps, row1 disjoint
    m = ev.span_prf(gold, pred, iou_thresh=0.5)
    assert m["person"]["tp"] == 1 and m["person"]["fp"] == 0 and m["person"]["fn"] == 0
    # phone: predicted span disjoint -> fp=1, gold unmatched -> fn=1
    assert m["phone number"]["fp"] == 1 and m["phone number"]["fn"] == 1


def test_iou():
    assert ev._iou((0, 10), (0, 10)) == 1.0
    assert ev._iou((0, 10), (10, 20)) == 0.0
    assert abs(ev._iou((0, 10), (5, 15)) - (5 / 15)) < 1e-9


def test_bootstrap_ci_shape_and_bounds():
    gold = ["Public", "Private", "Confidential", "Highly Confidential"] * 10
    pred = ["Public", "Private", "Confidential", "Highly Confidential"] * 10
    ci = ev.bootstrap_ci(gold, pred, n_boot=200, seed=1)
    for k in ("accuracy", "balanced_accuracy", "under"):
        lo, hi = ci[k]
        assert 0.0 <= lo <= hi <= 1.0, (k, lo, hi)
    # perfect predictions -> accuracy CI pinned at 1.0
    assert ci["accuracy"] == [1.0, 1.0]


def test_recommend_threshold_respects_under_cap():
    result = {"sweep": [
        {"threshold": 0.3, "balanced_accuracy": 0.90, "under": 0.20},  # best bal but over cap
        {"threshold": 0.7, "balanced_accuracy": 0.80, "under": 0.05},  # within cap
        {"threshold": 0.9, "balanced_accuracy": 0.70, "under": 0.02},
    ]}
    thr, within = ev.recommend_threshold(result, under_cap=0.10)
    assert thr == 0.7 and within is True
    # if cap unsatisfiable, fall back to max balanced accuracy among all
    thr2, within2 = ev.recommend_threshold(result, under_cap=0.01)
    assert within2 is False and thr2 == 0.3


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
    print(f"\nevaluate metrics: {passed}/{len(fns)} passed.")
    if passed != len(fns):
        raise SystemExit(1)
    print("All evaluate-metric tests passed.")


if __name__ == "__main__":
    _run()
