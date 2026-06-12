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


def test_binary_flag_metrics():
    # binary "PII present" flag = (level != "Public"); P/R of the present/absent decision
    gold = ["Public", "Public", "Private", "Confidential", "Highly Confidential"]
    pred = ["Public", "Private", "Public", "Confidential", "Confidential"]
    # gold flag: F F T T T   ;   pred flag: F T F T T
    #   row0 TN · row1 FP · row2 FN · row3 TP · row4 TP   -> tp=2 fp=1 fn=1 tn=1
    m = ev.binary_flag_metrics(gold, pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert abs(m["binary_precision"] - 2 / 3) < 1e-3
    assert abs(m["binary_recall"] - 2 / 3) < 1e-3
    assert abs(m["false_flag_rate"] - 0.5) < 1e-9   # fp / (fp + tn) among gold-absent rows


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


def test_bootstrap_ci_includes_binary():
    gold = ["Public", "Private", "Confidential", "Highly Confidential"] * 10
    pred = list(gold)   # perfect predictions
    ci = ev.bootstrap_ci(gold, pred, n_boot=200, seed=1)
    for k in ("binary_precision", "binary_recall"):
        lo, hi = ci[k]
        assert 0.0 <= lo <= hi <= 1.0, (k, lo, hi)
    # perfect predictions -> binary precision & recall pinned at 1.0
    assert ci["binary_precision"] == [1.0, 1.0] and ci["binary_recall"] == [1.0, 1.0]


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


def test_derive_honors_per_label_thresholds():
    """Lever A: a per-label threshold (empty by default) raises the bar for specific weak labels."""
    import aupii
    saved = aupii.PER_LABEL_THRESHOLDS
    aupii.PER_LABEL_THRESHOLDS = {"person": 0.9}     # demand high confidence for person
    try:
        text = "Anna Smith and 0412345678"
        # person at conf 0.75 < its 0.9 threshold -> dropped; phone at 0.75 >= default 0.7 -> kept
        el, _lvl, _sp = ev.derive([("person", 0, 10, 0.75), ("phone number", 15, 25, 0.75)],
                                  [], text, 0.7, "hybrid")
        assert "person" not in el and "phone number" in el
    finally:
        aupii.PER_LABEL_THRESHOLDS = saved


def test_aupii_filtered_confident_applies_per_label():
    """Lever A in the production twins: keep entity strings whose confidence clears the per-label bar."""
    import aupii
    saved = aupii.PER_LABEL_THRESHOLDS
    aupii.PER_LABEL_THRESHOLDS = {"person": 0.9}
    try:
        text = "Anna Smith calls 0412345678"
        ents = {"person": [{"start": 0, "end": 10, "confidence": 0.75}],          # below 0.9 -> dropped
                "phone number": [{"start": 17, "end": 27, "confidence": 0.75}]}    # >= default 0.7 -> kept
        out = aupii._filtered_confident(text, ents, 0.7)
        assert "person" not in out and out["phone number"] == ["0412345678"]
    finally:
        aupii.PER_LABEL_THRESHOLDS = saved


def test_evaluate_headline_and_sweep_carry_binary_flag():
    """Model-free integration: monkeypatch predict_rich/load_gold, drive derive() with canned
    spans, and confirm evaluate() surfaces the binary-flag metrics in headline + every sweep row."""
    import types
    recs = [
        # gold Public but a (valid) person fires -> false flag (FP)
        types.SimpleNamespace(id="a", source="t", text="Anna Smith",
                              gold_level="Public", gold_elements=[], spans=[]),
        # gold Public, nothing fires -> TN
        types.SimpleNamespace(id="b", source="t", text="the meeting notes",
                              gold_level="Public", gold_elements=[], spans=[]),
        # gold Private phone, phone fires -> TP
        types.SimpleNamespace(id="c", source="t", text="ring 0412345678 today",
                              gold_level="Private", gold_elements=["phone number"],
                              spans=[types.SimpleNamespace(label="phone number", start=5, end=15)]),
    ]
    gliner = [[("person", 0, 10, 0.9)], [], [("phone number", 5, 15, 0.95)]]

    def fake_predict_rich(engine, texts, *a, **k):
        return {"engine": "hybrid", "device": "cpu"}, gliner, [[] for _ in texts]

    saved_pr, saved_lg = ev.predict_rich, ev.load_gold
    ev.predict_rich = fake_predict_rich
    ev.load_gold = lambda path, limit=None: recs
    try:
        result, _ = ev.evaluate("ignored", engine="hybrid", threshold=0.7,
                                sweep=[0.5, 0.7], n_boot=50, progress=False)
    finally:
        ev.predict_rich, ev.load_gold = saved_pr, saved_lg

    h = result["headline"]
    for k in ("binary_precision", "binary_recall", "binary_f1", "false_flag_rate",
              "binary_precision_ci95", "binary_recall_ci95"):
        assert k in h, f"missing headline key {k}"
    # tp=1 (phone), fp=1 (person on Public row), fn=0, tn=1
    assert h["binary_precision"] == 0.5 and h["binary_recall"] == 1.0
    assert abs(h["false_flag_rate"] - 0.5) < 1e-9
    assert all("binary_precision" in s and "binary_recall" in s for s in result["sweep"])


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
