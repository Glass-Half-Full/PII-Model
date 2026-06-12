"""Tests for spotcheck.py — model-assisted spot-check labeling harness (offline, no model)."""
import types

import spotcheck as sc


def _rec(id, text, gold_level="Public", gold_elements=(), spans=()):
    return types.SimpleNamespace(id=id, text=text, source="kaggle-pii",
                                 gold_level=gold_level, gold_elements=list(gold_elements),
                                 spans=list(spans))


def _pred(present, level="Public", elements=(), spans=(), needs_review=False):
    return {"present": present, "level": level, "elements": list(elements),
            "spans": list(spans), "needs_review": needs_review}


# --- bucketing (precision enemy first) -----------------------------------------------------
def test_bucket_row_false_positive_is_top_priority():
    b, pri = sc.bucket_row(gold_present=False, pred_present=True, needs_review=False)
    assert b == "candidate_false_positive" and pri == 0


def test_bucket_row_false_negative():
    b, pri = sc.bucket_row(gold_present=True, pred_present=False, needs_review=False)
    assert b == "candidate_false_negative" and pri == 1


def test_bucket_row_needs_review_when_agreeing():
    b, pri = sc.bucket_row(gold_present=True, pred_present=True, needs_review=True)
    assert b == "needs_review" and pri == 2


def test_bucket_row_agree_is_dropped():
    b, pri = sc.bucket_row(gold_present=True, pred_present=True, needs_review=False)
    assert b == "agree"


def test_build_queue_orders_fp_first_and_caps():
    recs = [
        _rec("fn", "missed pii here", gold_level="Private", gold_elements=["person"],
             spans=[types.SimpleNamespace(label="person", start=0, end=6)]),
        _rec("fp", "order 0412345678", gold_level="Public"),     # model false-fires
        _rec("ok", "nothing", gold_level="Public"),
    ]
    preds = [_pred(False), _pred(True, "Private", ["phone number"]), _pred(False)]
    q = sc.build_queue(recs, preds, cap=10)
    assert [r["id"] for r in q] == ["fp", "fn"]          # FP first, agree("ok") dropped
    assert q[0]["bucket"] == "candidate_false_positive"


# --- verdict routing -----------------------------------------------------------------------
def _verdict(bucket, verdict, gold_spans=(), model_spans=(), gold_level="Public", model_level="Private"):
    return {"id": "x", "text": "some text", "source": "kaggle-pii", "bucket": bucket, "verdict": verdict,
            "gold": {"level": gold_level, "spans": list(gold_spans)},
            "model": {"level": model_level, "spans": list(model_spans)}}


def test_route_gold_right_fp_makes_negative_and_gold_fix():
    out = sc.route_verdict(_verdict("candidate_false_positive", "gold_right", gold_spans=[]))
    assert out["gold_fix"]["gold_level"] == "Public" and out["gold_fix"]["spans"] == []
    assert out["hard_example"]["negative"] is True and out["hard_example"]["reasons"] == ["false_positive"]


def test_route_gold_right_fn_makes_positive_detection_gap():
    gs = [{"label": "person", "start": 0, "end": 4}]
    out = sc.route_verdict(_verdict("candidate_false_negative", "gold_right",
                                    gold_spans=gs, gold_level="Private"))
    assert out["gold_fix"] is None
    assert out["hard_example"]["reasons"] == ["detection_gap"] and out["hard_example"]["spans"] == gs


def test_route_model_right_corrects_gold_only():
    ms = [{"label": "phone number", "start": 6, "end": 16}]
    out = sc.route_verdict(_verdict("candidate_false_negative", "model_right",
                                    model_spans=ms, model_level="Private"))
    assert out["hard_example"] is None
    assert out["gold_fix"]["spans"] == ms and out["gold_fix"]["gold_level"] == "Private"


def test_route_ambiguous_drops():
    out = sc.route_verdict(_verdict("needs_review", "ambiguous"))
    assert out["gold_fix"] is None and out["hard_example"] is None


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
    print(f"\nspotcheck: {passed}/{len(fns)} passed.")
    if passed != len(fns):
        raise SystemExit(1)
    print("All spotcheck tests passed.")


if __name__ == "__main__":
    _run()
