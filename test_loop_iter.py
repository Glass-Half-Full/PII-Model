"""Tests for loop_iter.py: binary-precision gate + FP-first error ordering (offline, no model)."""
import json
import os
import tempfile

import loop_iter as li


def _headline(bp, br, bal=0.85, health_under=0.0):
    return {"binary_precision": bp, "binary_recall": br, "balanced_accuracy": bal,
            "health_under": health_under, "under": 0.05, "over": 0.08}


def _write_eval(d, headline=None, confusion=None, mismatches=None):
    os.makedirs(d, exist_ok=True)
    metrics = {"config": {"engine": "hybrid", "threshold": 0.7, "n": 100}, "headline": headline or {}}
    if confusion is not None:
        metrics["confusion_matrix"] = {"matrix": confusion}
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    if mismatches is not None:
        with open(os.path.join(d, "mismatches.jsonl"), "w") as f:
            for m in mismatches:
                f.write(json.dumps(m) + "\n")
    return d


def test_decide_accepts_precision_up_recall_held():
    d = tempfile.mkdtemp()
    before = _write_eval(os.path.join(d, "before"), _headline(0.80, 0.98))
    after = _write_eval(os.path.join(d, "after"), _headline(0.88, 0.975))
    v = li.decide(before, after, recall_floor=0.97, state_path=os.path.join(d, "state.json"))
    assert v["accepted"] is True
    assert v["binary_precision_delta"] == 0.08 and v["recall_ok"] and v["precision_ok"]


def test_decide_rejects_recall_cliff():
    d = tempfile.mkdtemp()
    before = _write_eval(os.path.join(d, "before"), _headline(0.80, 0.98))
    after = _write_eval(os.path.join(d, "after"), _headline(0.95, 0.90))   # precision up but recall cliff
    v = li.decide(before, after, recall_floor=0.97, state_path=os.path.join(d, "state.json"))
    assert v["accepted"] is False and not v["recall_ok"]


def test_decide_rejects_precision_drop():
    d = tempfile.mkdtemp()
    before = _write_eval(os.path.join(d, "before"), _headline(0.88, 0.98))
    after = _write_eval(os.path.join(d, "after"), _headline(0.80, 0.98))   # precision regressed
    v = li.decide(before, after, recall_floor=0.97, state_path=os.path.join(d, "state.json"))
    assert v["accepted"] is False and not v["precision_ok"]


def test_decide_balanced_collapse_blocks_as_secondary_guard():
    d = tempfile.mkdtemp()
    before = _write_eval(os.path.join(d, "before"), _headline(0.80, 0.98, bal=0.85))
    after = _write_eval(os.path.join(d, "after"), _headline(0.90, 0.98, bal=0.60))   # precision up, tiering wrecked
    v = li.decide(before, after, recall_floor=0.97, balanced_tol=0.02,
                  state_path=os.path.join(d, "state.json"))
    assert v["accepted"] is False and not v["balanced_ok"]


def test_binary_pr_from_confusion_fallback():
    # headline lacks binary keys -> derive from the 4x4 confusion matrix (rows=gold, cols=pred)
    cm = [[10, 2, 0, 0],   # gold Public: 10 TN, 2 false flags (FP)
          [1, 20, 0, 0],   # gold Private: 1 missed (FN), 20 flagged (TP)
          [0, 0, 15, 0],   # gold Confidential: 15 TP
          [0, 0, 0, 8]]    # gold Highly: 8 TP
    prec, rec = li._binary_pr({"headline": {}, "confusion_matrix": {"matrix": cm}})
    assert abs(prec - 43 / 45) < 1e-3   # TP=43, FP=2
    assert abs(rec - 43 / 44) < 1e-3    # TP=43, FN=1


def test_extract_errors_orders_binary_false_positives_first():
    mismatches = [
        {"id": "under1", "direction": "under", "source": "s",
         "gold": {"level": "Confidential"}, "pred": {"level": "Private"},
         "probe": {"spurious_elements": [], "missed_elements": ["x"],
                   "rescuable_at_floor": [], "detection_gap": ["x"]}},
        {"id": "fp1", "direction": "over", "source": "s",
         "gold": {"level": "Public"}, "pred": {"level": "Confidential"},   # binary false positive
         "probe": {"spurious_elements": ["person"], "missed_elements": [],
                   "rescuable_at_floor": [], "detection_gap": []}},
    ]
    d = tempfile.mkdtemp()
    ed = _write_eval(os.path.join(d, "e"), _headline(0.8, 0.98), mismatches=mismatches)
    out = os.path.join(d, "errors.jsonl")
    capped = li.extract_errors(ed, out, cap=10)
    assert capped[0]["id"] == "fp1"   # the precision enemy is reviewed first


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
    print(f"\nloop_iter gate: {passed}/{len(fns)} passed.")
    if passed != len(fns):
        raise SystemExit(1)
    print("All loop_iter gate tests passed.")


if __name__ == "__main__":
    _run()
