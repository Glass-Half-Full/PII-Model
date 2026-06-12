"""Tests for eval_gate.py gate_check — precision-first release gate (offline, no model)."""
import eval_gate as eg


def _m(bp=0.92, br=0.98, bal=0.84, health=0.0, under=0.06, over=0.08):
    return {"binary_precision": bp, "binary_recall": br, "balanced_accuracy": bal,
            "health_under": health, "under": under, "over": over}


def test_gate_check_passes_precision_up():
    base = {"balanced_accuracy": 0.85, "binary_precision": 0.88, "under": 0.05}
    fails, warns = eg.gate_check(_m(bp=0.92, br=0.98), base)
    assert fails == []


def test_gate_check_fails_recall_below_floor():
    base = {"balanced_accuracy": 0.85, "binary_precision": 0.88, "under": 0.05}
    fails, _ = eg.gate_check(_m(bp=0.95, br=0.90), base)        # great precision, recall cliff
    assert any("recall" in f.lower() for f in fails)


def test_gate_check_fails_precision_regression():
    base = {"balanced_accuracy": 0.85, "binary_precision": 0.90, "under": 0.05}
    fails, _ = eg.gate_check(_m(bp=0.85, br=0.98), base)        # precision dropped >1pp vs baseline
    assert any("precision" in f.lower() for f in fails)


def test_gate_check_balanced_drop_is_warning_not_fail():
    base = {"balanced_accuracy": 0.85, "binary_precision": 0.88, "under": 0.05}
    fails, warns = eg.gate_check(_m(bp=0.92, br=0.98, bal=0.60), base)   # tiering down, precision up
    assert fails == [] and any("balanced" in w.lower() for w in warns)


def test_gate_check_health_under_hard_fail():
    base = {"balanced_accuracy": 0.85, "binary_precision": 0.88, "under": 0.05}
    fails, _ = eg.gate_check(_m(bp=0.95, br=0.99, health=0.05), base)
    assert any("health" in f.lower() for f in fails)


def test_gate_check_no_baseline_enforces_absolutes_only():
    fails, warns = eg.gate_check(_m(bp=0.10, br=0.99, health=0.0), None)
    assert fails == []          # no baseline -> can't check precision regression; recall+health OK
    fails2, _ = eg.gate_check(_m(br=0.90), None)
    assert any("recall" in f.lower() for f in fails2)   # absolute recall floor still enforced


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
    print(f"\neval_gate: {passed}/{len(fns)} passed.")
    if passed != len(fns):
        raise SystemExit(1)
    print("All eval_gate tests passed.")


if __name__ == "__main__":
    _run()
