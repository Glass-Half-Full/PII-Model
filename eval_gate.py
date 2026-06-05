"""Release eval gate — runs the classifier over a labeled holdout and enforces quality bars.
Exits non-zero on regression so CI can block merges. Fully offline.

The headline metric is BALANCED GIRP accuracy. The gate uses NO-REGRESSION-VS-BASELINE rather
than absolute bars: real-world balanced accuracy (~83% on data/gold) is far below the synthetic
100% the old absolute 95% bar assumed. It reads the current tagged baseline from models.lock and
fails a change only if it regresses balanced accuracy or makes the (dangerous) under-classification
worse. Health-tier under-classification keeps a tight absolute bar (compliance-critical).

Holdout: defaults to the in-repo synthetic generator. Prefer a real gold set:
    {"text": "...", "expected": "Public|..."}  OR a gold_data record ({"text","gold_level",...}).

Usage:
    python eval_gate.py --gold data/gold/v1/dev.jsonl --threshold 0.7
    python eval_gate.py                       # synthetic holdout (legacy smoke check)
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from girp import RANK, LEVELS

# No-regression tolerances vs the tagged baseline (models.lock). Health-under keeps a tight absolute
# bar because it is the compliance-critical, dangerous direction.
TOL = {
    "balanced_drop_max": 0.01,    # balanced accuracy may not drop more than 1pp vs baseline
    "under_rise_max": 0.01,       # under-classification may not rise more than 1pp vs baseline
    "health_under_abs_max": 0.03, # absolute ceiling on health-tier under-classification
}


def load_holdout(path, n=300, seed=12345):
    if path:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        texts = [r["text"] for r in rows]
        expected = [r.get("expected", r.get("gold_level")) for r in rows]
        return texts, expected
    from synthetic import generate_synthetic_dataset
    df = generate_synthetic_dataset(n, seed=seed)
    return df["text"].tolist(), df["expected"].tolist()


def classify(texts, threshold):
    import pandas as pd
    df = pd.DataFrame({"text": texts})
    try:
        from aupii import load_hybrid, classify_columns_hybrid
        model, analyzer, dev = load_hybrid()
        out = classify_columns_hybrid(model, analyzer, df, ["text"], threshold=threshold, progress=False)
        return out["girp_level"].tolist(), f"hybrid/{dev}"
    except Exception as e:
        print(f"[eval_gate] hybrid unavailable ({type(e).__name__}); using base classifier.")
        from girp import load_local_model, classify_columns
        model, dev = load_local_model()
        out = classify_columns(model, df, ["text"], threshold=threshold, progress=False)
        return out["girp_level"].tolist(), f"base/{dev}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", "--holdout", dest="gold", default=None,
                    help="labeled JSONL ({text,expected} or gold_data record); default = synthetic")
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    texts, gold = load_holdout(args.gold)
    pred, engine = classify(texts, args.threshold)

    from evaluate import balanced_accuracy, confusion, rates
    from metrics_io import current_metrics
    cm = confusion(gold, pred)
    bal, per_tier = balanced_accuracy(cm)
    rt = rates(gold, pred)
    n = len(gold)

    print(f"=== EVAL GATE ({engine}, {n} rows, threshold={args.threshold}) ===")
    print(f"balanced GIRP accuracy={bal*100:.1f}%  accuracy={rt['accuracy']*100:.1f}%  "
          f"under={rt['under']*100:.1f}%  over={rt['over']*100:.1f}%  "
          f"health-under={rt['health_under']*100:.1f}%")

    base_ver, base = current_metrics()
    fails = []
    # Absolute compliance ceiling on the dangerous health direction.
    if rt["health_under"] > TOL["health_under_abs_max"]:
        fails.append(f"health under-classification {rt['health_under']*100:.1f}% > "
                     f"{TOL['health_under_abs_max']*100:.0f}% (absolute)")
    if base:
        b_bal = base.get("balanced_accuracy", 0.0)
        b_under = base.get("under", 1.0)
        print(f"baseline v{base_ver}: balanced={b_bal*100:.1f}%  under={b_under*100:.1f}%")
        if bal < b_bal - TOL["balanced_drop_max"]:
            fails.append(f"balanced accuracy regressed {bal*100:.1f}% < "
                         f"{b_bal*100:.1f}%-{TOL['balanced_drop_max']*100:.0f}pp")
        if rt["under"] > b_under + TOL["under_rise_max"]:
            fails.append(f"under-classification rose {rt['under']*100:.1f}% > "
                         f"{b_under*100:.1f}%+{TOL['under_rise_max']*100:.0f}pp")
    else:
        print("no baseline tagged yet (models.lock) — establishing baseline; gate is informational")

    if fails:
        print("GATE FAILED: " + "; ".join(fails))
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
