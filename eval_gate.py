"""Release eval gate — runs the classifier over a fixed labeled holdout and enforces minimum
quality thresholds. Exits non-zero on regression so CI can block merges. Fully offline.

Holdout: defaults to the in-repo synthetic generator (reproducible, no download). For a real
production gate, point `--holdout` at a labeled JSONL of YOUR data, one object per line:
    {"text": "...", "expected": "Public|Private|Confidential|Highly Confidential"}

Usage:
    python eval_gate.py                       # synthetic holdout, hybrid if available else base
    python eval_gate.py --holdout my.jsonl --threshold 0.7
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from girp import RANK, LEVELS

# Release bars for the *automated* layer. Under-classification is the dangerous direction, so the
# bars are tight. The human-review band (PRODUCTION.md Part F) routes all Highly/Confidential
# predictions + engine disagreements to a reviewer, so the effective health-miss rate after review
# approaches 0; fine-tuning (train_lora.py) drives the automated rate down further. Target: 0.
THRESHOLDS = {
    "accuracy_min": 0.95,
    "under_max": 0.04,
    "health_under_max": 0.02,
}


def load_holdout(path, n=300, seed=12345):
    if path:
        rows = [json.loads(l) for l in open(path)]
        return [r["text"] for r in rows], [r["expected"] for r in rows]
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
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    texts, gold = load_holdout(args.holdout)
    pred, engine = classify(texts, args.threshold)
    n = len(gold)

    acc = sum(p == g for p, g in zip(pred, gold)) / n
    under = sum(RANK[p] < RANK[g] for p, g in zip(pred, gold)) / n
    over = sum(RANK[p] > RANK[g] for p, g in zip(pred, gold)) / n
    h_total = sum(1 for g in gold if g == "Highly Confidential")
    h_under = sum(1 for p, g in zip(pred, gold) if g == "Highly Confidential" and RANK[p] < RANK[g])
    h_under_rate = (h_under / h_total) if h_total else 0.0

    print(f"=== EVAL GATE ({engine}, {n} rows, threshold={args.threshold}) ===")
    print(f"accuracy={acc*100:.1f}%  under={under*100:.1f}%  over={over*100:.1f}%  "
          f"health-under={h_under}/{h_total}")

    fails = []
    if acc < THRESHOLDS["accuracy_min"]:
        fails.append(f"accuracy {acc*100:.1f}% < {THRESHOLDS['accuracy_min']*100:.0f}%")
    if under > THRESHOLDS["under_max"]:
        fails.append(f"under-classification {under*100:.1f}% > {THRESHOLDS['under_max']*100:.0f}%")
    if h_under_rate > THRESHOLDS["health_under_max"]:
        fails.append(f"health under-classification {h_under_rate*100:.1f}% > {THRESHOLDS['health_under_max']*100:.0f}%")

    if fails:
        print("GATE FAILED: " + "; ".join(fails))
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
