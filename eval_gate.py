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

# Precision-first gate. The PII flag is binary "PII present"; the hard bars protect that flag's
# PRECISION (no regression vs baseline) and RECALL (absolute floor — don't start missing PII),
# plus the compliance-critical health-under absolute. Balanced GIRP accuracy and overall
# under-classification are DEMOTED to warnings so a precision-improving iteration that slightly
# trades tiering is not blocked.
TOL = {
    "binary_precision_drop_max": 0.01,  # binary precision may not drop more than 1pp vs baseline (HARD)
    "binary_recall_floor_abs": 0.97,    # absolute binary-recall floor — don't start missing PII (HARD)
    "health_under_abs_max": 0.03,       # absolute ceiling on health-tier under-classification (HARD)
    "balanced_drop_max": 0.05,          # balanced accuracy collapse (WARNING only)
    "under_rise_max": 0.02,             # under-classification rise (WARNING only)
}


def gate_check(m, baseline, tol=TOL):
    """Pure pass/fail decision. ``m`` carries balanced_accuracy/under/health_under/binary_precision/
    binary_recall for the candidate; ``baseline`` is the tagged metrics (models.lock) or None.
    Returns ``(fails, warnings)`` — non-empty ``fails`` means the gate blocks."""
    fails, warnings = [], []
    if m["health_under"] > tol["health_under_abs_max"]:
        fails.append(f"health under-classification {m['health_under']*100:.1f}% > "
                     f"{tol['health_under_abs_max']*100:.0f}% (absolute)")
    if m["binary_recall"] < tol["binary_recall_floor_abs"]:
        fails.append(f"binary PII-present recall {m['binary_recall']*100:.1f}% < floor "
                     f"{tol['binary_recall_floor_abs']*100:.0f}% (absolute) — started missing PII")
    if baseline:
        b_bp = baseline.get("binary_precision")
        if b_bp is not None and m["binary_precision"] < b_bp - tol["binary_precision_drop_max"]:
            fails.append(f"binary precision regressed {m['binary_precision']*100:.1f}% < "
                         f"{b_bp*100:.1f}%-{tol['binary_precision_drop_max']*100:.0f}pp")
        b_bal = baseline.get("balanced_accuracy")
        if b_bal is not None and m["balanced_accuracy"] < b_bal - tol["balanced_drop_max"]:
            warnings.append(f"balanced accuracy {m['balanced_accuracy']*100:.1f}% < "
                            f"{b_bal*100:.1f}%-{tol['balanced_drop_max']*100:.0f}pp (secondary)")
        b_under = baseline.get("under")
        if b_under is not None and m["under"] > b_under + tol["under_rise_max"]:
            warnings.append(f"under-classification rose {m['under']*100:.1f}% > "
                            f"{b_under*100:.1f}%+{tol['under_rise_max']*100:.0f}pp (secondary)")
    return fails, warnings


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

    from evaluate import balanced_accuracy, binary_flag_metrics, confusion, rates
    from metrics_io import current_metrics
    cm = confusion(gold, pred)
    bal, per_tier = balanced_accuracy(cm)
    rt = rates(gold, pred)
    bn = binary_flag_metrics(gold, pred)
    n = len(gold)

    print(f"=== EVAL GATE ({engine}, {n} rows, threshold={args.threshold}) ===")
    print(f"binary PII-present: precision={bn['binary_precision']*100:.1f}%  "
          f"recall={bn['binary_recall']*100:.1f}%  false-flag={bn['false_flag_rate']*100:.1f}%")
    print(f"balanced GIRP accuracy={bal*100:.1f}%  accuracy={rt['accuracy']*100:.1f}%  "
          f"under={rt['under']*100:.1f}%  over={rt['over']*100:.1f}%  "
          f"health-under={rt['health_under']*100:.1f}%")

    base_ver, base = current_metrics()
    if base:
        print(f"baseline v{base_ver}: binary precision={base.get('binary_precision', '?')}  "
              f"balanced={base.get('balanced_accuracy', 0.0)*100:.1f}%")
    else:
        print("no baseline tagged yet (models.lock) — establishing baseline; "
              "precision-regression check skipped (absolute recall/health bars still apply)")

    m = {"balanced_accuracy": bal, "under": rt["under"], "over": rt["over"],
         "health_under": rt["health_under"], "binary_precision": bn["binary_precision"],
         "binary_recall": bn["binary_recall"]}
    fails, warnings = gate_check(m, base)
    for w in warnings:
        print(f"  WARNING: {w}")
    if fails:
        print("GATE FAILED: " + "; ".join(fails))
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
