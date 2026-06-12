"""Deterministic helpers for one iteration of the recursive-improvement loop (offline, no model).

Operates purely on evaluate.py's JSON outputs (metrics.json + mismatches.jsonl). The judgment
steps — reviewing/tagging errors, diagnosing root cause, editing rules — are Claude's (see LOOP.md);
these subcommands handle the mechanical bookkeeping so it is deterministic and auditable.

Subcommands:
    summary --eval DIR                         print the headline metrics of an eval run
    errors  --eval DIR --out PATH [--cap N]     extract + prioritise the error set for review
    decide  --before DIR --after DIR            accept/reject verdict (balanced acc up, no regression)
"""
from __future__ import annotations

import argparse
import collections
import json
import os

HEALTH_UNDER_ABS_MAX = 0.03
BINARY_RECALL_FLOOR = 0.97   # default binary-flag recall floor (don't miss real PII); revise per real baseline


def _binary_pr(metrics):
    """Binary PII-present (precision, recall), preferring the headline keys (emitted by evaluate.py),
    else derived from the 4x4 confusion matrix so pre-binary eval dirs still gate. Flag = rank > 0."""
    h = metrics.get("headline", {})
    if h.get("binary_precision") is not None and h.get("binary_recall") is not None:
        return h["binary_precision"], h["binary_recall"]
    cm = (metrics.get("confusion_matrix") or {}).get("matrix")
    if not cm:
        raise KeyError("metrics.json has neither headline binary metrics nor a confusion matrix")
    tp = sum(cm[i][j] for i in range(1, 4) for j in range(1, 4))
    fp = sum(cm[0][j] for j in range(1, 4))
    fn = sum(cm[i][0] for i in range(1, 4))
    return (round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
            round(tp / (tp + fn), 4) if (tp + fn) else 0.0)


def _load(eval_dir, name):
    with open(os.path.join(eval_dir, name)) as f:
        if name.endswith(".jsonl"):
            return [json.loads(l) for l in f if l.strip()]
        return json.load(f)


def summary(eval_dir):
    m = _load(eval_dir, "metrics.json")
    h = m["headline"]
    print(f"=== {eval_dir} ({m['config']['engine']} @ {m['config']['threshold']}, n={m['config']['n']}) ===")
    print(f"balanced GIRP accuracy {h['balanced_accuracy']*100:.1f}% "
          f"(CI {h['balanced_accuracy_ci95'][0]*100:.1f}-{h['balanced_accuracy_ci95'][1]*100:.1f})")
    print(f"under {h['under']*100:.1f}%  over {h['over']*100:.1f}%  health-under {h['health_under']*100:.1f}%")
    print("per-tier recall:", {k: (None if v is None else round(v*100, 1)) for k, v in h["per_tier_recall"].items()})
    return m


def _fp_rank(e):
    """Sort rank for precision work: binary false positives (the precision enemy) first, then other
    over-classifications, then under-classifications. A binary FP = gold Public but model flagged it."""
    if e["gold"]["level"] == "Public" and e["pred"]["level"] != "Public":
        return 0
    return 1 if e["direction"] == "over" else 2


def extract_errors(eval_dir, out_path, cap=80):
    """Prioritise mismatches (binary false positives first — the precision killers) and write the
    review set. Precision-oriented inversion of the old under-first ordering."""
    mm = _load(eval_dir, "mismatches.jsonl")
    mm.sort(key=lambda e: (_fp_rank(e), e["id"]))
    capped = mm[:cap]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for e in capped:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # review-guiding summary
    by_dir = collections.Counter(e["direction"] for e in mm)
    by_src = collections.Counter(e["source"] for e in mm)
    spurious = collections.Counter(x for e in mm for x in e["probe"]["spurious_elements"])
    missed = collections.Counter(x for e in mm for x in e["probe"]["missed_elements"])
    rescuable = sum(1 for e in mm if e["probe"]["rescuable_at_floor"])
    gap = sum(1 for e in mm if e["probe"]["detection_gap"] and not e["probe"]["rescuable_at_floor"])
    print(f"errors: {len(mm)} total ({dict(by_dir)}), wrote top {len(capped)} -> {out_path}")
    print(f"  by source: {dict(by_src)}")
    print(f"  top spurious (false-positive) elements: {spurious.most_common(8)}")
    print(f"  top missed (false-negative) elements:   {missed.most_common(8)}")
    print(f"  rescuable-at-floor (Stage-1 tunable): {rescuable}   detection-gap (Stage-2): {gap}")
    return capped


def accumulate(errors_path, iter_n, hard_path="data/hard_examples.jsonl"):
    """Route reviewed errors to the Stage-2 training pool. Each hard row stores the FULL gold spans
    (the correct training target): for a detection gap this teaches the missed entity; for a false
    positive the gold (which omits the spurious label) teaches suppression. One row per text,
    idempotent by id. A Public row legitimately has spans=[] (a no-entity hard negative)."""
    mm = [json.loads(l) for l in open(errors_path) if l.strip()]
    os.makedirs(os.path.dirname(hard_path), exist_ok=True)
    seen = set()
    if os.path.exists(hard_path):
        for line in open(hard_path):
            if line.strip():
                seen.add(json.loads(line).get("id"))
    gaps = negs = 0
    with open(hard_path, "a") as f:
        for e in mm:
            gap, spur = e["probe"]["detection_gap"], e["probe"]["spurious_elements"]
            if not (gap or spur) or e["id"] in seen:
                continue
            seen.add(e["id"])
            reasons = (["detection_gap"] if gap else []) + (["false_positive"] if spur else [])
            spans = [{"label": l, "start": s, "end": en} for (l, s, en) in e["gold"]["spans"]]
            f.write(json.dumps({"id": e["id"], "text": e["text"], "spans": spans,
                                "source": e["source"], "iter": iter_n, "reasons": reasons,
                                "gold_level": e["gold"]["level"]}, ensure_ascii=False) + "\n")
            gaps += bool(gap)
            negs += bool(spur)
    print(f"accumulated -> {hard_path}: +{gaps} detection-gap, +{negs} false-positive rows "
          f"(pool now {len(seen)} unique texts)")


def decide(before_dir, after_dir, precision_min_delta=0.0, recall_floor=BINARY_RECALL_FLOOR,
           balanced_tol=0.02, state_path="loop_state.json"):
    """Accept an iteration iff BINARY PRECISION holds/improves AND binary recall stays >= floor AND
    health-under is within the absolute ceiling. Balanced GIRP accuracy is a DEMOTED secondary guard
    (blocks only a large collapse, ``balanced_tol``), so a precision-improving iteration that slightly
    trades tiering is not blocked. This is the precision-first inversion of the old balanced-acc gate."""
    bm = _load(before_dir, "metrics.json")
    am = _load(after_dir, "metrics.json")
    b, a = bm["headline"], am["headline"]
    bp, br = _binary_pr(bm)
    ap, ar = _binary_pr(am)
    prec_delta = round(ap - bp, 4)
    bal_delta = round(a["balanced_accuracy"] - b["balanced_accuracy"], 4)
    precision_ok = prec_delta >= precision_min_delta
    recall_ok = ar >= recall_floor
    health_ok = a["health_under"] <= HEALTH_UNDER_ABS_MAX
    balanced_ok = bal_delta >= -balanced_tol
    accepted = bool(precision_ok and recall_ok and health_ok and balanced_ok)
    verdict = {
        "accepted": accepted,
        "binary_precision_before": bp, "binary_precision_after": ap, "binary_precision_delta": prec_delta,
        "binary_recall_before": br, "binary_recall_after": ar, "recall_floor": recall_floor,
        "balanced_before": b["balanced_accuracy"], "balanced_after": a["balanced_accuracy"],
        "balanced_delta": bal_delta, "health_under_after": a["health_under"],
        "precision_ok": precision_ok, "recall_ok": recall_ok,
        "balanced_ok": balanced_ok, "health_ok": health_ok, "reasons": [],
    }
    if not precision_ok:
        verdict["reasons"].append(f"binary precision delta {prec_delta:+.4f} < {precision_min_delta}")
    if not recall_ok:
        verdict["reasons"].append(f"binary recall {ar*100:.1f}% < floor {recall_floor*100:.1f}%")
    if not balanced_ok:
        verdict["reasons"].append(f"balanced accuracy collapsed {bal_delta*100:+.1f}pp "
                                  f"(> {balanced_tol*100:.0f}pp secondary guard)")
    if not health_ok:
        verdict["reasons"].append(f"health-under {a['health_under']*100:.1f}% > {HEALTH_UNDER_ABS_MAX*100:.0f}%")
    with open(state_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(("ACCEPT" if accepted else "REJECT") + f"  binary precision {bp*100:.1f}% -> {ap*100:.1f}% "
          f"({prec_delta*100:+.1f}pp); recall {ar*100:.1f}% (floor {recall_floor*100:.0f}%); "
          f"balanced {b['balanced_accuracy']*100:.1f}% -> {a['balanced_accuracy']*100:.1f}%")
    if verdict["reasons"]:
        print("  " + "; ".join(verdict["reasons"]))
    return verdict


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("--eval", required=True)
    e = sub.add_parser("errors"); e.add_argument("--eval", required=True)
    e.add_argument("--out", required=True); e.add_argument("--cap", type=int, default=80)
    d = sub.add_parser("decide"); d.add_argument("--before", required=True)
    d.add_argument("--after", required=True)
    d.add_argument("--precision-min-delta", type=float, default=0.0)
    d.add_argument("--recall-floor", type=float, default=BINARY_RECALL_FLOOR)
    a = sub.add_parser("accumulate"); a.add_argument("--errors", required=True)
    a.add_argument("--iter", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "summary":
        summary(args.eval)
    elif args.cmd == "errors":
        extract_errors(args.eval, args.out, cap=args.cap)
    elif args.cmd == "decide":
        decide(args.before, args.after, precision_min_delta=args.precision_min_delta,
               recall_floor=args.recall_floor)
    elif args.cmd == "accumulate":
        accumulate(args.errors, args.iter)


if __name__ == "__main__":
    main()
