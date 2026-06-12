"""Binary 'PII present / absent' precision evaluation on real free text.

ONE inference pass (reuses ``evaluate.predict_rich`` at FLOOR) → re-thresholds the cached spans
with ``evaluate.derive`` across a fine sweep → the full precision-recall-vs-threshold curve for the
binary flag "does this field contain any PII":

  * mode A (default): present = (len(elements) > 0)     — any PII element at all
  * mode B          : present = (level != "Public")     — the tier the model actually ships

(the two differ only on lone-DOB rows). Headline = precision at a RECALL FLOOR (the high-precision
operating point), precision at the shipped threshold, and best-F1 — with bootstrap CIs and a
per-source breakdown. Balanced GIRP accuracy is kept as a SECONDARY guard so optimising the binary
flag can't silently wreck tiering. Writes ``data/eval/<out>/{metrics.json, REPORT.md, mismatches.jsonl}``.

This is dev-time tooling; the model loads from LOCAL weights only (HF_HUB_OFFLINE).
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

import evaluate as ev
from girp import LEVELS, RANK


# ---------------------------------------------------------------------------
# Pure binary-flag helpers (no model)
# ---------------------------------------------------------------------------
def present_flag(elements, level, mode="A"):
    """Binary PII-present flag. mode 'A' = any element present; 'B' = level != 'Public'."""
    return (len(elements) > 0) if mode == "A" else (level != "Public")


def flag_prf(gold_flags, pred_flags):
    """Precision/recall/F1 + false-flag rate for the binary present/absent decision."""
    tp = fp = fn = tn = 0
    for g, p in zip(gold_flags, pred_flags):
        if g and p:
            tp += 1
        elif p:
            fp += 1
        elif g:
            fn += 1
        else:
            tn += 1
    p, r, f = ev._prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f,
            "false_flag_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pick_precision_at_recall(curve, recall_floor):
    """Highest-precision operating point whose recall >= floor (the high-precision target).

    Returns ``(point, achieved)``. If the floor is unreachable, returns the highest-recall point
    and ``achieved=False`` so the caller can flag it.
    """
    ok = [c for c in curve if c["recall"] >= recall_floor]
    if not ok:
        return max(curve, key=lambda c: (c["recall"], c["precision"])), False
    return max(ok, key=lambda c: (c["precision"], c["recall"])), True


def best_f1_point(curve):
    return max(curve, key=lambda c: (c["f1"], c["precision"]))


# Default high-FP fuzzy labels to calibrate (Lever A). Checksum IDs (bank/TFN/card) come from Presidio,
# not the fuzzy model, so they are tightened Presidio-side (Lever B), not here.
WEAK_LABELS = ["date of birth", "driver's licence number", "passport number",
               "address", "person", "phone number"]


def per_label_pr(gliner_per_text, texts, gold_sets, label, thr, validate=True):
    """Presence-level precision/recall for ONE label at confidence threshold ``thr`` (Lever-A calibration).
    A text 'predicts' the label if any extracted span of that label at conf >= thr passes validation."""
    from girp import is_valid_entity
    tp = fp = fn = 0
    for spans, text, gold in zip(gliner_per_text, texts, gold_sets):
        pred = any(l == label and c >= thr and (not validate or is_valid_entity(label, text[s:e]))
                   for (l, s, e, c) in spans)
        g = label in gold
        tp += g and pred
        fp += pred and not g
        fn += g and not pred
    return ev._prf(tp, fp, fn)


def per_entity_sweep(gold_path, labels=None, thresholds=None, engine="hybrid", model_dir=None,
                     limit=None, progress=True, precision_target=0.90, recall_floor=0.90):
    """Sweep each weak label's threshold over ONE cached inference pass; recommend the smallest
    threshold per label hitting ``precision_target`` without dropping that label's recall below floor."""
    labels = labels or WEAK_LABELS
    thresholds = thresholds or [round(0.50 + 0.05 * i, 2) for i in range(9)]   # 0.50 .. 0.90
    recs = ev.load_gold(gold_path, limit=limit)
    texts = [r.text for r in recs]
    gold_sets = [set(r.gold_elements) for r in recs]
    meta, gliner, _presidio = ev.predict_rich(engine, texts, progress=progress, model_dir=model_dir)
    out = {}
    for lab in labels:
        rows = [{"threshold": t, "precision": (pr := per_label_pr(gliner, texts, gold_sets, lab, t))[0],
                 "recall": pr[1], "f1": pr[2]} for t in thresholds]
        ok = [x for x in rows if x["precision"] >= precision_target and x["recall"] >= recall_floor]
        out[lab] = {"sweep": rows, "recommended": min(ok, key=lambda x: x["threshold"]) if ok else None}
    return out, meta


def print_per_entity_sweep(result, precision_target, recall_floor):
    suggested = {}
    for lab, d in result.items():
        print(f"\n{lab}:")
        print("  thr    P%     R%")
        for x in d["sweep"]:
            rec = d["recommended"]
            mark = "  <-- recommended" if rec and x["threshold"] == rec["threshold"] else ""
            print(f"  {x['threshold']:.2f}  {x['precision']*100:5.1f}  {x['recall']*100:5.1f}{mark}")
        if d["recommended"] is None:
            print(f"  (no threshold hits precision>={precision_target:.0%} with recall>={recall_floor:.0%})")
        else:
            suggested[lab] = d["recommended"]["threshold"]
    print(f"\nSuggested aupii.PER_LABEL_THRESHOLDS (precision>={precision_target:.0%}, "
          f"recall>={recall_floor:.0%}):\n  {suggested!r}")
    print("  -> paste into aupii.py, then VERIFY on this box: python test_aupii.py && python selfcheck.py")


def _bootstrap_binary(gold_flags, pred_flags, n_boot=2000, seed=12345):
    rng = np.random.default_rng(seed)
    g = np.array(gold_flags, dtype=bool)
    p = np.array(pred_flags, dtype=bool)
    n = len(g)
    precs, recs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gi, pi = g[idx], p[idx]
        tp = int((gi & pi).sum()); fp = int((~gi & pi).sum()); fn = int((gi & ~pi).sum())
        precs.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recs.append(tp / (tp + fn) if (tp + fn) else 0.0)

    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return ci(precs), ci(recs)


# ---------------------------------------------------------------------------
# Orchestration — one inference pass, sweep over cached spans
# ---------------------------------------------------------------------------
def run(gold_path, engine="hybrid", sweep=None, operating_threshold=0.7, recall_floor=0.97,
        mode="A", n_boot=2000, seed=12345, limit=None, model_dir=None, progress=True):
    recs = ev.load_gold(gold_path, limit=limit)
    texts = [r.text for r in recs]
    gold_levels = [r.gold_level for r in recs]
    sources = [r.source for r in recs]
    gold_flags = [present_flag(set(r.gold_elements), r.gold_level, mode) for r in recs]

    meta, gliner, presidio = ev.predict_rich(engine, texts, progress=progress, model_dir=model_dir)
    eng = meta["engine"]

    op = round(operating_threshold, 3)
    thresholds = sorted(set(sweep or [op]) | {op})

    # one derive pass per threshold over the cached spans (no re-inference)
    curve, pred_at = [], {}
    for t in thresholds:
        flags, levels, elems = [], [], []
        for i, text in enumerate(texts):
            el, lvl, _ = ev.derive(gliner[i], presidio[i], text, t, eng)
            flags.append(present_flag(el, lvl, mode)); levels.append(lvl); elems.append(el)
        pred_at[round(t, 3)] = {"flags": flags, "levels": levels, "elements": elems}
        curve.append({"threshold": round(t, 3), **flag_prf(gold_flags, flags)})

    op_flags = pred_at[op]["flags"]
    op_levels = pred_at[op]["levels"]
    op_point = next(c for c in curve if c["threshold"] == op)
    floor_point, floor_ok = pick_precision_at_recall(curve, recall_floor)
    f1_point = best_f1_point(curve)
    pci, rci = _bootstrap_binary(gold_flags, op_flags, n_boot, seed)

    by_source = {}
    for src in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == src]
        by_source[src] = {"n": len(idx),
                          **flag_prf([gold_flags[i] for i in idx], [op_flags[i] for i in idx])}

    cm = ev.confusion(gold_levels, op_levels)
    bal, per_tier = ev.balanced_accuracy(cm)

    mismatches = _binary_mismatches(recs, gliner, presidio, eng, mode, gold_flags,
                                    op_flags, pred_at[op]["elements"], op_levels)

    n_present = int(sum(gold_flags))
    result = {
        "config": {
            "engine": eng, "device": meta.get("device"), "mode": mode,
            "operating_threshold": op, "recall_floor": recall_floor,
            "floor": ev.FLOOR, "health_threshold": ev.HEALTH_THR, "presidio_threshold": ev.PRESIDIO_THR,
            "gold_path": gold_path, "n": len(recs), "n_present": n_present,
            "n_absent": len(recs) - n_present, "seed": seed, "n_boot": n_boot,
            "girp_rules_sha": ev.girp_rules_sha(), "weights_sha256": ev.weights_sha256(model_dir),
        },
        "headline": {
            "mode": mode,
            "precision_at_recall_floor": {"recall_floor": recall_floor, "achieved": floor_ok,
                                          **{k: floor_point[k] for k in ("threshold", "precision", "recall", "f1")}},
            "precision_at_operating": {**{k: op_point[k] for k in ("threshold", "precision", "recall",
                                                                   "f1", "false_flag_rate")},
                                       "binary_precision_ci95": pci, "binary_recall_ci95": rci},
            "best_f1": {k: f1_point[k] for k in ("threshold", "precision", "recall", "f1")},
        },
        "curve": curve,
        "by_source": by_source,
        "secondary_balanced_accuracy": round(bal, 4),
        "secondary_per_tier_recall": per_tier,
        "secondary_confusion": {"levels": LEVELS, "rows_are_gold": True, "matrix": cm.tolist()},
    }
    return result, mismatches


def _binary_mismatches(recs, gliner, presidio, engine, mode, gold_flags, pred_flags,
                       pred_elements, pred_levels):
    """Binary FP / FN rows at the operating threshold, with the FLOOR probe (rescuable vs gap)."""
    out = []
    for i, r in enumerate(recs):
        if gold_flags[i] == pred_flags[i]:
            continue
        floor_elems, _floor_lvl, _ = ev.derive(gliner[i], presidio[i], r.text, ev.FLOOR, engine)
        gold_e = set(r.gold_elements)
        pred_e = set(pred_elements[i])
        floor_e = set(floor_elems)
        missed = gold_e - pred_e
        out.append({
            "id": r.id, "source": r.source, "text": r.text,
            "direction": "false_positive" if (pred_flags[i] and not gold_flags[i]) else "false_negative",
            "gold": {"level": r.gold_level, "elements": sorted(gold_e)},
            "pred": {"level": pred_levels[i], "elements": sorted(pred_e)},
            "probe": {
                "spurious_elements": sorted(pred_e - gold_e),     # the precision enemy (FP)
                "missed_elements": sorted(missed),
                "rescuable_at_floor": sorted(missed & floor_e),   # threshold/validation -> Stage 1
                "detection_gap": sorted(missed - floor_e),        # never surfaced -> Stage 2
            },
        })
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_report(result, out_dir, mismatches=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    if mismatches is not None:
        with open(os.path.join(out_dir, "mismatches.jsonl"), "w") as f:
            for m in mismatches:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    c = result["config"]
    h = result["headline"]
    fl = h["precision_at_recall_floor"]
    op = h["precision_at_operating"]
    bf = h["best_f1"]
    L = [
        f"# Binary PII-present flag evaluation — {c['engine']} (mode {c['mode']})",
        "",
        f"Gold: `{c['gold_path']}`  ·  n={c['n']} ({c['n_present']} present / {c['n_absent']} absent)  "
        f"·  device={c['device']}  ·  rules={c['girp_rules_sha']}",
        "",
        "## Headline — high-precision operating point",
        "",
        f"- **Precision @ recall ≥ {fl['recall_floor']*100:.0f}%: {fl['precision']*100:.1f}%** "
        f"(recall {fl['recall']*100:.1f}%, threshold {fl['threshold']})"
        + ("" if fl["achieved"] else "  ⚠️ recall floor UNREACHABLE — showing highest-recall point"),
        f"- Precision @ shipped threshold {op['threshold']}: {op['precision']*100:.1f}% "
        f"(95% CI {op['binary_precision_ci95'][0]*100:.1f}–{op['binary_precision_ci95'][1]*100:.1f})  ·  "
        f"recall {op['recall']*100:.1f}% "
        f"(95% CI {op['binary_recall_ci95'][0]*100:.1f}–{op['binary_recall_ci95'][1]*100:.1f})  ·  "
        f"false-flag rate {op['false_flag_rate']*100:.1f}%",
        f"- Best F1: {bf['f1']*100:.1f}% at threshold {bf['threshold']} "
        f"(P {bf['precision']*100:.1f}% / R {bf['recall']*100:.1f}%)",
        "",
        "## Precision–recall vs threshold",
        "",
        "| threshold | precision% | recall% | F1% | false-flag% | tp/fp/fn/tn |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for s in result["curve"]:
        L.append(f"| {s['threshold']} | {s['precision']*100:.1f} | {s['recall']*100:.1f} | "
                 f"{s['f1']*100:.1f} | {s['false_flag_rate']*100:.1f} | "
                 f"{s['tp']}/{s['fp']}/{s['fn']}/{s['tn']} |")
    L += ["", "## Binary flag by source", "",
          "| source | n | precision% | recall% | F1% | tp/fp/fn/tn |", "|---|---:|---:|---:|---:|---|"]
    for src, m in sorted(result["by_source"].items()):
        L.append(f"| {src} | {m['n']} | {m['precision']*100:.1f} | {m['recall']*100:.1f} | "
                 f"{m['f1']*100:.1f} | {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} |")
    L += ["", "## Secondary guard — balanced GIRP accuracy (must not collapse)", "",
          f"- Balanced GIRP accuracy @ operating threshold: {result['secondary_balanced_accuracy']*100:.1f}%",
          f"- Per-tier recall: " + ", ".join(f"{k} {v if v is None else round(v*100,1)}%"
                                             for k, v in result["secondary_per_tier_recall"].items()),
          ""]
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return os.path.join(out_dir, "REPORT.md")


def main():
    ap = argparse.ArgumentParser(description="Binary PII-present precision evaluation on real text.")
    ap.add_argument("--gold", default="data/gold/real-v1/test.jsonl")
    ap.add_argument("--engine", choices=["hybrid", "base"], default="hybrid")
    ap.add_argument("--sweep", default="0.30:0.90:0.02", help="lo:hi:step")
    ap.add_argument("--operating-threshold", type=float, default=0.7)
    ap.add_argument("--recall-floor", type=float, default=0.97)
    ap.add_argument("--binary-mode", choices=["A", "B"], default="A")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/eval/real-baseline")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--per-entity-sweep", action="store_true",
                    help="Lever-A calibration: per-label precision/recall sweep (one inference pass)")
    ap.add_argument("--labels", default=None, help="comma-separated labels for --per-entity-sweep")
    ap.add_argument("--precision-target", type=float, default=0.90)
    args = ap.parse_args()

    if args.per_entity_sweep:
        labels = [x.strip() for x in args.labels.split(",")] if args.labels else None
        result, _meta = per_entity_sweep(args.gold, labels=labels, engine=args.engine,
                                         model_dir=args.model_dir, limit=args.limit,
                                         precision_target=args.precision_target,
                                         recall_floor=args.recall_floor)
        print_per_entity_sweep(result, args.precision_target, args.recall_floor)
        return

    result, mismatches = run(
        args.gold, engine=args.engine, sweep=ev._parse_sweep(args.sweep),
        operating_threshold=args.operating_threshold, recall_floor=args.recall_floor,
        mode=args.binary_mode, n_boot=args.bootstrap, seed=args.seed, limit=args.limit,
        model_dir=args.model_dir)
    path = write_report(result, args.out, mismatches=mismatches)
    h = result["headline"]
    fl, op = h["precision_at_recall_floor"], h["precision_at_operating"]
    print(f"\nbinary PII-present flag (mode {args.binary_mode}) on {result['config']['n']} rows "
          f"({result['config']['n_present']} present / {result['config']['n_absent']} absent)")
    print(f"  precision @ recall>={fl['recall_floor']*100:.0f}%: {fl['precision']*100:.1f}% "
          f"(recall {fl['recall']*100:.1f}%, thr {fl['threshold']})"
          + ("" if fl["achieved"] else "  [floor unreachable]"))
    print(f"  precision @ thr {op['threshold']}: {op['precision']*100:.1f}% / recall {op['recall']*100:.1f}%")
    print(f"  report: {path}  ·  mismatches: {len(mismatches)}")


if __name__ == "__main__":
    main()
