"""Model-assisted spot-check labeling harness (the ground-truth loop for real free text).

Two subcommands:
  * ``queue``  — run the hybrid model over rows, compare to gold, and emit a LOW-EFFORT human review
                 queue of three buckets, ordered by how much each threatens PRECISION:
                   1. candidate false positives (model flags, gold says none) — the precision enemy
                   2. candidate false negatives (model misses, gold has PII)  — recall-floor watch
                   3. needs_review rows (top tier OR the two engines disagree)
                 The human marks each ``model_right | gold_right | ambiguous`` (one line per row).
  * ``route``  — take the marked verdicts and route them: confirmed false positives become no-entity
                 NEGATIVE hard examples (teach suppression) + a gold fix; confirmed misses become
                 positive detection-gap hard examples; "model_right" corrects the gold label only.

The bucketing + routing are pure (unit-tested offline); only ``queue``'s prediction step needs the
model. Reuses evaluate.predict_rich/derive and loop_iter's idempotent append. See LOOP.md for the
review rubric.
"""
from __future__ import annotations

import argparse
import json
import os


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------
def bucket_row(gold_present, pred_present, needs_review):
    """Bucket + priority (lower = reviewed first). gold_present may be None (unlabeled row)."""
    if gold_present is False and pred_present:
        return "candidate_false_positive", 0      # precision enemy — top priority
    if gold_present and not pred_present:
        return "candidate_false_negative", 1       # recall-floor watch
    if gold_present is None and pred_present:
        return "model_flagged_unlabeled", 1        # no gold yet — human creates the label
    if needs_review:
        return "needs_review", 2
    return "agree", 3


def build_queue(recs, preds, cap=40):
    """Pure: recs (gold) + parallel preds -> sorted, capped review queue (agree rows dropped)."""
    items = []
    for rec, pred in zip(recs, preds):
        gold_elems = list(getattr(rec, "gold_elements", []) or [])
        gold_present = (len(gold_elems) > 0) if getattr(rec, "gold_level", None) is not None else None
        bucket, pri = bucket_row(gold_present, pred["present"], pred.get("needs_review", False))
        if bucket == "agree":
            continue
        items.append({
            "id": rec.id, "text": rec.text, "source": getattr(rec, "source", ""),
            "bucket": bucket, "priority": pri,
            "model": {"level": pred["level"], "elements": list(pred.get("elements", [])),
                      "spans": list(pred.get("spans", []))},
            "gold": {"level": getattr(rec, "gold_level", None), "elements": gold_elems,
                     "spans": [[s.label, s.start, s.end] for s in getattr(rec, "spans", [])]},
            "verdict": "", "note": "",
        })
    items.sort(key=lambda e: (e["priority"], e["id"]))
    return items[:cap]


def _spans_as_dicts(side):
    out = []
    for sp in side.get("spans", []) or []:
        if isinstance(sp, dict):
            out.append({"label": sp["label"], "start": sp["start"], "end": sp["end"]})
        else:  # [label, start, end]
            out.append({"label": sp[0], "start": sp[1], "end": sp[2]})
    return out


def route_verdict(rec):
    """Pure: a marked verdict -> {gold_fix, hard_example} (either may be None)."""
    base = {"id": rec["id"], "text": rec["text"], "source": rec.get("source", "spotcheck")}
    v, bucket = rec["verdict"], rec["bucket"]
    if v == "gold_right":
        if bucket == "candidate_false_positive":
            gspans = _spans_as_dicts(rec["gold"])
            return {
                "gold_fix": {**base, "spans": gspans, "gold_level": rec["gold"]["level"]},
                "hard_example": {**base, "spans": gspans, "reasons": ["false_positive"],
                                 "negative": len(gspans) == 0, "gold_level": rec["gold"]["level"]},
            }
        # model missed real PII (false negative / under-flagged review row): positive training target
        return {"gold_fix": None,
                "hard_example": {**base, "spans": _spans_as_dicts(rec["gold"]),
                                 "reasons": ["detection_gap"], "gold_level": rec["gold"]["level"]}}
    if v == "model_right":
        # gold was wrong -> adopt the model's view as the corrected gold; model already detects it
        return {"gold_fix": {**base, "spans": _spans_as_dicts(rec["model"]),
                             "gold_level": rec["model"]["level"]},
                "hard_example": None}
    return {"gold_fix": None, "hard_example": None}   # ambiguous / unmarked -> drop


# ---------------------------------------------------------------------------
# Model-dependent: build the queue from a gold file (needs the hybrid model)
# ---------------------------------------------------------------------------
def _predict(gold_path, engine, model_dir, limit, progress):
    import evaluate as ev
    from aupii import REVIEW_LEVELS
    from girp import classify_elements
    recs = ev.load_gold(gold_path, limit=limit)
    texts = [r.text for r in recs]
    meta, gliner, presidio = ev.predict_rich(engine, texts, progress=progress, model_dir=model_dir)
    preds = []
    for i, t in enumerate(texts):
        elements, level, spans = ev.derive(gliner[i], presidio[i], t, 0.7, meta["engine"])
        structured = {g for (g, _, _, sc_) in presidio[i] if sc_ >= ev.PRESIDIO_THR}
        fuzzy = set(elements) - structured
        needs_review = (level in REVIEW_LEVELS
                        or classify_elements(fuzzy) != classify_elements(structured))
        preds.append({"present": len(elements) > 0, "level": level, "elements": sorted(elements),
                      "spans": [[l, s, e] for (l, s, e) in spans], "needs_review": needs_review})
    return recs, preds


def run_queue(gold_path, out_path, engine="hybrid", cap=40, model_dir=None, limit=None, progress=True):
    recs, preds = _predict(gold_path, engine, model_dir, limit, progress)
    queue = build_queue(recs, preds, cap=cap)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for q in queue:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    from collections import Counter
    by_bucket = Counter(q["bucket"] for q in queue)
    print(f"spot-check queue: {len(queue)} rows (cap {cap}) -> {out_path}")
    print(f"  by bucket: {dict(by_bucket)}")
    print("  mark each row's \"verdict\": model_right | gold_right | ambiguous, then: "
          "python spotcheck.py route --reviews " + out_path)
    return queue


def route(reviews_path, gold_fixes_path="data/gold_fixes.jsonl", hard_path="data/hard_examples.jsonl"):
    """Route marked verdicts to gold_fixes + the Stage-2 hard-example pool (idempotent by id)."""
    rows = [json.loads(l) for l in open(reviews_path) if l.strip()]
    seen_hard = _existing_ids(hard_path)
    seen_fix = _existing_ids(gold_fixes_path)
    n_fix = n_hard = n_neg = 0
    os.makedirs(os.path.dirname(hard_path) or ".", exist_ok=True)
    with open(gold_fixes_path, "a") as ff, open(hard_path, "a") as hf:
        for rec in rows:
            if not rec.get("verdict"):
                continue
            r = route_verdict(rec)
            gf = r["gold_fix"]
            if gf and gf["id"] not in seen_fix:
                seen_fix.add(gf["id"]); ff.write(json.dumps(gf, ensure_ascii=False) + "\n"); n_fix += 1
            he = r["hard_example"]
            if he and he["id"] not in seen_hard:
                seen_hard.add(he["id"])
                hf.write(json.dumps({**he, "source": he.get("source", "spotcheck"),
                                     "iter": "spotcheck"}, ensure_ascii=False) + "\n")
                n_hard += 1
                n_neg += bool(he.get("negative"))
    print(f"routed {sum(1 for r in rows if r.get('verdict'))} verdicts: +{n_fix} gold-fixes -> "
          f"{gold_fixes_path}; +{n_hard} hard examples ({n_neg} negatives) -> {hard_path}")


def _existing_ids(path):
    ids = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                ids.add(json.loads(line).get("id"))
    return ids


def main():
    ap = argparse.ArgumentParser(description="Model-assisted spot-check labeling harness.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queue")
    q.add_argument("--gold", required=True)
    q.add_argument("--out", default="reviews/spotcheck.jsonl")
    q.add_argument("--engine", choices=["hybrid", "base"], default="hybrid")
    q.add_argument("--cap", type=int, default=40)
    q.add_argument("--model-dir", default=None)
    q.add_argument("--limit", type=int, default=None)
    r = sub.add_parser("route")
    r.add_argument("--reviews", required=True)
    r.add_argument("--gold-fixes", default="data/gold_fixes.jsonl")
    r.add_argument("--hard", default="data/hard_examples.jsonl")
    args = ap.parse_args()
    if args.cmd == "queue":
        run_queue(args.gold, args.out, engine=args.engine, cap=args.cap,
                  model_dir=args.model_dir, limit=args.limit)
    else:
        route(args.reviews, gold_fixes_path=args.gold_fixes, hard_path=args.hard)


if __name__ == "__main__":
    main()
