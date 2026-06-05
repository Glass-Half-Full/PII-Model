"""Weak-labeling + review-queue tool — the data engine for the recursive enhancement loop.

Runs the hybrid over unlabeled text, emits candidate GIRP labels + per-engine detections, and flags
rows for human review where the two engines DISAGREE (the most informative cases). Corrected rows
become gold training data for `train_lora.py`. Fully local/offline.

    python weak_label.py --in unlabeled.txt --out candidates.jsonl
    # a reviewer fixes the `needs_review` rows -> append to data/gold.jsonl -> fine-tune -> re-eval
"""
import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from girp import found_labels, classify_elements


def label_text(model, analyzer, text, threshold=0.7):
    from aupii import GLINER_FUZZY_GROUPS, presidio_elements, _regex_phone
    merged = {}
    for labs, thr_override in GLINER_FUZZY_GROUPS:
        thr = thr_override if thr_override is not None else threshold
        for lbl, vals in model.extract_entities(text, labs, threshold=thr)["entities"].items():
            if vals:
                merged.setdefault(lbl, []).extend(vals)
    g = found_labels(merged) | _regex_phone(text)        # ML-side elements
    p = presidio_elements(analyzer, text)                # checksum/regex-side elements
    found = g | p
    level = classify_elements(found)
    # The two engines disagreeing on the resulting level is the highest-value review signal.
    needs_review = classify_elements(g) != classify_elements(p)
    return {
        "text": text,
        "girp_level": level,
        "elements": sorted(found),
        "gliner_elements": sorted(g),
        "presidio_elements": sorted(p),
        "needs_review": bool(needs_review),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="text file, one row per line")
    ap.add_argument("--out", default="candidates.jsonl")
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    from aupii import load_hybrid
    rows = [l.strip() for l in open(args.inp) if l.strip()]
    model, analyzer, dev = load_hybrid()
    n_review = 0
    with open(args.out, "w") as f:
        for t in rows:
            rec = label_text(model, analyzer, t, args.threshold)
            n_review += rec["needs_review"]
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(rows)} candidates to {args.out} ({n_review} flagged needs_review) on {dev}")


if __name__ == "__main__":
    main()
