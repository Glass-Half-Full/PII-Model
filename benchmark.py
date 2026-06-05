"""Performance baseline against a public *labeled* PII dataset.

NOTE: this is a dev/evaluation tool and needs INTERNET + `pip install datasets` to download the
benchmark (ai4privacy/pii-masking-200k). The classifier itself (girp.py) runs fully offline; this
script only exists to measure/track performance. Run it on a machine with internet:

    python -m pip install datasets
    python benchmark.py             # 2000 rows, threshold 0.5
    python benchmark.py 1000 0.7    # rows, threshold

It reports per-element precision/recall/F1 (presence-level) and end-to-end GIRP-level accuracy
versus a gold level derived from the dataset's labels.
"""
from __future__ import annotations
import sys, collections
from girp import load_local_model, classify_columns, classify_elements, RANK

# ai4privacy gold labels -> GIRP element labels (personal elements only).
GOLD2GIRP = {
    "FIRSTNAME": "person", "LASTNAME": "person", "MIDDLENAME": "person", "PREFIX": "person",
    "EMAIL": "email address", "PHONENUMBER": "phone number", "DOB": "date of birth",
    "STREET": "address", "BUILDINGNUMBER": "address", "SECONDARYADDRESS": "address",
    "ACCOUNTNUMBER": "bank account number", "IBAN": "bank account number",
    "CREDITCARDNUMBER": "credit card number", "SSN": "tax file number",
}
CORE = ["person", "email address", "phone number", "credit card number",
        "date of birth", "address", "bank account number"]


def load_eval(n):
    from datasets import load_dataset
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True)
    rows = []
    for ex in ds:
        if ex.get("language") != "en":
            continue
        gold = {GOLD2GIRP[m["label"]] for m in ex["privacy_mask"] if m["label"] in GOLD2GIRP}
        rows.append((ex["source_text"], gold))
        if len(rows) >= n:
            break
    return rows


def main(n=2000, threshold=0.5):
    import pandas as pd
    rows = load_eval(n)
    print(f"eval rows: {len(rows)}  threshold: {threshold}")
    df = pd.DataFrame({"text": [t for t, _ in rows]})
    gold_sets = [g for _, g in rows]
    gold_level = [classify_elements(g) for g in gold_sets]

    model, dev = load_local_model()
    out = classify_columns(model, df, ["text"], threshold=threshold)
    msets = [set(e) for e in out["text_girp_elements"]]
    mlvl = list(out["text_girp_level"])

    print(f"\n{'element':22s} {'P%':>6} {'R%':>6} {'F1%':>6}   TP/FP/FN")
    mtp = mfp = mfn = 0
    for t in CORE:
        tp = fp = fn = 0
        for g, m in zip(gold_sets, msets):
            gh, mh = t in g, t in m
            tp += gh and mh; fp += mh and not gh; fn += gh and not mh
        P = tp/(tp+fp) if tp+fp else 0.0
        R = tp/(tp+fn) if tp+fn else 0.0
        F = 2*P*R/(P+R) if P+R else 0.0
        mtp += tp; mfp += fp; mfn += fn
        print(f"{t:22s} {P*100:6.1f} {R*100:6.1f} {F*100:6.1f}   {tp}/{fp}/{fn}")
    P = mtp/(mtp+mfp) if mtp+mfp else 0
    R = mtp/(mtp+mfn) if mtp+mfn else 0
    print(f"{'MICRO-AVG':22s} {P*100:6.1f} {R*100:6.1f} {2*P*R/(P+R)*100 if P+R else 0:6.1f}   {mtp}/{mfp}/{mfn}")

    acc = sum(a == b for a, b in zip(gold_level, mlvl))/len(rows)
    over = sum(RANK[m] > RANK[g] for g, m in zip(gold_level, mlvl))/len(rows)
    under = sum(RANK[m] < RANK[g] for g, m in zip(gold_level, mlvl))/len(rows)
    print(f"\nGIRP level accuracy {acc*100:.1f}%   over-classified {over*100:.1f}%   under-classified {under*100:.1f}%")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    main(n, th)
