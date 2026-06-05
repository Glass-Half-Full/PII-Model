"""Build a versioned gold set from public PII datasets + the Australian synthetic slice.

DEV-TIME (needs network for the public sources). Streams each source through its ingester,
maps labels -> GIRP, derives the gold tier via girp.classify_elements, validates every record,
assigns a deterministic train/dev/test split, and writes:

    data/gold/<version>/{train,dev,test}.jsonl
    data/gold/<version>/manifest.json   (per-source / per-tier / per-element counts, provenance)

Usage:
    python -m gold_data.build_gold --version v1 \
        --n-synth-au 500 --n-ai4privacy 500 --n-gretel 400 \
        --splits 0.4/0.2/0.4 --seed 12345
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import hashlib
import inspect
import json
import os

import girp
from gold_data import ingest
from gold_data.schema import to_jsonl, validate

AU_FLOOR_ELEMENTS = ["tax file number", "medicare number", "bank account number",
                     "phone number", "address"]
AU_FLOOR = 30  # each AU-critical element must have at least this many gold spans


def _girp_rules_sha() -> str:
    src = (inspect.getsource(girp.classify_elements) + repr(sorted(girp.GIRP_PII_LABELS))
           + repr(sorted(girp.SENSITIVE_LABELS)) + repr(sorted(girp.CONFIDENTIAL_ISOLATION_LABELS)))
    return "sha256:" + hashlib.sha256(src.encode()).hexdigest()[:16]


def _assign_split(rec_id, version, ratios):
    h = int(hashlib.sha1(f"{version}|{rec_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    train, dev, _test = ratios
    if h < train:
        return "train"
    if h < train + dev:
        return "dev"
    return "test"


def _ingest_source(name, gen, ratios, version, stats):
    """Drain a (record, unknown_labels) generator -> list[GoldRecord] with splits assigned.

    Resilient: if the source fails mid-stream (e.g. a network/rate-limit error), keep whatever
    was ingested so far and record the error, so one flaky source never loses the others.
    """
    out = []
    s = stats[name]
    try:
        for rec, unknown in gen:
            for u in unknown:
                s["unknown_labels"][u] += 1
            problems = validate(rec)
            if problems:
                s["invalid_skipped"] += 1
                continue
            rec = dataclasses.replace(rec, split=_assign_split(rec.id, version, ratios))
            out.append(rec)
            s["ingested"] += 1
    except Exception as e:  # noqa: BLE001 - keep partial results from a flaky remote source
        s["error"] = f"{type(e).__name__}: {e}"
        print(f"[build_gold] WARNING: source {name!r} stopped early: {s['error']} "
              f"(kept {len(out)} records)")
    return out


def build(version, n_synth_au, n_ai4privacy, n_gretel, kaggle_path, ratios, seed, out_dir):
    stats = collections.defaultdict(lambda: {"ingested": 0, "invalid_skipped": 0,
                                             "unknown_labels": collections.Counter()})
    records = []

    print(f"[build_gold] synthetic-au: target {n_synth_au}")
    records += _ingest_source("synthetic-au", ingest.ingest_synthetic_au(n_synth_au, seed=seed),
                              ratios, version, stats)
    if n_ai4privacy:
        print(f"[build_gold] ai4privacy (validation, en): target {n_ai4privacy}")
        records += _ingest_source("ai4privacy", ingest.ingest_ai4privacy(n_ai4privacy),
                                  ratios, version, stats)
    if n_gretel:
        print(f"[build_gold] gretel-finance (en, <=600 chars): target {n_gretel}")
        records += _ingest_source("gretel-finance", ingest.ingest_gretel_finance(n_gretel),
                                  ratios, version, stats)
    if kaggle_path:
        print(f"[build_gold] kaggle-pii (local file): {kaggle_path}")
        records += _ingest_source("kaggle-pii", ingest.ingest_kaggle_pii(kaggle_path),
                                  ratios, version, stats)

    # AU-element coverage check (informational). Synthetic-au is the ONLY source of AU IDs
    # (Medicare especially), so if a critical element falls under the floor, the manifest flags it
    # and the operator raises --n-synth-au — we do NOT silently inflate (that would skew tier balance).
    elem_counts = collections.Counter(s.label for r in records for s in r.spans)
    au_met = {e: elem_counts[e] >= AU_FLOOR for e in AU_FLOOR_ELEMENTS}

    # Write splits.
    vdir = os.path.join(out_dir, version)
    os.makedirs(vdir, exist_ok=True)
    by_split = collections.Counter()
    handles = {sp: open(os.path.join(vdir, f"{sp}.jsonl"), "w") for sp in ("train", "dev", "test")}
    try:
        for rec in records:
            handles[rec.split].write(to_jsonl(rec) + "\n")
            by_split[rec.split] += 1
    finally:
        for h in handles.values():
            h.close()

    manifest = {
        "version": version,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "split_ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
        "girp_rules_sha": _girp_rules_sha(),
        "sources": {name: {"ingested": s["ingested"], "invalid_skipped": s["invalid_skipped"],
                           "unknown_labels": dict(s["unknown_labels"]), "error": s.get("error")}
                    for name, s in stats.items()},
        "counts": {
            "total": len(records),
            "by_split": dict(by_split),
            "by_tier": dict(collections.Counter(r.gold_level for r in records)),
            "by_split_tier": {sp: dict(collections.Counter(r.gold_level for r in records if r.split == sp))
                              for sp in ("train", "dev", "test")},
            "by_element": dict(elem_counts),
        },
        "au_floor": {"floor": AU_FLOOR, "met": au_met},
        "notes": "Public sources are PII-dense/synthetic-realistic; real-prose anchor (Kaggle PIILO) "
                 "needs Kaggle credentials and is wired but not downloaded here.",
    }
    with open(os.path.join(vdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[build_gold] wrote {len(records)} records to {vdir}")
    print(f"  by split: {dict(by_split)}")
    print(f"  by tier:  {manifest['counts']['by_tier']}")
    print(f"  AU floor met: {au_met}")
    for name, s in stats.items():
        if s["unknown_labels"]:
            print(f"  [{name}] unknown labels (dropped): {dict(s['unknown_labels'])}")
    return manifest


def _parse_ratios(s):
    parts = [float(x) for x in s.split("/")]
    assert len(parts) == 3 and abs(sum(parts) - 1.0) < 1e-6, "splits must be three fractions summing to 1"
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--n-synth-au", type=int, default=800)
    ap.add_argument("--n-ai4privacy", type=int, default=500)
    ap.add_argument("--n-gretel", type=int, default=400)
    ap.add_argument("--kaggle-path", default=None, help="local Kaggle/PIILO json (optional)")
    ap.add_argument("--splits", type=_parse_ratios, default="0.4/0.2/0.4")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", default="data/gold")
    args = ap.parse_args()
    ratios = args.splits if isinstance(args.splits, tuple) else _parse_ratios(args.splits)
    build(args.version, args.n_synth_au, args.n_ai4privacy, args.n_gretel,
          args.kaggle_path, ratios, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
