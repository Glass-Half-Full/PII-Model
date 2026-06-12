"""Build a HELD-OUT real-free-text eval set (test-only) for binary PII-present precision.

DEV-TIME (needs network for TAB; Kaggle PIILO is a local file). Unlike ``build_gold``, this writes
ONLY a test split — every record is forced to ``split="test"`` — to ``data/gold/<version>/``, so the
real set is structurally impossible to train on. It exists purely to measure the HONEST binary
precision/recall of the model on real complex free text.

Sources:
  * kaggle-pii (PIILO student essays) — REAL, PII-SPARSE prose. Supplies the PII-ABSENT rows that
    make binary precision measurable. Needs a local json (download with Kaggle creds, see --help).
  * tab (Text Anonymization Benchmark, ECHR judgments) — REAL, PII-DENSE legal prose: a recall +
    per-entity-precision (distractor) stress test. ECHR docs always name people, so TAB adds few
    PII-absent rows on its own.

Usage:
    python -m gold_data.build_real --version real-v1 \
        --kaggle-path data/_raw/kaggle-pii/train.json --n-kaggle 1500 --n-tab 1000
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os

from gold_data import ingest
from gold_data.build_gold import _girp_rules_sha, _ingest_source
from gold_data.schema import to_jsonl

ALL_TEST = (0.0, 0.0, 1.0)   # _assign_split with these ratios -> every record lands in "test"


def build(version, kaggle_path, n_kaggle, n_tab, out_dir):
    stats = collections.defaultdict(lambda: {"ingested": 0, "invalid_skipped": 0,
                                             "unknown_labels": collections.Counter()})
    records = []
    if kaggle_path:
        print(f"[build_real] kaggle-pii (local file): {kaggle_path}")
        records += _ingest_source("kaggle-pii", ingest.ingest_kaggle_pii(kaggle_path, n=n_kaggle),
                                  ALL_TEST, version, stats)
    else:
        print("[build_real] kaggle-pii: SKIPPED (no --kaggle-path). The PII-absent rows that make "
              "binary precision measurable come from PIILO — download it to complete the baseline.")
    if n_tab:
        print(f"[build_real] tab (ECHR judgments, test split): target {n_tab}")
        records += _ingest_source("tab", ingest.ingest_tab(n_tab, split="test"),
                                  ALL_TEST, version, stats)

    # de-dup by stable id (build_record makes id = sha1(source|text)) so re-runs/overlap are idempotent
    seen, deduped = set(), []
    for r in records:
        if r.id not in seen:
            seen.add(r.id)
            deduped.append(r)
    records = deduped

    present = sum(1 for r in records if len(r.gold_elements) > 0)   # Option A: any PII element present
    level_present = sum(1 for r in records if r.gold_level != "Public")

    vdir = os.path.join(out_dir, version)
    os.makedirs(vdir, exist_ok=True)
    with open(os.path.join(vdir, "test.jsonl"), "w") as f:        # ONLY a test split — no train/dev
        for rec in records:
            f.write(to_jsonl(rec) + "\n")

    elem_counts = collections.Counter(s.label for r in records for s in r.spans)
    manifest = {
        "version": version,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "split": "test-only (held out; never trained)",
        "girp_rules_sha": _girp_rules_sha(),
        "sources": {name: {"ingested": s["ingested"], "invalid_skipped": s["invalid_skipped"],
                           "unknown_labels": dict(s["unknown_labels"]), "error": s.get("error")}
                    for name, s in stats.items()},
        "counts": {
            "total": len(records),
            "pii_present": present,                   # len(gold_elements) > 0  (binary Option A)
            "pii_absent": len(records) - present,     # the precision-measuring rows
            "pii_present_by_level": level_present,    # level != "Public" (secondary definition)
            "by_tier": dict(collections.Counter(r.gold_level for r in records)),
            "by_source": dict(collections.Counter(r.source for r in records)),
            "by_element": dict(elem_counts),
        },
        "notes": "Held-out real-free-text eval for BINARY PII-present precision. PII-absent rows "
                 "(needed to measure precision) come from PIILO/kaggle-pii; TAB is PII-dense legal "
                 "prose (recall + distractor stress). If pii_absent is ~0, add --kaggle-path PIILO.",
    }
    with open(os.path.join(vdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[build_real] wrote {len(records)} records to {vdir}/test.jsonl")
    print(f"  PII present/absent: {present}/{len(records) - present}  "
          f"·  by source: {manifest['counts']['by_source']}")
    print(f"  by tier: {manifest['counts']['by_tier']}")
    if present and (len(records) - present) < max(20, 0.05 * len(records)):
        print("  WARNING: very few PII-absent rows — binary precision will be optimistic. "
              "Add PIILO (--kaggle-path) for sparse/absent real essays.")
    for name, s in stats.items():
        if s["unknown_labels"]:
            print(f"  [{name}] unknown labels (dropped): {dict(s['unknown_labels'])}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="real-v1")
    ap.add_argument("--kaggle-path", default=None,
                    help="local PIILO json (download: kaggle competitions download -c "
                         "pii-detection-removal-from-educational-data, then unzip train.json); "
                         "supplies the PII-absent rows for precision")
    ap.add_argument("--n-kaggle", type=int, default=1500)
    ap.add_argument("--n-tab", type=int, default=1000)
    ap.add_argument("--out-dir", default="data/gold")
    args = ap.parse_args()
    build(args.version, args.kaggle_path, args.n_kaggle, args.n_tab, args.out_dir)


if __name__ == "__main__":
    main()
