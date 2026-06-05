"""Unified offline evaluation harness for the GIRP PII classifier.

Supersedes benchmark.py. Computes, against a versioned gold set (gold_data.build_gold):
  * balanced GIRP accuracy (mean of the four per-tier recalls) + the full 4x4 confusion matrix,
  * over/under-classification rates overall and per gold tier (+ health-tier under-classification),
  * per-entity precision/recall/F1 at BOTH presence-level and span-level (char-overlap IoU),
  * bootstrap 95% confidence intervals on the headline metrics,
  * a confidence-based threshold sweep (ONE inference pass) recommending the balanced-accuracy-
    optimal global threshold subject to an under-classification cap.

The headline metric is BALANCED GIRP accuracy: it penalises the model's ~30% over-classification
(which shows up as low recall on the lower tiers it leaks from) while still tracking the dangerous
direction (under-classification) separately.

Prediction reuses the exact production decision path (girp.found_labels / is_valid_entity /
regex_elements / classify_elements, aupii.presidio_*), so the harness measures the real model.
Char offsets + confidences come from gliner2's include_spans / include_confidence. The model loads
from LOCAL weights only; this is dev-time tooling and writes reports under data/eval/<version>/.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

import girp
from girp import (LEVELS, RANK, SENSITIVE_LABELS, classify_elements, found_labels,
                  is_valid_entity, regex_elements)
from gold_data.schema import from_jsonl

FLOOR = 0.3          # extract gliner spans down to this confidence; sweep filters upward
HEALTH_THR = 0.4     # fixed health-pass override (matches aupii.GLINER_FUZZY_GROUPS)
HEALTH_LABELS = set(SENSITIVE_LABELS)


# ---------------------------------------------------------------------------
# Gold loading
# ---------------------------------------------------------------------------
def load_gold(path, limit=None):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(from_jsonl(line))
            if limit and len(recs) >= limit:
                break
    return recs


# ---------------------------------------------------------------------------
# Prediction — rich (offset + confidence) extraction, cached once per text
# ---------------------------------------------------------------------------
def _gliner_rich(model, texts, groups, batch_size, progress, desc):
    """Per-text list of (label, start, end, conf) over all label groups, extracted at FLOOR."""
    from girp import _robust_batch_extract
    per_text = [[] for _ in texts]
    for gi, grp in enumerate(groups, 1):
        labels = list(grp[0]) if isinstance(grp, tuple) else list(grp)
        res = _robust_batch_extract(model, texts, labels, FLOOR, batch_size, progress,
                                    f"{desc}[pass {gi}/{len(groups)}]",
                                    include_spans=True, include_confidence=True)
        for i, r in enumerate(res):
            for lbl, vals in r["entities"].items():
                for v in (vals or []):
                    if isinstance(v, dict) and v.get("start") is not None:
                        per_text[i].append((lbl, int(v["start"]), int(v["end"]), float(v["confidence"])))
    return per_text


def predict_rich(engine, texts, batch_size=None, progress=True, model_dir=None):
    """Return (meta, gliner_per_text, presidio_per_text).

    gliner_per_text[i] = [(label, start, end, conf), ...]   (extracted at FLOOR)
    presidio_per_text[i] = [(label, start, end, score), ...] (hybrid only; [] for base)
    """
    if engine == "hybrid":
        from aupii import GLINER_FUZZY_GROUPS, load_hybrid, presidio_spans
        model, analyzer, dev = load_hybrid(model_dir)
        bs = batch_size or girp._auto_batch_size(model)
        gliner = _gliner_rich(model, texts, GLINER_FUZZY_GROUPS, bs, progress, "hybrid")
        presidio = [presidio_spans(analyzer, t) for t in texts]
        return {"engine": "hybrid", "device": dev}, gliner, presidio
    else:
        from girp import DETECTION_GROUPS, load_local_model
        model, dev = load_local_model(model_dir)
        bs = batch_size or girp._auto_batch_size(model)
        gliner = _gliner_rich(model, texts, DETECTION_GROUPS, bs, progress, "base")
        return {"engine": "base", "device": dev}, gliner, [[] for _ in texts]


def derive(gliner_spans, presidio_spans, text, threshold, engine, validate=True):
    """Apply a global threshold to cached rich spans -> (elements set, level, pred_spans).

    Health labels use the fixed HEALTH_THR; all other gliner labels use ``threshold``. The element
    set and level are computed by the SAME production functions used at inference time.
    """
    from aupii import SUPPRESSED_FUZZY_LABELS, _regex_phone
    merged = collections.defaultdict(list)
    kept = []
    for (lbl, s, e, c) in gliner_spans:
        thr = HEALTH_THR if lbl in HEALTH_LABELS else threshold
        if c >= thr:
            merged[lbl].append(text[s:e])
            kept.append((lbl, s, e))
    if engine == "hybrid":
        fuzzy = (found_labels(dict(merged), validate=validate) | _regex_phone(text)) - SUPPRESSED_FUZZY_LABELS
        structured = {g for (g, _, _, _) in presidio_spans}
        elements = fuzzy | structured
    else:
        elements = found_labels(dict(merged), validate=validate) | regex_elements(text)
    level = classify_elements(elements)
    suppressed = SUPPRESSED_FUZZY_LABELS if engine == "hybrid" else frozenset()
    pred_spans = [(lbl, s, e) for (lbl, s, e) in kept
                  if lbl not in suppressed and (not validate or is_valid_entity(lbl, text[s:e]))]
    if engine == "hybrid":
        pred_spans += [(g, s, e) for (g, s, e, _sc) in presidio_spans]
    return elements, level, pred_spans


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def confusion(gold_levels, pred_levels):
    cm = np.zeros((4, 4), dtype=int)
    for g, p in zip(gold_levels, pred_levels):
        cm[RANK[g]][RANK[p]] += 1
    return cm


def balanced_accuracy(cm):
    recalls, per_tier = [], {}
    for i, lvl in enumerate(LEVELS):
        support = cm[i].sum()
        r = cm[i][i] / support if support else None
        per_tier[lvl] = None if r is None else round(float(r), 4)
        if r is not None:
            recalls.append(r)
    return (float(np.mean(recalls)) if recalls else 0.0), per_tier


def _bal_from_ranks(g, p):
    recalls = []
    for i in range(4):
        mask = g == i
        support = mask.sum()
        if support:
            recalls.append((p[mask] == i).mean())
    return float(np.mean(recalls)) if recalls else 0.0


def rates(gold_levels, pred_levels):
    n = len(gold_levels)
    gr = np.array([RANK[g] for g in gold_levels])
    pr = np.array([RANK[p] for p in pred_levels])
    over = float((pr > gr).mean()) if n else 0.0
    under = float((pr < gr).mean()) if n else 0.0
    h_total = int((gr == 3).sum())
    h_under = int(((gr == 3) & (pr < gr)).sum())
    return {"accuracy": float((pr == gr).mean()) if n else 0.0,
            "over": over, "under": under,
            "health_under": (h_under / h_total) if h_total else 0.0,
            "health_total": h_total}


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def presence_prf(gold_sets, pred_sets):
    labels = sorted(set().union(*gold_sets, *pred_sets)) if gold_sets else []
    out = {}
    for lab in labels:
        tp = fp = fn = 0
        for g, m in zip(gold_sets, pred_sets):
            gh, mh = lab in g, lab in m
            tp += gh and mh
            fp += mh and not gh
            fn += gh and not mh
        p, r, f = _prf(tp, fp, fn)
        out[lab] = {"P": p, "R": r, "F1": f, "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}
    return out


def _iou(a, b):
    s, e = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def span_prf(gold_spans_all, pred_spans_all, iou_thresh=0.5):
    tp = collections.Counter()
    fp = collections.Counter()
    fn = collections.Counter()
    for gold, pred in zip(gold_spans_all, pred_spans_all):
        gby = collections.defaultdict(list)
        pby = collections.defaultdict(list)
        for (l, s, e) in gold:
            gby[l].append((s, e))
        for (l, s, e) in pred:
            pby[l].append((s, e))
        for lab in set(gby) | set(pby):
            gs, ps = gby[lab], pby[lab]
            matched = set()
            for (ps_, pe_) in ps:
                best, best_iou = -1, iou_thresh
                for gj, (gs_, ge_) in enumerate(gs):
                    if gj in matched:
                        continue
                    iou = _iou((ps_, pe_), (gs_, ge_))
                    if iou >= best_iou:
                        best, best_iou = gj, iou
                if best >= 0:
                    matched.add(best)
                    tp[lab] += 1
                else:
                    fp[lab] += 1
            fn[lab] += len(gs) - len(matched)
    out = {}
    for lab in sorted(set(tp) | set(fp) | set(fn)):
        p, r, f = _prf(tp[lab], fp[lab], fn[lab])
        out[lab] = {"P": p, "R": r, "F1": f, "tp": tp[lab], "fp": fp[lab], "fn": fn[lab],
                    "support": tp[lab] + fn[lab]}
    return out


def bootstrap_ci(gold_levels, pred_levels, n_boot=2000, seed=12345):
    rng = np.random.default_rng(seed)
    g = np.array([RANK[x] for x in gold_levels])
    p = np.array([RANK[x] for x in pred_levels])
    n = len(g)
    accs, bals, unders = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gi, pi = g[idx], p[idx]
        accs.append((gi == pi).mean())
        unders.append((pi < gi).mean())
        bals.append(_bal_from_ranks(gi, pi))

    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {"accuracy": ci(accs), "balanced_accuracy": ci(bals), "under": ci(unders)}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def girp_rules_sha():
    src = (inspect.getsource(girp.classify_elements) + inspect.getsource(girp.is_valid_entity)
           + repr(sorted(girp.GIRP_PII_LABELS)) + repr(sorted(girp.SENSITIVE_LABELS)))
    return "sha256:" + hashlib.sha256(src.encode()).hexdigest()[:16]


def weights_sha256(model_dir=None):
    path = os.path.join(model_dir or girp.default_model_dir(), "model.safetensors")
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def evaluate(gold_path, engine="hybrid", threshold=0.7, sweep=None, span_iou=0.5,
             n_boot=2000, seed=12345, limit=None, batch_size=None, progress=True, model_dir=None):
    recs = load_gold(gold_path, limit=limit)
    texts = [r.text for r in recs]
    gold_levels = [r.gold_level for r in recs]
    gold_sets = [set(r.gold_elements) for r in recs]
    gold_spans = [[(s.label, s.start, s.end) for s in r.spans] for r in recs]

    meta, gliner, presidio = predict_rich(engine, texts, batch_size, progress, model_dir)

    def at(threshold_):
        elements, levels, spans = [], [], []
        for i, t in enumerate(texts):
            el, lvl, sp = derive(gliner[i], presidio[i], t, threshold_, meta["engine"])
            elements.append(el)
            levels.append(lvl)
            spans.append(sp)
        cm = confusion(gold_levels, levels)
        bal, per_tier = balanced_accuracy(cm)
        rt = rates(gold_levels, levels)
        return {"threshold": round(threshold_, 3), "confusion": cm.tolist(),
                "balanced_accuracy": round(bal, 4), "per_tier_recall": per_tier, **rt,
                "_levels": levels, "_elements": elements, "_spans": spans}

    sweep_thresholds = sweep or [round(threshold, 3)]
    if round(threshold, 3) not in sweep_thresholds:
        sweep_thresholds = sorted(set(sweep_thresholds) | {round(threshold, 3)})
    sweep_results = [at(t) for t in sweep_thresholds]

    primary = next(s for s in sweep_results if s["threshold"] == round(threshold, 3))
    presence = presence_prf(gold_sets, primary["_elements"])
    spans_m = span_prf(gold_spans, primary["_spans"], iou_thresh=span_iou)
    ci = bootstrap_ci(gold_levels, primary["_levels"], n_boot=n_boot, seed=seed)

    result = {
        "config": {
            "engine": meta["engine"], "device": meta.get("device"), "threshold": threshold,
            "floor": FLOOR, "health_threshold": HEALTH_THR, "span_iou": span_iou,
            "gold_path": gold_path, "n": len(recs), "seed": seed, "n_boot": n_boot,
            "girp_rules_sha": girp_rules_sha(), "weights_sha256": weights_sha256(model_dir),
        },
        "headline": {
            "balanced_accuracy": primary["balanced_accuracy"],
            "balanced_accuracy_ci95": ci["balanced_accuracy"],
            "accuracy": primary["accuracy"], "accuracy_ci95": ci["accuracy"],
            "over": primary["over"], "under": primary["under"], "under_ci95": ci["under"],
            "health_under": primary["health_under"], "per_tier_recall": primary["per_tier_recall"],
        },
        "confusion_matrix": {"levels": LEVELS, "rows_are_gold": True, "matrix": primary["confusion"]},
        "per_entity_presence": presence,
        "per_entity_span": spans_m,
        "sweep": [{k: v for k, v in s.items() if not k.startswith("_")} for s in sweep_results],
    }
    # per-row mismatches at the primary threshold (feeds the Claude-review loop).
    # Each carries a FLOOR probe: of the gold elements the model missed at the primary threshold,
    # which were still surfaced at FLOOR (0.3) — "rescuable" by threshold/validation tuning (Stage 1)
    # — versus never surfaced even at FLOOR — a real detection gap needing fine-tuning (Stage 2).
    mismatches = []
    for i, r in enumerate(recs):
        pl = primary["_levels"][i]
        if pl == r.gold_level:
            continue
        floor_elems, floor_level, _ = derive(gliner[i], presidio[i], r.text, FLOOR, meta["engine"])
        gold_e = set(r.gold_elements)
        pred_e = set(primary["_elements"][i])
        floor_e = set(floor_elems)
        missed = gold_e - pred_e
        spurious = pred_e - gold_e
        mismatches.append({
            "id": r.id, "text": r.text, "source": r.source,
            "gold": {"level": r.gold_level, "elements": sorted(gold_e),
                     "spans": [[s.label, s.start, s.end] for s in r.spans]},
            "pred": {"level": pl, "elements": sorted(pred_e),
                     "spans": [[l, s, e] for (l, s, e) in primary["_spans"][i]]},
            "direction": ("under" if RANK[pl] < RANK[r.gold_level] else "over"),
            "probe": {
                "missed_elements": sorted(missed),
                "spurious_elements": sorted(spurious),
                "rescuable_at_floor": sorted(missed & floor_e),    # threshold/validation -> Stage 1
                "detection_gap": sorted(missed - floor_e),         # never surfaced -> Stage 2
                "floor_level": floor_level,
            },
        })
    return result, mismatches


def recommend_threshold(result, under_cap=0.10):
    """Pick the sweep threshold maximising balanced accuracy with under-classification <= cap."""
    ok = [s for s in result["sweep"] if s["under"] <= under_cap]
    pool = ok or result["sweep"]
    best = max(pool, key=lambda s: s["balanced_accuracy"])
    return best["threshold"], bool(ok)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _md_confusion(cm):
    lines = ["| gold \\ pred | " + " | ".join(L[:8] for L in LEVELS) + " |",
             "|---|" + "---|" * 4]
    for i, L in enumerate(LEVELS):
        lines.append(f"| **{L[:8]}** | " + " | ".join(str(cm[i][j]) for j in range(4)) + " |")
    return "\n".join(lines)


def _md_entity_table(title, table):
    lines = [f"### {title}", "", "| element | P% | R% | F1% | support | tp/fp/fn |",
             "|---|---:|---:|---:|---:|---|"]
    for lab, m in sorted(table.items(), key=lambda kv: -kv[1]["support"]):
        lines.append(f"| {lab} | {m['P']*100:.1f} | {m['R']*100:.1f} | {m['F1']*100:.1f} | "
                     f"{m['support']} | {m['tp']}/{m['fp']}/{m['fn']} |")
    return "\n".join(lines)


def write_report(result, out_dir, mismatches=None, under_cap=0.10):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    if mismatches is not None:
        with open(os.path.join(out_dir, "mismatches.jsonl"), "w") as f:
            for m in mismatches:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    h = result["headline"]
    cfg = result["config"]
    rec_thr, within = recommend_threshold(result, under_cap)
    cm = result["confusion_matrix"]["matrix"]
    L = [
        f"# Evaluation report — {cfg['engine']} @ threshold {cfg['threshold']}",
        "",
        f"Gold: `{cfg['gold_path']}`  ·  n={cfg['n']}  ·  device={cfg['device']}  ·  rules={cfg['girp_rules_sha']}",
        "",
        "## Headline (balanced GIRP accuracy)",
        "",
        f"- **Balanced GIRP accuracy: {h['balanced_accuracy']*100:.1f}%**  "
        f"(95% CI {h['balanced_accuracy_ci95'][0]*100:.1f}–{h['balanced_accuracy_ci95'][1]*100:.1f})",
        f"- Plain accuracy: {h['accuracy']*100:.1f}%  "
        f"(95% CI {h['accuracy_ci95'][0]*100:.1f}–{h['accuracy_ci95'][1]*100:.1f})",
        f"- Over-classification: {h['over']*100:.1f}%   ·   "
        f"Under-classification (dangerous): {h['under']*100:.1f}% "
        f"(95% CI {h['under_ci95'][0]*100:.1f}–{h['under_ci95'][1]*100:.1f})",
        f"- Health-tier under-classification: {h['health_under']*100:.1f}%",
        f"- Per-tier recall: " + ", ".join(f"{k} {v if v is None else round(v*100,1)}%"
                                           for k, v in h["per_tier_recall"].items()),
        "",
        "## Confusion matrix (rows = gold tier, columns = predicted)",
        "",
        _md_confusion(cm),
        "",
        "## Threshold sweep",
        "",
        "| threshold | balanced acc% | accuracy% | over% | under% | health-under% |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in result["sweep"]:
        L.append(f"| {s['threshold']} | {s['balanced_accuracy']*100:.1f} | {s['accuracy']*100:.1f} | "
                 f"{s['over']*100:.1f} | {s['under']*100:.1f} | {s['health_under']*100:.1f} |")
    L += [
        "",
        f"**Recommended threshold: {rec_thr}** (max balanced accuracy with under-classification "
        f"≤ {under_cap*100:.0f}%{'' if within else '; cap not satisfiable — min-under chosen'}).",
        "",
        _md_entity_table("Per-entity — presence-level", result["per_entity_presence"]),
        "",
        _md_entity_table(f"Per-entity — span-level (IoU≥{cfg['span_iou']})", result["per_entity_span"]),
        "",
    ]
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return os.path.join(out_dir, "REPORT.md")


def _parse_sweep(s):
    if not s:
        return None
    lo, hi, step = (float(x) for x in s.split(":"))
    out, v = [], lo
    while v <= hi + 1e-9:
        out.append(round(v, 3))
        v += step
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold/v1/test.jsonl")
    ap.add_argument("--engine", choices=["hybrid", "base"], default="hybrid")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--sweep", default=None, help="lo:hi:step, e.g. 0.3:0.9:0.1")
    ap.add_argument("--span-iou", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--under-cap", type=float, default=0.10)
    ap.add_argument("--out-version", default=None, help="writes data/eval/<version>/")
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    result, mismatches = evaluate(
        args.gold, engine=args.engine, threshold=args.threshold, sweep=_parse_sweep(args.sweep),
        span_iou=args.span_iou, n_boot=args.bootstrap, seed=args.seed, limit=args.limit,
        model_dir=args.model_dir)
    version = args.out_version or "latest"
    out_dir = os.path.join("data", "eval", version)
    path = write_report(result, out_dir, mismatches=mismatches, under_cap=args.under_cap)
    h = result["headline"]
    print(f"\nbalanced GIRP accuracy {h['balanced_accuracy']*100:.1f}% "
          f"(CI {h['balanced_accuracy_ci95'][0]*100:.1f}-{h['balanced_accuracy_ci95'][1]*100:.1f}); "
          f"under {h['under']*100:.1f}%; over {h['over']*100:.1f}%")
    print(f"report: {path}  ·  mismatches: {len(mismatches)}")


if __name__ == "__main__":
    main()
