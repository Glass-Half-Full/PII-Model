"""Ingest public PII datasets into the common gold schema (DEV-TIME; needs network).

Remote sources are pulled via the HuggingFace datasets-server HTTP ``/rows`` API — reliable,
paginated, and (unlike ``load_dataset(streaming=True)``) it does not stall downloading large
parquet shards. The Australian slice comes from the local synthetic generator. All output is
local JSONL consumed OFFLINE by evaluate.py / train_lora.py; production code never imports this.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from gold_data.mappings import is_known, map_label
from gold_data.schema import build_record

_DS_ROWS = "https://datasets-server.huggingface.co/rows"
_CACHE_DIR = "data/_cache"   # cached raw pages -> re-runs are fast and offline after first fetch


def _cache_path(dataset, config, split, offset, length):
    safe = dataset.replace("/", "_")
    h = hashlib.sha1(f"{dataset}|{config}|{split}|{offset}|{length}".encode()).hexdigest()[:10]
    return os.path.join(_CACHE_DIR, safe, f"{split}_{offset}_{length}_{h}.json")


def _fetch_page(dataset, config, split, offset, length, retries=6):
    """Fetch one page from datasets-server with on-disk caching and 429-aware backoff."""
    cp = _cache_path(dataset, config, split, offset, length)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    qs = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                 "offset": offset, "length": length})
    url = f"{_DS_ROWS}?{qs}"
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            with open(cp, "w") as f:
                json.dump(data, f)
            return data
        except urllib.error.HTTPError as e:  # rate limit / transient server errors
            last = e
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", 0)) or 12 * (attempt + 1)
                print(f"[ingest] 429 rate-limited at offset={offset}; backing off {wait}s")
                time.sleep(wait)
            elif e.code >= 500:
                time.sleep(3 * (attempt + 1))
            else:
                raise
        except Exception as e:  # noqa: BLE001 - dev-time tool; retry transient network errors
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"datasets-server fetch failed: {dataset} offset={offset}: {last}")


def _iter_raw(dataset, config, split, max_scan, page=100, pause=0.6):
    """Yield up to ``max_scan`` raw row dicts from datasets-server, paginated (cached)."""
    offset = 0
    while offset < max_scan:
        data = _fetch_page(dataset, config, split, offset, min(page, max_scan - offset))
        rows = data.get("rows", [])
        if not rows:
            break
        for rw in rows:
            yield rw["row"]
        offset += len(rows)
        total = data.get("num_rows_total")
        if total is not None and offset >= total:
            break
        time.sleep(pause)


def _spans_from(text, entries, source):
    """Map a list of {label,start,end} entries to GIRP raw spans; return (raw_spans, unknown_labels)."""
    raw_spans, unknown = [], []
    if isinstance(entries, str):
        try:
            entries = json.loads(entries)
        except Exception:
            entries = []
    for m in entries or []:
        if not isinstance(m, dict):
            continue
        lab = m.get("label")
        start, end = m.get("start"), m.get("end")
        if start is None or end is None:
            continue
        if not is_known(source, lab):
            unknown.append(lab)
            continue
        girp = map_label(source, lab)
        if girp is None:
            continue
        start, end = int(start), int(end)
        if not (0 <= start < end <= len(text)):
            continue
        raw_spans.append((girp, start, end, lab))
    return raw_spans, unknown


# --- public sources --------------------------------------------------------
def ingest_ai4privacy(n, split="validation", max_scan=None, lang="en"):
    """ai4privacy/open-pii-masking-500k (English subset). Yields (GoldRecord, unknown_labels)."""
    dataset = "ai4privacy/open-pii-masking-500k-ai4privacy"
    max_scan = max_scan or n * 10
    got = 0
    for row in _iter_raw(dataset, "default", split, max_scan):
        if row.get("language") != lang:
            continue
        text = row.get("source_text") or ""
        if not text:
            continue
        raw_spans, unknown = _spans_from(text, row.get("privacy_mask"), "ai4privacy")
        yield build_record(text, raw_spans, source="ai4privacy", split=""), unknown
        got += 1
        if got >= n:
            break


def ingest_gretel_finance(n, split="train", max_scan=None, max_chars=600, lang="English"):
    """gretelai/synthetic_pii_finance_multilingual (English, shorter docs). Yields (rec, unknown)."""
    dataset = "gretelai/synthetic_pii_finance_multilingual"
    max_scan = max_scan or n * 8
    got = 0
    for row in _iter_raw(dataset, "default", split, max_scan):
        if row.get("language") != lang:
            continue
        text = row.get("generated_text") or ""
        if not text or (max_chars and len(text) > max_chars):
            continue
        raw_spans, unknown = _spans_from(text, row.get("pii_spans"), "gretel-finance")
        yield build_record(text, raw_spans, source="gretel-finance", split=""), unknown
        got += 1
        if got >= n:
            break


def ingest_tab(n, split="test", max_scan=None):
    """Text Anonymization Benchmark — ECHR court judgments (real long-form legal prose).

    ``mattmdjaga/text-anonymization-benchmark-val-test``, fetched via datasets-server like the
    other public sources. Conservative mapping (PERSON -> person; coarse/quasi types dropped)
    means TAB rows are person-name PII embedded in distractor-dense prose — a recall + per-entity
    precision stress test, NOT a source of PII-absent rows (ECHR judgments always name people).
    TAB ships ~12 parallel annotations per doc; ``_tab_mentions`` picks one quality-checked
    annotator deterministically. Yields (GoldRecord, unknown_labels).
    """
    dataset = "mattmdjaga/text-anonymization-benchmark-val-test"
    max_scan = max_scan or n * 3
    got = 0
    for row in _iter_raw(dataset, "default", split, max_scan):
        text = row.get("text") or ""
        if not text:
            continue
        entries = _tab_mentions(row.get("annotations") or {}, row.get("quality_checked"))
        raw_spans, unknown = _spans_from(text, entries, "tab")
        yield build_record(text, raw_spans, source="tab", split=""), unknown
        got += 1
        if got >= n:
            break


def _tab_mentions(annotations, quality_checked=None):
    """Flatten TAB's per-annotator entity_mentions to ``[{label,start,end}]`` using ONE annotator.

    Prefer a quality-checked annotator (those that passed TAB's QC); otherwise the first by sorted
    name. One consistent annotator gives a deterministic gold without having to merge the
    boundary-divergent parallel annotations.
    """
    if not annotations:
        return []
    prefer = [a for a in (quality_checked or []) if a in annotations]
    key = sorted(prefer)[0] if prefer else sorted(annotations)[0]
    out = []
    for m in (annotations.get(key) or {}).get("entity_mentions", []):
        s, e = m.get("start_offset"), m.get("end_offset")
        if s is None or e is None:
            continue
        out.append({"label": m.get("entity_type"), "start": s, "end": e})
    return out


def ingest_kaggle_pii(path, n=None):
    """Kaggle 'PII Data Detection' / PIILO essays from a LOCAL json file (needs Kaggle creds to
    download separately). Expects the competition format: [{"full_text", "tokens", "trailing_whitespace",
    "labels"(BIO)}]. Yields (GoldRecord, unknown_labels). Skipped automatically if the file is absent.
    """
    import os
    if not path or not os.path.exists(path):
        return
    data = json.load(open(path))
    for i, ex in enumerate(data):
        if n and i >= n:
            break
        text = ex.get("full_text") or ""
        tokens = ex.get("tokens") or []
        trailing = ex.get("trailing_whitespace") or [True] * len(tokens)
        bio = ex.get("labels") or []
        raw_spans, unknown = _bio_to_spans(text, tokens, trailing, bio)
        yield build_record(text, raw_spans, source="kaggle-pii", split=""), unknown


def _bio_to_spans(text, tokens, trailing, bio):
    """Convert token-level BIO labels to char spans by walking the reconstructed text."""
    raw_spans, unknown = [], []
    pos, cur = 0, None  # cur = [girp_label, start, end]
    for tok, ws, tag in zip(tokens, trailing, bio):
        idx = text.find(tok, pos)
        if idx < 0:
            idx = pos
        start, end = idx, idx + len(tok)
        pos = end + (1 if ws else 0)
        if tag and tag != "O":
            prefix, _, raw = tag.partition("-")
            if not is_known("kaggle-pii", raw):
                unknown.append(raw)
                girp = None
            else:
                girp = map_label("kaggle-pii", raw)
            if girp is None:
                if cur:
                    raw_spans.append(tuple(cur)); cur = None
                continue
            if prefix == "B" or cur is None or cur[0] != girp:
                if cur:
                    raw_spans.append(tuple(cur))
                cur = [girp, start, end]
            else:
                cur[2] = end
        else:
            if cur:
                raw_spans.append(tuple(cur)); cur = None
    if cur:
        raw_spans.append(tuple(cur))
    raw_spans = [(g, s, e, g) for (g, s, e) in raw_spans]
    return raw_spans, unknown


def ingest_synthetic_au(n, seed=0):
    """Australian-specific gold from the local synthetic generator (OFFLINE). Yields (rec, [])."""
    from synthetic import generate_synthetic_dataset
    for row in generate_synthetic_dataset(n, seed=seed, return_spans=True):
        raw_spans = [(label, start, end) for (label, start, end) in row["spans"]]
        yield build_record(row["text"], raw_spans, source="synthetic-au", split=""), []
