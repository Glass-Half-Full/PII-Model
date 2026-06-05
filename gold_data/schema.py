"""Common gold-record schema for the recursive-improvement gold set (offline, no model).

One JSONL record per text. Char offsets are mandatory so the same file serves BOTH
presence-level and span-level evaluation AND is directly consumable by train_lora.py,
which expects ``{"text", "spans": [{"label", "start", "end"}, ...]}`` (train_lora.py:24).
So gold doubles as training data and closes the recursive loop.

The gold GIRP ``gold_level`` is ALWAYS derivable from the spans via
``girp.classify_elements`` — it is re-derived in ``build_record`` and cross-checked in
``validate``. Changing the GIRP rules therefore re-derives gold tiers without re-downloading.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from girp import classify_elements, GIRP_PII_LABELS

SCHEMA_VERSION = "gold-1.0"
_VOCAB = set(GIRP_PII_LABELS)


@dataclass(frozen=True)
class GoldSpan:
    label: str            # GIRP element label (girp vocabulary), already mapped
    start: int            # char offset into text, inclusive
    end: int              # char offset, exclusive
    raw_label: str = ""   # original source label, kept for audit
    text: str = ""        # convenience copy of text[start:end]


@dataclass(frozen=True)
class GoldRecord:
    id: str
    text: str
    spans: tuple          # tuple[GoldSpan, ...]
    gold_elements: tuple  # sorted unique element labels present in spans
    gold_level: str       # girp.classify_elements(gold_elements) — DERIVED, never hand-set
    source: str           # "ai4privacy" | "kaggle-pii" | "gretel-finance" | "synthetic-au" | ...
    split: str            # "train" | "dev" | "test"
    lang: str = "en"
    schema_version: str = SCHEMA_VERSION


def make_id(source: str, text: str) -> str:
    """Stable per-(source,text) id so re-ingesting the same row is idempotent."""
    return hashlib.sha1(f"{source}|{text}".encode("utf-8")).hexdigest()[:16]


def derive_elements(spans) -> tuple:
    """Sorted unique element labels present in the spans (the presence set)."""
    return tuple(sorted({s.label for s in spans}))


def build_record(text: str, raw_spans, source: str, split: str, lang: str = "en") -> GoldRecord:
    """Build a GoldRecord from text + raw spans.

    ``raw_spans`` is an iterable of ``(label, start, end)`` or ``(label, start, end, raw_label)``;
    ``label`` must already be a GIRP element label. ``gold_elements`` and ``gold_level`` are derived.
    """
    spans = []
    for sp in raw_spans:
        if len(sp) == 4:
            label, start, end, raw_label = sp
        else:
            label, start, end = sp
            raw_label = label
        spans.append(GoldSpan(label, int(start), int(end), raw_label, text[int(start):int(end)]))
    spans = tuple(spans)
    elements = derive_elements(spans)
    level = classify_elements(elements)
    return GoldRecord(
        id=make_id(source, text), text=text, spans=spans,
        gold_elements=elements, gold_level=level,
        source=source, split=split, lang=lang,
    )


def to_jsonl(rec: GoldRecord) -> str:
    """Serialize one record to a single JSON line."""
    d = asdict(rec)
    d["spans"] = [asdict(s) if not isinstance(s, dict) else s for s in rec.spans]
    d["gold_elements"] = list(rec.gold_elements)
    return json.dumps(d, ensure_ascii=False)


def from_jsonl(line: str) -> GoldRecord:
    """Parse one JSON line back into a GoldRecord."""
    d = json.loads(line)
    spans = tuple(
        GoldSpan(s["label"], int(s["start"]), int(s["end"]), s.get("raw_label", ""), s.get("text", ""))
        for s in d.get("spans", [])
    )
    return GoldRecord(
        id=d["id"], text=d["text"], spans=spans,
        gold_elements=tuple(d.get("gold_elements", [])),
        gold_level=d["gold_level"], source=d.get("source", ""), split=d.get("split", ""),
        lang=d.get("lang", "en"), schema_version=d.get("schema_version", SCHEMA_VERSION),
    )


def validate(rec: GoldRecord) -> list:
    """Return a list of human-readable problems; empty list means the record is valid."""
    problems = []
    n = len(rec.text)
    for i, s in enumerate(rec.spans):
        if not (0 <= s.start < s.end <= n):
            problems.append(f"span[{i}] offset out of bounds: ({s.start},{s.end}) for text len {n}")
            continue
        if s.text and s.text != rec.text[s.start:s.end]:
            problems.append(f"span[{i}] text {s.text!r} != slice {rec.text[s.start:s.end]!r}")
        if s.label not in _VOCAB:
            problems.append(f"span[{i}] label {s.label!r} not in GIRP vocabulary")
    expected_elements = derive_elements(rec.spans)
    if tuple(rec.gold_elements) != expected_elements:
        problems.append(f"gold_elements {rec.gold_elements} != derived {expected_elements}")
    expected_level = classify_elements(expected_elements)
    if rec.gold_level != expected_level:
        problems.append(f"gold_level {rec.gold_level!r} != classify_elements -> {expected_level!r}")
    return problems
