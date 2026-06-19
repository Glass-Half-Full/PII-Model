"""Australian-ready hybrid PII system — combined assessment for best overall performance.

Combines two engines, each used where it is strongest:
  * **gliner2** (zero-shot) for fuzzy / contextual elements — names, phone, email, address,
    date of birth, birthplace, mother's maiden name, health — where it has the best recall.
  * **Microsoft Presidio** for structured identifiers validated by checksum/regex — credit cards
    (Luhn), IBAN & bank accounts, government IDs — plus **Australian** recognizers (Tax File Number,
    Medicare, ABN, ACN) and a custom **BSB** recognizer — where it has the best precision.

Both feed the GIRP rules (`girp.classify_elements`) to assign Public / Private / Confidential /
Highly Confidential. Runs fully locally/offline (gliner2 weights + spaCy model + regex; no API calls).

Why hybrid: the zero-shot model gives recall for contextual elements, while checksum and regex
recognizers give precision for structured identifiers.
"""
from __future__ import annotations

import os
import re

from girp import (classify_elements, explain, found_labels, RANK, LEVELS,
                  load_local_model, _robust_batch_extract)

# Phone-only regex backstop: rescues well-formed international (+NN ...) numbers the model
# misses (e.g. "ph +61 4.. .. ..") so the name+phone combination still fires. Higher precision
# than the full regex backstop (no card/email), which over-fired on PII-dense text.
_PHONE_RX = re.compile(r"\+\d[\d \-().]{6,}\d")


def _regex_phone(text: str) -> set:
    for m in _PHONE_RX.finditer(str(text)):
        if 8 <= sum(c.isdigit() for c in m.group()) <= 15:
            return {"phone number"}
    return set()

# The ML model detects fuzzy/contextual elements in two focused passes (a separate health pass
# keeps recall high for the top tier). Email is handled by Presidio (100% precision), not here.
# Each group is (labels, threshold_override). The health group runs at a LOWER threshold because
# it drives the top tier (Highly Confidential) — we want high recall there even at some extra
# health false positives (which only over-classify in the safe direction).
GLINER_FUZZY_GROUPS = [
    (["person", "phone number", "date of birth", "address", "birthplace", "mother's maiden name",
      "passport number", "driver's licence number"], None),   # zero-shot handles AU formats Presidio misses
    (["medical condition", "health condition", "illness"], 0.6),   # raised 0.4->0.6 in loop iter-002
]
GLINER_FUZZY_LABELS = [l for labs, _ in GLINER_FUZZY_GROUPS for l in labs]

# Lever A (per-entity precision): raise the confidence bar on individual high-false-positive labels
# without disturbing the rest. EMPTY by default = no behavior change. Calibrate per label from the
# a labeled validation set, e.g.
#   {"date of birth": 0.9, "driver's licence number": 0.9, "passport number": 0.85}
# evaluate.derive and the production hybrid twins BOTH consult this map, so eval == production.
PER_LABEL_THRESHOLDS: dict = {}
# The health pass ran at 0.4 for maximum recall, but that hallucinated health conditions in long
# finance/EDIFACT text (15 over-classified rows). At 0.6, balanced GIRP accuracy rises while Highly
# Confidential recall stays 100% and health under-classification stays 0% on gold v1 — true health
# conditions are detected with high confidence. Kept below 0.8 (where Highly recall starts to drop).

# Low-precision zero-shot labels SUPPRESSED from the final element set (loop iter-001). "birthplace"
# and "mother's maiden name" fire on ordinary place names in real text ("Albuquerque museum",
# "Saint-Robert-Bellarmin"), driving Public over-classification, with ~0 precision and no gold
# support. They are kept in the EXTRACTION label set above (removing them perturbs zero-shot person
# detection — measured -1.7pp balanced accuracy in iter-001) but dropped AFTER extraction, so the
# decision is unperturbed. They remain in the GIRP rule engine for any future dedicated recognizer.
SUPPRESSED_FUZZY_LABELS = frozenset({"birthplace", "mother's maiden name"})

# Presidio entity -> GIRP element. Structured / high-precision items (checksum or regex validated).
PRESIDIO2GIRP = {
    "EMAIL_ADDRESS": "email address",
    "CREDIT_CARD": "credit card number",
    "IBAN_CODE": "bank account number", "US_BANK_NUMBER": "bank account number", "AU_BSB": "bank account number",
    "AU_TFN": "tax file number", "US_SSN": "tax file number", "US_ITIN": "tax file number",
    "AU_MEDICARE": "medicare number",
    "US_PASSPORT": "passport number",
    "US_DRIVER_LICENSE": "driver's licence number",
}
# Business identifiers Presidio also finds (not customer-personal under GIRP; reported, not scored).
PRESIDIO_BUSINESS = {"AU_ABN": "abn", "AU_ACN": "acn"}

# Human-review band: rows at these levels — plus any row where the two engines disagree on the level —
# are flagged `needs_review` so a human checks the highest-stakes / most-uncertain cases. This is the
# production safety net for the residual zero-shot misses (drive the automated rate down via fine-tuning).
REVIEW_LEVELS = {"Highly Confidential"}


def build_analyzer(spacy_model: str = "en_core_web_sm"):
    """Build a Presidio analyzer (default + Australian recognizers + custom BSB) on a lightweight
    spaCy model. The hybrid uses Presidio's pattern/checksum recognizers (regex/Luhn/context), not
    its spaCy NER (gliner2 does NER), so `en_core_web_sm` (~12 MB) is enough and keeps the footprint
    small for offline / RTX 2050 use. Falls back to Presidio's default model if the named one is absent.
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import (
        AuTfnRecognizer, AuMedicareRecognizer, AuAbnRecognizer, AuAcnRecognizer)
    nlp_engine = None
    try:
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }).create_engine()
    except Exception as e:
        print(f"[aupii] spaCy model {spacy_model!r} unavailable ({e}); using Presidio default.")
    reg = RecognizerRegistry()
    reg.load_predefined_recognizers()
    for rec in (AuTfnRecognizer(), AuMedicareRecognizer(), AuAbnRecognizer(), AuAcnRecognizer()):
        reg.add_recognizer(rec)
    # Australian BSB (bank-state-branch): 6 digits, usually NNN-NNN; needs bank context to count.
    bsb = PatternRecognizer(
        supported_entity="AU_BSB",
        patterns=[Pattern(name="bsb", regex=r"\b\d{3}-\d{3}\b", score=0.3)],
        context=["bsb", "branch", "bank", "account"],
    )
    reg.add_recognizer(bsb)
    return AnalyzerEngine(registry=reg, nlp_engine=nlp_engine) if nlp_engine else AnalyzerEngine(registry=reg)


def presidio_elements(analyzer, text, score_threshold=0.4):
    """GIRP element labels found by Presidio's structured/checksum recognizers."""
    found = set()
    for x in analyzer.analyze(text, language="en", score_threshold=score_threshold):
        g = PRESIDIO2GIRP.get(x.entity_type)
        if g:
            found.add(g)
    return found


def presidio_spans(analyzer, text, score_threshold=0.4):
    """Like presidio_elements but keeps char offsets: list of (girp_label, start, end, score).

    Used by the evaluator for span-level metrics; production uses presidio_elements (set) for speed.
    """
    out = []
    for x in analyzer.analyze(text, language="en", score_threshold=score_threshold):
        g = PRESIDIO2GIRP.get(x.entity_type)
        if g:
            out.append((g, x.start, x.end, float(x.score)))
    return out


def _group_floor(labs, default_thr):
    """Lowest threshold relevant to a group = min(default, any per-label overrides in the group),
    so a per-label override below the group default still surfaces its candidates for filtering."""
    return min([default_thr] + [PER_LABEL_THRESHOLDS[l] for l in labs if l in PER_LABEL_THRESHOLDS])


def _filtered_confident(text, entities, default_thr):
    """Keep entity strings whose confidence clears PER_LABEL_THRESHOLDS[label] (else default_thr).
    Mirrors evaluate.derive so the hybrid twins and the evaluator make the identical decision.
    ``entities`` is gliner2's include_spans/include_confidence output: {label: [{start,end,confidence}]}."""
    out = {}
    for l, vals in entities.items():
        for v in (vals or []):
            if isinstance(v, dict) and v.get("start") is not None \
                    and v.get("confidence", 1.0) >= PER_LABEL_THRESHOLDS.get(l, default_thr):
                out.setdefault(l, []).append(text[v["start"]:v["end"]])
    return out


def detect_and_classify_hybrid(model, analyzer, text, threshold: float = 0.7, validate: bool = True) -> dict:
    """Detect with gliner2 (fuzzy) + Presidio (structured) and classify per GIRP."""
    merged = {}
    for labs, thr_override in GLINER_FUZZY_GROUPS:
        thr = thr_override if thr_override is not None else threshold
        if PER_LABEL_THRESHOLDS:        # Lever A: per-label confidence filtering (mirrors derive)
            res = _robust_batch_extract(model, [text], labs, _group_floor(labs, thr), 1, False, "",
                                        include_spans=True, include_confidence=True)[0]
            for l, vals in _filtered_confident(text, res["entities"], thr).items():
                merged.setdefault(l, []).extend(vals)
        else:
            for l, v in model.extract_entities(text, labs, threshold=thr)["entities"].items():
                if v:
                    merged.setdefault(l, []).extend(v)
    fuzzy = (found_labels(merged, validate=validate) | _regex_phone(text)) - SUPPRESSED_FUZZY_LABELS
    structured = presidio_elements(analyzer, text)                        # checksum/regex side
    found = fuzzy | structured
    out = explain(found)
    out["entities"] = {l: v for l, v in merged.items() if l in found}
    out["structured"] = sorted(structured)
    out["needs_review"] = bool(out["level"] in REVIEW_LEVELS
                               or classify_elements(fuzzy) != classify_elements(structured))
    return out


def classify_columns_hybrid(model, analyzer, df, columns, threshold: float = 0.7,
                            validate: bool = True, progress: bool = True, batch_size: int = None):
    """Hybrid classification of DataFrame columns: gliner2 (batched) + Presidio (per row), merged.

    Adds `<col>_girp_level` / `<col>_girp_elements` per column and an overall `girp_level` per row.
    """
    if isinstance(columns, str):
        columns = [columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}. Available: {list(df.columns)}")
    from girp import _auto_batch_size
    bs = batch_size or _auto_batch_size(model)
    out = df.copy()
    row_rank = [0] * len(out)
    row_review = [False] * len(out)
    for col in columns:
        texts = out[col].fillna("").astype(str).tolist()
        merged = [dict() for _ in texts]
        for gi, (labs, thr_override) in enumerate(GLINER_FUZZY_GROUPS, 1):
            thr = thr_override if thr_override is not None else threshold
            desc = f"hybrid {col} [pass {gi}/{len(GLINER_FUZZY_GROUPS)}]"
            if PER_LABEL_THRESHOLDS:    # Lever A: per-label confidence filtering (mirrors derive)
                gres = _robust_batch_extract(model, texts, labs, _group_floor(labs, thr), bs,
                                             progress, desc, include_spans=True, include_confidence=True)
                for i, r in enumerate(gres):
                    for l, vals in _filtered_confident(texts[i], r["entities"], thr).items():
                        merged[i].setdefault(l, []).extend(vals)
            else:
                gres = _robust_batch_extract(model, texts, labs, thr, bs, progress, desc)
                for i, r in enumerate(gres):
                    for l, v in r["entities"].items():
                        if v:
                            merged[i].setdefault(l, []).extend(v)
        levels, elements, review = [], [], []
        for i, t in enumerate(texts):
            g = (found_labels(merged[i], validate=validate) | _regex_phone(t)) - SUPPRESSED_FUZZY_LABELS
            p = presidio_elements(analyzer, t)                                # checksum/regex side
            found = g | p
            lvl = classify_elements(found)
            levels.append(lvl)
            elements.append(sorted(found))
            needs = lvl in REVIEW_LEVELS or classify_elements(g) != classify_elements(p)
            review.append(needs)
            row_rank[i] = max(row_rank[i], RANK[lvl])
            row_review[i] = row_review[i] or needs
        out[f"{col}_girp_level"] = levels
        out[f"{col}_girp_elements"] = elements
        out[f"{col}_needs_review"] = review
    out["girp_level"] = [LEVELS[r] for r in row_rank]
    out["needs_review"] = row_review
    return out


def load_hybrid(model_dir: str = None):
    """Load the full Australian-ready hybrid: (gliner2 model, Presidio analyzer). Local/offline."""
    model, dev = load_local_model(model_dir)
    analyzer = build_analyzer()
    return model, analyzer, dev


class GlinerV1Adapter:
    """Adapt an original-GLiNER model (the `gliner` library, e.g. knowledgator/gliner-pii-*)
    to the gliner2 `extract_entities` / `batch_extract_entities` interface used by the hybrid.

    The small PII-specialized GLiNER models (~330 MB) are a great fit for a 4 GB RTX 2050 / CPU.
    """

    def __init__(self, model):
        self.model = model
        self.device = str(getattr(model, "device", "cpu"))

    def _to_dict(self, ents, labels):
        d = {l: [] for l in labels}
        for e in ents:
            d.setdefault(e["label"], []).append(e["text"])
        return {"entities": d}

    def extract_entities(self, text, labels, threshold=0.5, **kw):
        return self._to_dict(self.model.predict_entities(text, labels, threshold=threshold), labels)

    def batch_extract_entities(self, texts, labels, batch_size=8, threshold=0.5, **kw):
        try:
            res = self.model.batch_predict_entities(texts, labels, threshold=threshold)
        except Exception:
            res = [self.model.predict_entities(t, labels, threshold=threshold) for t in texts]
        return [self._to_dict(e, labels) for e in res]

    def to(self, device):
        self.model = self.model.to(device)
        self.device = str(device)
        return self


def load_gliner_pii(model: str = "knowledgator/gliner-pii-small-v1.0", device: str = None,
                    revision: str = None, offline: bool = True):
    """Load a PII-specialized original-GLiNER model (wrapped for the hybrid) — LOCAL-ONLY by default.

    `model` may be a local directory OR a Hugging Face id that is already cached. With offline=True
    (default) no network is used: the Hub is disabled via env vars. To fetch it the first time, run
    once on a networked machine (`GLiNER.from_pretrained("knowledgator/gliner-pii-small-v1.0")`) or
    vendor the model directory and pass its path. Pin `revision=` for reproducibility.
    """
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from gliner import GLiNER
    rev = {"revision": revision} if revision else {}
    try:
        try:
            m = GLiNER.from_pretrained(model, local_files_only=True, **rev)
        except TypeError:          # older gliner without the kwarg; offline env still enforces local-only
            m = GLiNER.from_pretrained(model, **rev)
    except Exception as e:
        raise RuntimeError(
            f"Could not load {model!r} from local files ({type(e).__name__}: {e}). "
            "Provide a local model directory, or pre-download it once on a networked machine "
            f"(GLiNER.from_pretrained({model!r})) so it is cached; for air-gapped use, vendor the "
            "directory and pass its path. Set offline=False only to allow a one-time Hub fetch."
        ) from e
    if device:
        m = m.to(device)
    return GlinerV1Adapter(m)
