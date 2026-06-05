"""Australian-ready hybrid PII system — combined assessment for best overall performance.

Combines two engines, each used where it is strongest:
  * **gliner2** (zero-shot) for fuzzy / contextual elements — names, phone, email, address,
    date of birth, birthplace, mother's maiden name, health — where it has the best recall.
  * **Microsoft Presidio** for structured identifiers validated by checksum/regex — credit cards
    (Luhn), IBAN & bank accounts, government IDs — plus **Australian** recognizers (Tax File Number,
    Medicare, ABN, ACN) and a custom **BSB** recognizer — where it has the best precision.

Both feed the GIRP rules (`girp.classify_elements`) to assign Public / Private / Confidential /
Highly Confidential. Runs fully locally/offline (gliner2 weights + spaCy model + regex; no API calls).

Why hybrid: a bigger model gave ~0 gain (see PRODUCTION.md). The real win is letting the zero-shot
model do recall and a checksum engine do precision — which kills the account->card / numeric-ID
false positives that drove over-classification.
"""
from __future__ import annotations

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
GLINER_FUZZY_GROUPS = [
    ["person", "phone number", "date of birth", "address", "birthplace", "mother's maiden name",
     "passport number", "driver's licence number"],   # zero-shot handles AU formats Presidio's US recognizers miss
    ["medical condition", "health condition", "illness"],
]
GLINER_FUZZY_LABELS = [l for g in GLINER_FUZZY_GROUPS for l in g]

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


def build_analyzer():
    """Build a Presidio analyzer with the default + Australian recognizers + a custom BSB recognizer."""
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
    from presidio_analyzer.predefined_recognizers import (
        AuTfnRecognizer, AuMedicareRecognizer, AuAbnRecognizer, AuAcnRecognizer)
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
    return AnalyzerEngine(registry=reg)


def presidio_elements(analyzer, text, score_threshold=0.4):
    """GIRP element labels found by Presidio's structured/checksum recognizers."""
    found = set()
    for x in analyzer.analyze(text, language="en", score_threshold=score_threshold):
        g = PRESIDIO2GIRP.get(x.entity_type)
        if g:
            found.add(g)
    return found


def detect_and_classify_hybrid(model, analyzer, text, threshold: float = 0.7, validate: bool = True) -> dict:
    """Detect with gliner2 (fuzzy) + Presidio (structured) and classify per GIRP."""
    merged = {}
    for labs in GLINER_FUZZY_GROUPS:
        for l, v in model.extract_entities(text, labs, threshold=threshold)["entities"].items():
            if v:
                merged.setdefault(l, []).extend(v)
    fuzzy = found_labels(merged, validate=validate)
    structured = presidio_elements(analyzer, text)
    found = fuzzy | structured | _regex_phone(text)
    out = explain(found)
    out["entities"] = {l: v for l, v in merged.items() if l in found}
    out["structured"] = sorted(structured)
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
    for col in columns:
        texts = out[col].fillna("").astype(str).tolist()
        merged = [dict() for _ in texts]
        for gi, labs in enumerate(GLINER_FUZZY_GROUPS, 1):
            gres = _robust_batch_extract(model, texts, labs, threshold, bs, progress,
                                         f"hybrid {col} [pass {gi}/{len(GLINER_FUZZY_GROUPS)}]")
            for i, r in enumerate(gres):
                for l, v in r["entities"].items():
                    if v:
                        merged[i].setdefault(l, []).extend(v)
        levels, elements = [], []
        for i, t in enumerate(texts):
            found = found_labels(merged[i], validate=validate) | presidio_elements(analyzer, t) | _regex_phone(t)
            lvl = classify_elements(found)
            levels.append(lvl)
            elements.append(sorted(found))
            row_rank[i] = max(row_rank[i], RANK[lvl])
        out[f"{col}_girp_level"] = levels
        out[f"{col}_girp_elements"] = elements
    out["girp_level"] = [LEVELS[r] for r in row_rank]
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


def load_gliner_pii(model_name: str = "knowledgator/gliner-pii-small-v1.0", device: str = None):
    """Load a PII-specialized original-GLiNER model, wrapped for the hybrid interface."""
    from gliner import GLiNER
    m = GLiNER.from_pretrained(model_name)
    if device:
        m = m.to(device)
    return GlinerV1Adapter(m)
