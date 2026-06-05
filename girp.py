"""GIRP personal-information classifier (fully local / offline).

"GIRP" here denotes a generic four-tier information-classification scheme; all rules are
defined in this module — there is no external or proprietary source.

Maps personal-data elements detected in free text to the four-tier classification scheme:

    Public  <  Private  <  Confidential  <  Highly Confidential

No network, Hub, or API calls are made: the model loads from local files only and the
Hub is explicitly disabled. Configured for CPU or NVIDIA CUDA (e.g. an RTX 2050), with
out-of-memory recovery (adaptive batch size + CPU fallback).

Scope
-----
This automates the *personal-information* (Customer) rules of this four-tier scheme — the
element- and combination-based rules that can be derived from entities found in text.
Document-type rules (e.g. board papers, audit reports, product strategy, remuneration)
require document-level context beyond entity detection and are intentionally out of scope.

Precision
---------
Detection is zero-shot, so detected spans are format-validated before they count (a "credit
card" must actually have 13-19 digits, an "address" must contain a number or a street word,
pronouns are never names, etc.). This removes most false positives. Set validate=False to
disable. Raise `threshold` to be stricter still.

Interpretation notes:
  * A full name in isolation defaults to **Private** (identifying personal information).
  * Email address is treated like a contact detail: Private alone, Confidential with a name.
  * A lone date of birth stays Public (GIRP lists DOB only as a combination element).
"""

from __future__ import annotations

import os
import re

LEVELS = ["Public", "Private", "Confidential", "Highly Confidential"]
RANK = {lvl: i for i, lvl in enumerate(LEVELS)}

# --- element groups, expressed as zero-shot entity labels --------------------
NAME_LABELS = {"person"}

COMBINATION_LABELS = {
    "phone number", "email address", "date of birth",
    "address", "birthplace", "mother's maiden name",
}
PRIVATE_ISOLATION_LABELS = {
    "phone number", "email address", "address", "birthplace",
    "mother's maiden name", "bank account number",
}
CONFIDENTIAL_ISOLATION_LABELS = {
    "credit card number", "tax file number", "medicare number",
    "passport number", "driver's licence number",
    "biometric data", "digital signature",
}
SENSITIVE_LABELS = {"medical condition", "health condition", "illness"}

GIRP_PII_LABELS = sorted(
    NAME_LABELS | COMBINATION_LABELS | PRIVATE_ISOLATION_LABELS
    | CONFIDENTIAL_ISOLATION_LABELS | SENSITIVE_LABELS
)

# Detection runs as separate passes over smaller label groups, then merges the results.
# Too many zero-shot labels competing in one pass lowers recall (subtle health terms get
# crowded out), so each pass stays focused. This trades ~Nx inference for much better recall.
DETECTION_GROUPS = [
    GIRP_PII_LABELS,            # full pass — best disambiguation for names / cards / gov IDs
    sorted(SENSITIVE_LABELS),   # focused health pass — reliably catches conditions for the top tier
]

# Pronouns/stopwords the zero-shot model sometimes mis-tags as a "person".
PERSON_STOPWORDS = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours",
    "they", "them", "their", "theirs", "it", "its", "this", "that",
    "someone", "anyone", "everyone", "customer", "client", "applicant", "patient",
    "team", "staff", "manager", "office", "everybody", "nobody",
}

# Common nouns / roles / gender terms the zero-shot model often mis-tags as a person name.
_PERSON_NONNAME = {
    "student", "students", "child", "children", "parent", "parents", "family", "spouse",
    "partner", "guardian", "minor", "adult", "adults", "baby", "kid", "kids", "boy", "girl",
    "male", "female", "man", "woman", "men", "women", "person", "people", "individual",
    "trans male", "trans female", "transgender", "transgender person", "nonbinary", "non-binary",
    "employee", "employer", "member", "members", "user", "users", "senior",
    # NOTE: "citizen" is intentionally NOT here — it is a real surname (the AU placeholder name
    # "John Citizen"); "senior citizen" stays filtered via "senior". (loop iter-001)
    "teacher", "teachers", "facilitator", "investor", "supervisor", "coordinator", "director",
}
# Job-title / role words: if a detected "person" contains one, it's a title, not a name.
_ROLE_WORDS = {
    "coordinator", "supervisor", "manager", "director", "officer", "analyst", "engineer",
    "specialist", "administrator", "assistant", "executive", "consultant", "technician",
    "operator", "representative", "agent", "advisor", "clerk", "intern", "trainee", "program",
    "group", "department", "teacher", "student", "facilitator", "investor", "developer",
    "designer", "planner", "auditor", "lead", "head", "chief", "officer", "associate",
}


def _is_personish(value: str) -> bool:
    """True if a detected 'person' span looks like an actual name, not a noun/role/gender term."""
    s = value.strip().lower()
    if s in PERSON_STOPWORDS or s in _PERSON_NONNAME or value.strip().isdigit():
        return False
    tokens = s.replace("-", " ").split()
    junk = PERSON_STOPWORDS | _PERSON_NONNAME | _ROLE_WORDS
    if any(t in junk for t in tokens):   # generic noun / role / gender token -> not a real name
        return False
    return True

# ---------------------------------------------------------------------------
# Format validation — removes zero-shot false positives before they count.
# ---------------------------------------------------------------------------
_STREET_RE = re.compile(
    r"\b(st|street|rd|road|ave|avenue|ln|lane|dr|drive|ct|court|blvd|boulevard|"
    r"way|pl|place|terrace|tce|hwy|highway|crescent|cres|close|parade|pde|square|sq)\b",
    re.IGNORECASE,
)
_HAS_DIGIT = re.compile(r"\d")


def _ndigits(s: str) -> int:
    return sum(c.isdigit() for c in s)


def _looks_like_phone(value: str) -> bool:
    """True for an international '+' prefixed number with 8-13 digits. Used to rescue phones
    that the model mislabelled as another numeric ID, so the name+phone combination still fires."""
    s = str(value)
    return "+" in s and 8 <= _ndigits(s) <= 13


# High-precision regex backstops for well-formatted structured PII the zero-shot model
# sometimes misses (e.g. valid card numbers it doesn't recognise). Deterministic; Luhn-checked.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){13,19}(?<=\d)")
_INTL_PHONE_RE = re.compile(r"\+\d[\d \-().]{6,}\d")   # international '+' phone, model-independent


def _luhn_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def regex_elements(text: str) -> set:
    """Backstop set of labels found by high-precision regex on the raw text (model-independent)."""
    found = set()
    s = str(text)
    if _EMAIL_RE.search(s):
        found.add("email address")
    for m in _CARD_RE.finditer(s):
        if _luhn_ok(m.group()):
            found.add("credit card number")
            break
    for m in _INTL_PHONE_RE.finditer(s):
        if 8 <= _ndigits(m.group()) <= 15:
            found.add("phone number")
            break
    return found


def is_valid_entity(label: str, value: str) -> bool:
    """True if a detected span plausibly matches its label's format.

    Conservative: it only rejects clear format mismatches, so it removes false positives
    without dropping well-formed real values.
    """
    v = str(value).strip()
    if len(v) < 2:
        return False
    d = _ndigits(v)
    alpha = any(c.isalpha() for c in v)
    if label == "person":
        return _is_personish(v)
    if label == "email address":
        return "@" in v and "." in v.rsplit("@", 1)[-1]
    if label == "phone number":
        # phones: no letters (excludes crypto/hex), no ':' or ',' (IPv6/MAC/coords),
        # 7-14 digits (excludes 15-digit IMEIs and 16-digit account/card numbers), and not an IPv4.
        if alpha or ":" in v or "," in v:
            return False
        if re.match(r"^\s*\d{1,3}(\.\d{1,3}){3}\s*$", v):   # IPv4 dotted quad
            return False
        return 7 <= d <= 14
    if label == "credit card number":
        # 13-19 digits, no letters. (Luhn is applied in the regex backstop for real cards;
        # not here, because many synthetic datasets use non-Luhn card numbers.)
        return (not alpha) and 13 <= d <= 19
    if label == "tax file number":
        return (not alpha) and 8 <= d <= 9
    if label == "medicare number":
        return (not alpha) and 10 <= d <= 11
    if label == "passport number":
        return 6 <= len(v.replace(" ", "")) <= 10 and d >= 1 and alpha
    if label == "driver's licence number":
        return 4 <= d <= 12 and ":" not in v
    if label == "bank account number":
        return 6 <= d <= 18 and len(v) <= 34 and ":" not in v and "." not in v
    if label == "date of birth":
        return 3 <= d <= 8 and ":" not in v
    if label == "address":
        if _STREET_RE.search(v):
            return True
        # otherwise require a multi-word string with a number and real words
        # (rejects GPS coords, IPs, zipcodes, and crypto/hex blobs).
        return (" " in v) and bool(_HAS_DIGIT.search(v)) and bool(re.search(r"[A-Za-z]{3,}", v)) \
            and ":" not in v
    if label in SENSITIVE_LABELS:
        return alpha and d < 7                        # a condition is words, not a number
    # birthplace, mother's maiden name, biometric, signature: accept (>= 2 chars, handled above).
    return True


def found_labels(entities: dict, validate: bool = True) -> set:
    """Set of element labels with at least one *significant* detected value.

    With validate=True, values are format-checked (see is_valid_entity). With validate=False,
    only pronoun/stopword 'person' values are dropped.
    """
    found = set()
    for label, values in entities.items():
        if not values:
            continue
        if validate:
            values = [v for v in values if is_valid_entity(label, v)]
        elif label in NAME_LABELS:
            values = [v for v in values if str(v).strip().lower() not in PERSON_STOPWORDS]
        if values:
            found.add(label)
    # Phone normalisation: rescue a phone that was mis-tagged as another numeric ID
    # (e.g. "ph +61 ..." labelled "bank account number"), preserving the name+phone rule.
    if "phone number" not in found and any(
        _looks_like_phone(v) for vals in entities.values() for v in (vals or [])
    ):
        found.add("phone number")
    return found


# ---------------------------------------------------------------------------
# Classification rules (deterministic).
# ---------------------------------------------------------------------------
def classify_elements(labels) -> str:
    """Apply the rules to a set of detected element labels; return the level."""
    labels = set(labels)
    has_name = bool(labels & NAME_LABELS)
    has_combo = bool(labels & COMBINATION_LABELS)
    has_private_iso = bool(labels & PRIVATE_ISOLATION_LABELS)
    has_conf_iso = bool(labels & CONFIDENTIAL_ISOLATION_LABELS)
    has_sensitive = bool(labels & SENSITIVE_LABELS)

    identifiable_pii = has_name and has_combo
    # Confidential-level customer PII = an identifiable combination (name + element) OR a
    # confidential-in-isolation identifier (credit card / TFN / Medicare / passport / licence).
    confidential_pii = identifiable_pii or has_conf_iso

    # Highly Confidential = sensitive (health) information together with Confidential-level PII.
    # NOTE: this includes health + a card/gov-ID even without a name (conservative, compliance-safe).
    if has_sensitive and confidential_pii:
        return "Highly Confidential"
    if confidential_pii or has_sensitive:
        return "Confidential"
    if has_private_iso or has_name:
        return "Private"
    return "Public"


def explain(labels) -> dict:
    """Return the level, numeric rank, detected elements, and which rule(s) fired."""
    labels = set(labels)
    level = classify_elements(labels)
    reasons = []
    if labels & NAME_LABELS and labels & COMBINATION_LABELS:
        reasons.append("name + combination element -> Confidential")
    if labels & CONFIDENTIAL_ISOLATION_LABELS:
        reasons.append("confidential element in isolation (e.g. card/gov-id) -> Confidential")
    if labels & SENSITIVE_LABELS:
        reasons.append("sensitive (health) information present")
    if level == "Highly Confidential":
        reasons.append("sensitive info + Confidential-level PII (name+combination or card/gov-id) "
                       "-> Highly Confidential")
    if not reasons and (labels & PRIVATE_ISOLATION_LABELS or labels & NAME_LABELS):
        reasons.append("personal element in isolation -> Private")
    if not reasons:
        reasons.append("no personal information detected -> Public")
    return {"level": level, "rank": RANK[level], "elements": sorted(labels), "reasons": reasons}


def detect_and_classify(model, text, threshold: float = 0.7, validate: bool = True) -> dict:
    """Detect elements in `text` (grouped passes), then classify."""
    merged = {}
    for labels in DETECTION_GROUPS:
        res = model.extract_entities(text, labels, threshold=threshold)
        for lbl, vals in res["entities"].items():
            if vals:
                merged.setdefault(lbl, []).extend(vals)
    found = found_labels(merged, validate=validate) | regex_elements(text)
    out = explain(found)
    out["entities"] = {lbl: vals for lbl, vals in merged.items() if lbl in found}
    return out


# ---------------------------------------------------------------------------
# Robust batched inference: OOM-adaptive batch size, CPU fallback, progress.
# ---------------------------------------------------------------------------
def _is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in msg or "alloc" in msg and "memory" in msg


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _log(msg: str):
    print(f"[girp] {msg}", flush=True)


def _make_bar(total: int, enabled: bool, desc: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit="row", leave=True)
    except Exception:
        return _SimpleBar(total, desc)


class _SimpleBar:
    """Lightweight fallback progress bar if tqdm is unavailable."""
    def __init__(self, total, desc):
        self.total, self.desc, self.n, self._last = max(1, total), desc, 0, -10

    def update(self, k):
        self.n += k
        pct = int(self.n * 100 / self.total)
        if pct >= self._last + 10:
            self._last = pct - (pct % 10)
            print(f"{self.desc}: {pct}%  ({self.n}/{self.total})", flush=True)

    def close(self):
        print(f"{self.desc}: done ({self.n}/{self.total})", flush=True)


def _robust_batch_extract(model, texts, labels, threshold, batch_size, progress, desc, **kwargs):
    """batch_extract_entities with adaptive batch size on OOM and CPU fallback as last resort.

    Extra keyword args (e.g. include_spans, include_confidence) are forwarded to the model so
    evaluation can request char offsets / confidence scores without a second code path.
    """
    results = []
    bs = max(1, int(batch_size))
    bar = _make_bar(len(texts), progress, desc)
    i, cpu_fallback = 0, False
    try:
        while i < len(texts):
            chunk = texts[i:i + bs]
            try:
                res = model.batch_extract_entities(chunk, labels, batch_size=bs, threshold=threshold,
                                                   **kwargs)
            except Exception as e:
                if _is_oom(e):
                    _empty_cache()
                    if bs > 1:
                        bs = max(1, bs // 2)
                        _log(f"out-of-memory: retrying with smaller batch_size={bs}")
                        continue
                    if not cpu_fallback:
                        _log("out-of-memory at batch_size=1: moving model to CPU for the remainder")
                        try:
                            model.to("cpu")
                        except Exception:
                            pass
                        cpu_fallback = True
                        continue
                raise
            results.extend(res)
            i += len(chunk)
            if bar:
                bar.update(len(chunk))
    finally:
        if bar:
            bar.close()
    return results


def _detect_grouped(model, texts, threshold, batch_size, progress, desc):
    """Run each DETECTION_GROUPS label set as its own robust batched pass; merge per-row entities."""
    merged = [dict() for _ in texts]
    n = len(DETECTION_GROUPS)
    for gi, labels in enumerate(DETECTION_GROUPS, 1):
        tag = f"{desc} [pass {gi}/{n}]"
        res = _robust_batch_extract(model, texts, labels, threshold, batch_size, progress, tag)
        for i, r in enumerate(res):
            for lbl, vals in r["entities"].items():
                if vals:
                    merged[i].setdefault(lbl, []).extend(vals)
    return merged


def _auto_batch_size(model) -> int:
    """A safe default batch size: small on GPU (esp. <6 GB cards), larger on CPU."""
    dev = str(getattr(model, "device", "cpu"))
    if "cuda" in dev:
        try:
            import torch
            _, total = torch.cuda.mem_get_info()
            return 4 if total / 1e9 < 6 else 16
        except Exception:
            return 4
    return 16


def classify_columns(model, df, columns, threshold: float = 0.7, batch_size: int = None,
                     validate: bool = True, progress: bool = True, max_chars: int = None):
    """Classify one or more text columns of a DataFrame against the GIRP scheme.

    For each column, adds `<col>_girp_level` and `<col>_girp_elements`; also adds an overall
    `girp_level` = the most sensitive level across the given columns for each row.

    Robust by design: NaN/non-string cells are handled, a progress bar shows row-by-row
    progress, and CUDA out-of-memory is recovered automatically (batch size is halved, then
    the model falls back to CPU). batch_size defaults to a safe value for the active device.
    """
    if isinstance(columns, str):
        columns = [columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}. Available: {list(df.columns)}")

    bs = batch_size or _auto_batch_size(model)
    out = df.copy()
    row_rank = [0] * len(out)
    for col in columns:
        texts = out[col].fillna("").astype(str)
        if max_chars:
            texts = texts.str.slice(0, max_chars)
        texts = texts.tolist()
        merged = _detect_grouped(model, texts, threshold, bs, progress, f"GIRP: {col}")
        levels, elements = [], []
        for i, ent in enumerate(merged):
            found = found_labels(ent, validate=validate) | regex_elements(texts[i])
            lvl = classify_elements(found)
            levels.append(lvl)
            elements.append(sorted(found))
            row_rank[i] = max(row_rank[i], RANK[lvl])
        out[f"{col}_girp_level"] = levels
        out[f"{col}_girp_elements"] = elements
    out["girp_level"] = [LEVELS[r] for r in row_rank]
    return out


def classify_dataframe_girp(model, df, text_col, threshold: float = 0.7, batch_size: int = None,
                            validate: bool = True, progress: bool = True):
    """Single-column helper: adds `girp_level` and `girp_elements`."""
    res = classify_columns(model, df, [text_col], threshold=threshold, batch_size=batch_size,
                           validate=validate, progress=progress)
    res = res.rename(columns={f"{text_col}_girp_elements": "girp_elements"})
    res = res.drop(columns=[f"{text_col}_girp_level"])
    return res


# ---------------------------------------------------------------------------
# Local, offline model loading (no Hub / API calls).
# ---------------------------------------------------------------------------
def pick_device() -> str:
    """Fastest available local backend: CUDA (NVIDIA) else CPU."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_model_dir() -> str:
    """The folder holding the local model files (same folder as this module)."""
    return os.path.dirname(os.path.abspath(__file__))


def load_local_model(model_dir: str = None, device: str = "auto"):
    """Load GLiNER2 from LOCAL files only — no Hub/internet/API calls.

    Returns (model, device_str). On CUDA, loads in fp16 (quantize=True) to fit small GPUs
    (e.g. a 4 GB RTX 2050). Raises a clear error if the weights are missing or are a
    Git-LFS pointer stub instead of the real ~800 MB file.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_dir = os.path.abspath(model_dir or default_model_dir())
    cfg = os.path.join(model_dir, "config.json")
    weights = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(cfg):
        raise FileNotFoundError(f"No config.json in {model_dir!r}; point model_dir at the model folder.")
    if not os.path.exists(weights) or os.path.getsize(weights) < 1_000_000:
        raise RuntimeError(
            f"{weights!r} is missing or is a Git-LFS pointer stub (the real file is ~800 MB). "
            "Download the real weights: 'git lfs pull', or fetch model.safetensors from the repo's "
            "raw/LFS URL and place it here.")

    from gliner2 import GLiNER2
    dev = pick_device() if device == "auto" else device
    if dev == "cuda":
        try:
            model = GLiNER2.from_pretrained(model_dir, map_location="cuda", quantize=True)
        except Exception as e:
            _log(f"fp16/cuda load failed ({e}); falling back to fp32 on CUDA")
            try:
                model = GLiNER2.from_pretrained(model_dir, map_location="cuda")
            except Exception as e2:
                _log(f"CUDA load failed ({e2}); falling back to CPU")
                model, dev = GLiNER2.from_pretrained(model_dir), "cpu"
    else:
        model = GLiNER2.from_pretrained(model_dir)
    return model, dev
