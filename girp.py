"""GIRP personal-information classifier (fully local / offline).

"GIRP" here denotes a generic four-tier information-classification scheme; all rules are
defined in this module — there is no external or proprietary source.

Maps personal-data elements detected in free text to the four-tier classification scheme:

    Public  <  Private  <  Confidential  <  Highly Confidential

No network, Hub, or API calls are made: the model loads from local files only and the
Hub is explicitly disabled. Configured for CPU or NVIDIA CUDA (e.g. an RTX 2050).

Scope
-----
This automates the *personal-information* (Customer) rules of this four-tier scheme — the
element- and combination-based rules that can be derived from entities found in text.
Document-type rules (e.g. board papers, audit reports, product strategy, remuneration)
require document-level context beyond entity detection and are intentionally out of scope.

Interpretation notes (where GIRP is explicit vs. where we apply a documented default):
  * A full name in *isolation* is not listed in GIRP's isolation tables; we default it to
    **Private** (it is identifying personal information). Adjust if your policy differs.
  * Email address is not in GIRP's Customer lists; we treat it like other contact details
    (phone): Private in isolation, and a combination element with a name -> Confidential.
  * Date of birth is listed only as a *combination* element (name + DOB -> Confidential),
    not as Private-in-isolation, so a lone DOB stays Public — faithful to the rules defined here.
    Move it into PRIVATE_ISOLATION_LABELS below if your policy requires stricter handling.
"""

from __future__ import annotations

import os

LEVELS = ["Public", "Private", "Confidential", "Highly Confidential"]
RANK = {lvl: i for i, lvl in enumerate(LEVELS)}

# --- GIRP element groups, expressed as zero-shot entity labels ----------------
# Identifying information: a person's full name (First + Last).
NAME_LABELS = {"person"}

# Combination elements: name + ANY of these => Confidential (GIRP "Confidential", Customer).
COMBINATION_LABELS = {
    "phone number", "email address", "date of birth",
    "address", "birthplace", "mother's maiden name",
}

# Private when seen in isolation (GIRP "Private", Customer).
PRIVATE_ISOLATION_LABELS = {
    "phone number", "email address", "address", "birthplace",
    "mother's maiden name", "bank account number",
}

# Confidential when seen in isolation, no identity required (GIRP "Confidential", Customer).
CONFIDENTIAL_ISOLATION_LABELS = {
    "credit card number", "tax file number", "medicare number",
    "passport number", "driver's licence number",
    "biometric data", "digital signature",
}

# Sensitive information -> Highly Confidential when combined with Confidential customer PII.
# ("medical condition" is the label the model detects most reliably for health information.)
SENSITIVE_LABELS = {"medical condition"}

# Every label to ask the model for in one pass.
GIRP_PII_LABELS = sorted(
    NAME_LABELS | COMBINATION_LABELS | PRIVATE_ISOLATION_LABELS
    | CONFIDENTIAL_ISOLATION_LABELS | SENSITIVE_LABELS
)

# Pronouns/stopwords that the zero-shot model sometimes mis-tags as a "person".
# Dropping these prevents spurious name detections (e.g. "call me on ...") from inflating
# the classification. They are never genuine customer identifiers, so this only ever makes
# the result MORE accurate — it cannot cause under-classification of a real name.
PERSON_STOPWORDS = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours",
    "they", "them", "their", "theirs", "it", "its", "this", "that",
    "someone", "anyone", "everyone", "customer", "client", "applicant", "patient",
}


def found_labels(entities: dict) -> set:
    """Set of element labels that have at least one *significant* detected value.

    Filters out pronoun/stopword false-positives for the 'person' label.
    """
    found = set()
    for label, values in entities.items():
        if not values:
            continue
        if label in NAME_LABELS:
            values = [v for v in values if str(v).strip().lower() not in PERSON_STOPWORDS]
            if not values:
                continue
        found.add(label)
    return found


def classify_elements(labels) -> str:
    """Apply the GIRP rules to a set of detected element labels; return the GIRP level."""
    labels = set(labels)
    has_name = bool(labels & NAME_LABELS)
    has_combo = bool(labels & COMBINATION_LABELS)
    has_private_iso = bool(labels & PRIVATE_ISOLATION_LABELS)
    has_conf_iso = bool(labels & CONFIDENTIAL_ISOLATION_LABELS)
    has_sensitive = bool(labels & SENSITIVE_LABELS)

    identifiable_pii = has_name and has_combo            # GIRP "Confidential" combination

    # Highly Confidential = sensitive info + Confidential customer PII.
    if has_sensitive and identifiable_pii:
        return "Highly Confidential"
    # Confidential = identifiable PII combination, or a confidential/sensitive element alone.
    if identifiable_pii or has_conf_iso or has_sensitive:
        return "Confidential"
    # Private = a personal element in isolation (name alone defaults to Private).
    if has_private_iso or has_name:
        return "Private"
    return "Public"


def explain(labels) -> dict:
    """Return the level, numeric rank, detected elements, and which GIRP rule(s) fired."""
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
        reasons.append("sensitive info + Confidential customer PII -> Highly Confidential")
    if not reasons and (labels & PRIVATE_ISOLATION_LABELS or labels & NAME_LABELS):
        reasons.append("personal element in isolation -> Private")
    if not reasons:
        reasons.append("no personal information detected -> Public")
    return {"level": level, "rank": RANK[level], "elements": sorted(labels), "reasons": reasons}


def detect_and_classify(model, text, threshold: float = 0.5) -> dict:
    """Detect GIRP elements in `text` with a GLiNER2 model, then classify per GIRP."""
    res = model.extract_entities(text, GIRP_PII_LABELS, threshold=threshold)
    found = found_labels(res["entities"])
    out = explain(found)
    out["entities"] = {lbl: vals for lbl, vals in res["entities"].items() if lbl in found}
    return out


def classify_columns(model, df, columns, threshold: float = 0.5, batch_size: int = 8,
                     include_elements: bool = True):
    """Classify one or more text columns of a DataFrame against GIRP, in batched passes.

    For each column in `columns`, adds:
        <col>_girp_level     - GIRP level for that cell
        <col>_girp_elements  - detected elements (if include_elements)
    and adds an overall row-level column:
        girp_level           - the highest (most sensitive) level across the given columns

    This is the main entry point: pick your DataFrame, list the columns to scan, run.
    """
    if isinstance(columns, str):
        columns = [columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}. Available: {list(df.columns)}")

    out = df.copy()
    row_rank = [0] * len(out)
    for col in columns:
        texts = out[col].fillna("").astype(str).tolist()
        results = model.batch_extract_entities(texts, GIRP_PII_LABELS,
                                               batch_size=batch_size, threshold=threshold)
        levels, elements = [], []
        for i, r in enumerate(results):
            found = found_labels(r["entities"])
            lvl = classify_elements(found)
            levels.append(lvl)
            elements.append(sorted(found))
            row_rank[i] = max(row_rank[i], RANK[lvl])
        out[f"{col}_girp_level"] = levels
        if include_elements:
            out[f"{col}_girp_elements"] = elements

    out["girp_level"] = [LEVELS[r] for r in row_rank]
    return out


# Backwards-compatible single-column helper.
def classify_dataframe_girp(model, df, text_col, threshold: float = 0.5, batch_size: int = 8):
    """Add `girp_level` and `girp_elements` columns for a single text column."""
    res = classify_columns(model, df, [text_col], threshold=threshold,
                           batch_size=batch_size, include_elements=True)
    res = res.rename(columns={f"{text_col}_girp_elements": "girp_elements"})
    res = res.drop(columns=[f"{text_col}_girp_level"])
    return res


# --- Local, offline model loading (no Hub / API calls) -----------------------
def pick_device() -> str:
    """Fastest available local backend: CUDA (NVIDIA, e.g. RTX 2050) else CPU."""
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
    # Hard-disable any Hub/telemetry network access before importing the libraries.
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
            "Make sure the actual model.safetensors weights are present in this folder "
            "(use 'git lfs pull' or copy the file in).")

    from gliner2 import GLiNER2  # imported only after offline env vars are set
    dev = pick_device() if device == "auto" else device
    if dev == "cuda":
        try:
            model = GLiNER2.from_pretrained(model_dir, map_location="cuda", quantize=True)
        except Exception as e:  # fall back to fp32 if fp16 isn't supported here
            print("fp16 load failed, falling back to fp32:", e)
            model = GLiNER2.from_pretrained(model_dir, map_location="cuda")
    else:
        model = GLiNER2.from_pretrained(model_dir)  # loads on CPU
    return model, dev
