"""Model versioning / tagging ledger for the recursive-improvement loop (offline, no model).

Each accepted iteration is "tagged": its metrics snapshot + full provenance (weights sha, GIRP-rule
and config hashes, threshold, gold version, parent) are appended to ``models.lock`` (machine-readable)
and a human-readable block is prepended to ``CHANGELOG_MODEL.md``. Because config hashes are recorded,
a config-only change (Stage-1 system tuning) is auditable and versioned distinctly from a real
retrained checkpoint (Stage-2), making "the refined model was just threshold changes" impossible to
hide. Semver: PATCH = system-tuning iteration, MINOR = fine-tuned checkpoint, MAJOR = GIRP rule change.
"""
from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import os

import girp

LOCK_PATH = "models.lock"
CHANGELOG_PATH = "CHANGELOG_MODEL.md"
LOCK_SCHEMA = "models.lock-1.0"


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def config_hashes():
    """Hash the editable Stage-1 decision surface so config drift is detectable without weights."""
    import aupii
    return {
        "girp_rules": _sha(inspect.getsource(girp.classify_elements)),
        "validation": _sha(inspect.getsource(girp.is_valid_entity)),
        "regex": _sha(inspect.getsource(girp.regex_elements) + repr(girp._EMAIL_RE.pattern)
                      + repr(girp._CARD_RE.pattern) + repr(girp._INTL_PHONE_RE.pattern)),
        "label_sets": _sha(repr(sorted(girp.GIRP_PII_LABELS)) + repr(sorted(girp.SENSITIVE_LABELS))
                           + repr(sorted(girp.CONFIDENTIAL_ISOLATION_LABELS))),
        "fuzzy_groups": _sha(repr(aupii.GLINER_FUZZY_GROUPS)),
        "presidio_map": _sha(repr(sorted(aupii.PRESIDIO2GIRP.items()))),
    }


def weights_sha256(model_dir=None):
    path = os.path.join(model_dir or girp.default_model_dir(), "model.safetensors")
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()[:16]


def read_lock(lock_path=LOCK_PATH):
    if not os.path.exists(lock_path):
        return {"schema": LOCK_SCHEMA, "current": None, "models": {}}
    with open(lock_path) as f:
        return json.load(f)


def current_metrics(lock_path=LOCK_PATH):
    """Return the (version, metrics) of the currently-tagged model, or (None, None)."""
    lock = read_lock(lock_path)
    cur = lock.get("current")
    if cur and cur in lock.get("models", {}):
        return cur, lock["models"][cur].get("metrics", {})
    return None, None


def bump_version(current, part="patch"):
    if not current:
        return "1.0.0"
    major, minor, patch = (int(x) for x in current.lstrip("v").split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def tag_model(version, stage, headline, gate, parent=None, threshold=0.7, gold_version=None,
              code_commit=None, model_dir=None, note="", iteration=None,
              lock_path=LOCK_PATH, changelog_path=CHANGELOG_PATH):
    """Record an accepted model version in models.lock and CHANGELOG_MODEL.md. Returns the record."""
    lock = read_lock(lock_path)
    record = {
        "version": version,
        "stage": stage,                      # "system-tuning" | "fine-tune" | "baseline"
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "parent": parent or lock.get("current"),
        "iteration": iteration,
        "threshold": threshold,
        "weights_sha256": weights_sha256(model_dir),
        "config_hashes": config_hashes(),
        "gold_version": gold_version,
        "code_commit": code_commit,
        "gate": gate,                        # "PASS" | "FAIL" | detail dict
        "metrics": headline,                 # the evaluate.py headline block
    }
    lock["schema"] = LOCK_SCHEMA
    lock.setdefault("models", {})[version] = record
    lock["current"] = version
    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2)

    _prepend_changelog(changelog_path, record, note)
    return record


def _prepend_changelog(path, record, note):
    h = record["metrics"]
    bal = h.get("balanced_accuracy")
    ci = h.get("balanced_accuracy_ci95") or [None, None]
    head = (f"## v{record['version']} — {record['created'][:10]} "
            f"({record['stage']}, threshold {record['threshold']}"
            + (f", iter {record['iteration']}" if record.get('iteration') is not None else "") + ")")
    body = [
        head, "",
        (f"Balanced GIRP accuracy {bal*100:.1f}%"
         + (f" (95% CI {ci[0]*100:.1f}–{ci[1]*100:.1f})" if ci[0] is not None else "")
         + f"; under {h.get('under',0)*100:.1f}%; over {h.get('over',0)*100:.1f}%; "
         + f"health-under {h.get('health_under',0)*100:.1f}%."),
        f"Gold: {record.get('gold_version')}. Gate: {record.get('gate')}. "
        f"Parent: v{record['parent']}." if record.get("parent") else
        f"Gold: {record.get('gold_version')}. Gate: {record.get('gate')}.",
    ]
    if note:
        body.append(note)
    body.append("")
    block = "\n".join(body) + "\n"
    existing = ""
    if os.path.exists(path):
        with open(path) as f:
            existing = f.read()
    else:
        existing = "# Model changelog\n\nBalanced GIRP accuracy per tagged version (newest first).\n\n"
        header, existing = existing, ""
        block = header + block
    with open(path, "w") as f:
        f.write(block + existing)
