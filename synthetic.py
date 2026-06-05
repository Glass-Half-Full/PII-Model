"""Synthetic data generator for validating the classifier across all sensitivity tiers.

`generate_synthetic_dataset(n, seed)` returns a pandas DataFrame with:
  * text      - a synthetic sentence
  * expected  - the GIRP level it should be classified as

Coverage is comprehensive: every tier (Public/Private/Confidential/Highly Confidential),
every detectable element type, realistic combinations, and deliberate *false-positive bait*
(generic words like "office"/"team", stray numbers like order/invoice IDs) that must stay Public.
Fully offline; deterministic for a given seed.
"""
from __future__ import annotations

import random

FIRST = ["Sarah", "John", "Mary", "David", "Priya", "Wei", "Ahmed", "Liam", "Olivia",
         "Noah", "Emma", "Raj", "Chen", "Fatima", "Lucas", "Mia", "Grace", "Tom"]
LAST = ["Lee", "Smith", "Citizen", "Connor", "Patel", "Nguyen", "Khan", "O'Brien",
        "Garcia", "Mueller", "Rossi", "Tanaka", "Brown", "Singh", "Walker", "Young"]
STREETS = ["King Street", "Oak Avenue", "Pine Road", "George Street", "Bourke Street",
           "Elizabeth Lane", "High Street", "Park Terrace", "Collins Street"]
CITIES = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Auckland", "Hobart"]
CONDITIONS = ["asthma", "Type 2 diabetes", "depression", "high blood pressure",
              "anxiety", "a heart condition", "chronic migraines"]


def _phone(r):
    return f"+61 4{r.randint(0, 9)}{r.randint(0, 9)} {r.randint(100, 999)} {r.randint(100, 999)}"


def _luhn_check_digit(body: str) -> int:
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def _card(r):
    body = "4" + "".join(str(r.randint(0, 9)) for _ in range(14))   # Visa-style, 15 digits
    num = body + str(_luhn_check_digit(body))                       # append Luhn check digit
    return " ".join(num[i:i + 4] for i in range(0, 16, 4))


def _tfn(r):
    """A checksum-valid Australian Tax File Number (weights 1,4,3,7,5,8,6,9,10 sum % 11 == 0)."""
    w = [1, 4, 3, 7, 5, 8, 6, 9]
    while True:
        d = [r.randint(0, 9) for _ in range(8)]
        c = sum(x * wi for x, wi in zip(d, w)) % 11   # weight 10 ≡ -1 (mod 11) -> check = sum % 11
        if c < 10:
            dd = d + [c]
            return f"{dd[0]}{dd[1]}{dd[2]} {dd[3]}{dd[4]}{dd[5]} {dd[6]}{dd[7]}{dd[8]}"


def _medicare(r):
    """A checksum-valid Australian Medicare number (first 8 digits + check digit + issue)."""
    w = [1, 3, 7, 9, 1, 3, 7, 9]
    d = [r.randint(2, 6)] + [r.randint(0, 9) for _ in range(7)]
    d.append(sum(x * wi for x, wi in zip(d, w)) % 10)
    return f"{''.join(map(str, d[:4]))} {''.join(map(str, d[4:9]))} {r.randint(1, 9)}"


def _passport(r):
    return f"{r.choice('ABCDEFNP')}{r.choice('ABCDEFNP')}{r.randint(1000000, 9999999)}"


def _dob(r):
    return f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/{r.randint(1950, 2005)}"


def _bank(r):
    return f"BSB {r.randint(0, 99):02d}-{r.randint(100, 999)} account {r.randint(10000000, 99999999)}"


def _name(r):
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


# Each builder returns (text, expected_level). Equal weight across tiers.
def _public(r):
    return r.choice([
        "The quarterly report shows strong growth this year.",
        "Our office will be closed for the public holiday.",            # 'office' = FP bait
        "The team meeting is scheduled for Tuesday morning.",           # 'team' = FP bait
        "Thanks for the lovely afternoon at the park!",
        "Please review the attached marketing brochure.",
        f"Order {r.randint(10000, 99999)} has shipped from the warehouse.",   # stray number
        f"Invoice {r.randint(1000, 9999)} was approved by finance.",          # stray number
        "Reminder: submit your timesheet by Friday.",
        "The customer satisfaction survey closes next week.",
    ]), "Public"


def _private(r):
    return r.choice([
        f"You can reach me on {_phone(r)} any time.",
        f"Parcel shipped to {r.randint(1, 200)} {r.choice(STREETS)}, {r.choice(CITIES)}.",
        f"My email is {r.choice(FIRST).lower()}@example.com for enquiries.",
        f"Bank account {_bank(r)}.",
        f"Mailing address: {r.randint(1, 99)} {r.choice(STREETS)}.",
    ]), "Private"


def _confidential(r):
    return r.choice([
        f"Please contact {_name(r)} on {_phone(r)} about the order.",
        f"{_name(r)} lives at {r.randint(1, 200)} {r.choice(STREETS)}, {r.choice(CITIES)}.",
        f"Payment card {_card(r)} was charged today.",
        f"Applicant tax file number {_tfn(r)} is on file.",
        f"Passport number {_passport(r)} was sighted at the branch.",
        f"Medicare number {_medicare(r)} provided at intake.",
        f"{_name(r)} was born on {_dob(r)}.",
        f"Customer {_name(r)} emailed {r.choice(FIRST).lower()}.{r.choice(LAST).lower().replace(chr(39),'')}@example.com today.",
    ]), "Confidential"


def _highly(r):
    return r.choice([
        f"Patient {_name(r)}, phone {_phone(r)}, is being treated for {r.choice(CONDITIONS)}.",
        f"{_name(r)} at {r.randint(1, 200)} {r.choice(STREETS)} was diagnosed with {r.choice(CONDITIONS)}.",
        f"Medical note: {_name(r)} (ph {_phone(r)}) has {r.choice(CONDITIONS)}.",
    ]), "Highly Confidential"


_BUILDERS = [_public, _private, _confidential, _highly]


# ---------------------------------------------------------------------------
# Span-labelled generation (additive). Used to build the Australian-specific gold
# slice for the recursive-improvement loop, where public PII datasets have no AU
# coverage (TFN, Medicare, BSB, +61). Each builder assembles text from labelled
# fragments so every PII value carries exact char offsets. The original
# (text, expected) path above is left byte-for-byte unchanged.
# ---------------------------------------------------------------------------
def _assemble(fragments):
    """fragments: list of (string, label_or_None) -> (text, [(label, start, end), ...])."""
    parts, spans, pos = [], [], 0
    for s, label in fragments:
        if label is not None:
            spans.append((label, pos, pos + len(s)))
        parts.append(s)
        pos += len(s)
    return "".join(parts), spans


def _au_street(r):
    return f"{r.randint(1, 200)} {r.choice(STREETS)}"


def _licence(r):
    """AU-style driver's licence: a letter + digits (validated on digit count, not format)."""
    return f"{r.choice('ABCDEFGH')}{r.randint(100000, 9999999)}"


def _sp_public(r):
    text, spans = _assemble([(r.choice([
        "The quarterly report shows strong growth this year.",
        "Our office will be closed for the public holiday.",
        "The team meeting is scheduled for Tuesday morning.",
        f"Order {r.randint(10000, 99999)} has shipped from the warehouse.",
        f"Invoice {r.randint(1000, 9999)} was approved by finance.",
        "Reminder: submit your timesheet by Friday.",
    ]), None)])
    return text, spans, "Public"


def _sp_private(r):
    k = r.randint(0, 3)
    if k == 0:
        frags = [("You can reach me on ", None), (_phone(r), "phone number"), (" any time.", None)]
    elif k == 1:
        frags = [("Mailing address: ", None), (_au_street(r), "address"),
                 (f", {r.choice(CITIES)}.", None)]
    elif k == 2:
        frags = [("My email is ", None), (f"{r.choice(FIRST).lower()}@example.com", "email address"),
                 (" for enquiries.", None)]
    else:
        frags = [("Bank account number ", None),
                 (f"{r.randint(10000000, 99999999)}", "bank account number"), (".", None)]
    text, spans = _assemble(frags)
    return text, spans, "Private"


def _sp_confidential(r):
    k = r.randint(0, 7)
    if k == 0:
        frags = [("Please contact ", None), (_name(r), "person"), (" on ", None),
                 (_phone(r), "phone number"), (" about the order.", None)]
    elif k == 1:
        frags = [(_name(r), "person"), (" lives at ", None), (_au_street(r), "address"),
                 (f", {r.choice(CITIES)}.", None)]
    elif k == 2:
        frags = [("Payment card ", None), (_card(r), "credit card number"), (" was charged today.", None)]
    elif k == 3:
        frags = [("Applicant tax file number ", None), (_tfn(r), "tax file number"), (" is on file.", None)]
    elif k == 4:
        frags = [("Passport number ", None), (_passport(r), "passport number"),
                 (" was sighted at the branch.", None)]
    elif k == 5:
        frags = [("Medicare number ", None), (_medicare(r), "medicare number"), (" provided at intake.", None)]
    elif k == 6:
        frags = [(_name(r), "person"), (" was born on ", None), (_dob(r), "date of birth"), (".", None)]
    else:
        frags = [("Driver licence ", None), (_licence(r), "driver's licence number"), (" recorded.", None)]
    text, spans = _assemble(frags)
    return text, spans, "Confidential"


def _sp_highly(r):
    k = r.randint(0, 2)
    if k == 0:
        frags = [("Patient ", None), (_name(r), "person"), (", phone ", None), (_phone(r), "phone number"),
                 (", is being treated for ", None), (r.choice(CONDITIONS), "health condition"), (".", None)]
    elif k == 1:
        frags = [(_name(r), "person"), (" at ", None), (_au_street(r), "address"),
                 (" was diagnosed with ", None), (r.choice(CONDITIONS), "health condition"), (".", None)]
    else:
        frags = [("Medical note: ", None), (_name(r), "person"), (", TFN ", None), (_tfn(r), "tax file number"),
                 (", has ", None), (r.choice(CONDITIONS), "health condition"), (".", None)]
    text, spans = _assemble(frags)
    return text, spans, "Highly Confidential"


_SPAN_BUILDERS = [_sp_public, _sp_private, _sp_confidential, _sp_highly]


def generate_synthetic_dataset(n: int = 200, seed: int = 0, return_spans: bool = False):
    """Return n synthetic rows across all tiers (shuffled), deterministic for a given seed.

    Default: a DataFrame[text, expected] (unchanged behaviour). With ``return_spans=True``,
    returns a list of dicts ``{"text", "expected", "spans": [(label, start, end), ...]}`` where
    every PII value carries exact char offsets — the Australian-specific gold slice for the loop.
    """
    r = random.Random(seed)
    if return_spans:
        rows = []
        for i in range(n):
            text, spans, level = _SPAN_BUILDERS[i % len(_SPAN_BUILDERS)](r)
            rows.append({"text": text, "expected": level, "spans": spans})
        r.shuffle(rows)
        return rows
    import pandas as pd
    rows = []
    for i in range(n):
        text, level = _BUILDERS[i % len(_BUILDERS)](r)
        rows.append({"text": text, "expected": level})
    r.shuffle(rows)
    return pd.DataFrame(rows)
