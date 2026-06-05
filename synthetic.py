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
    return f"{r.randint(100, 999)} {r.randint(100, 999)} {r.randint(100, 999)}"


def _medicare(r):
    return f"{r.randint(1000, 9999)} {r.randint(10000, 99999)} {r.randint(0, 9)}"


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


def generate_synthetic_dataset(n: int = 200, seed: int = 0):
    """Return a DataFrame[text, expected] of n synthetic rows across all tiers (shuffled)."""
    import pandas as pd
    r = random.Random(seed)
    rows = []
    for i in range(n):
        text, level = _BUILDERS[i % len(_BUILDERS)](r)
        rows.append({"text": text, "expected": level})
    r.shuffle(rows)
    return pd.DataFrame(rows)
