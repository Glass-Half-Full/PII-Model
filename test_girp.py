"""Deterministic tests that the GIRP rule engine matches the four-tier classification rules.

Run directly (no pytest needed):   python test_girp.py
Or under pytest:                    pytest test_girp.py
"""
from girp import classify_elements, explain, found_labels, LEVELS, RANK

# (detected element labels, expected GIRP level, which GIRP rule this checks)
CASES = [
    # --- Public ---
    (set(),                                  "Public",  "no personal information"),
    ({"date of birth"},                      "Public",  "DOB is combination-only in GIRP, not Private-in-isolation"),

    # --- Private (element in isolation) ---
    ({"person"},                             "Private", "name in isolation (documented default)"),
    ({"phone number"},                       "Private", "phone in isolation"),
    ({"email address"},                      "Private", "email in isolation (interpretation: contact detail)"),
    ({"address"},                            "Private", "street/mailing address in isolation"),
    ({"birthplace"},                         "Private", "birthplace in isolation"),
    ({"mother's maiden name"},               "Private", "mother's maiden name in isolation"),
    ({"bank account number"},                "Private", "bank account (BSB + acct) in isolation"),
    ({"birthplace", "mother's maiden name"}, "Private", "two private-isolation elements, no name"),

    # --- Confidential: name + ANY combination element ---
    ({"person", "phone number"},             "Confidential", "name + phone"),
    ({"person", "date of birth"},            "Confidential", "name + birth date"),
    ({"person", "address"},                  "Confidential", "name + address"),
    ({"person", "birthplace"},               "Confidential", "name + birthplace"),
    ({"person", "email address"},            "Confidential", "name + email (interpretation)"),

    # --- Confidential: element confidential in isolation (no identity needed) ---
    ({"credit card number"},                 "Confidential", "credit card PAN in isolation"),
    ({"tax file number"},                    "Confidential", "TFN in isolation"),
    ({"medicare number"},                    "Confidential", "Medicare number in isolation"),
    ({"passport number"},                    "Confidential", "passport / government identifier in isolation"),
    ({"driver's licence number"},            "Confidential", "government identifier in isolation"),
    ({"biometric data"},                     "Confidential", "biometric information in isolation"),
    ({"digital signature"},                  "Confidential", "electronic/digitised signature in isolation"),
    ({"medical condition"},                  "Confidential", "health/sensitive information in isolation"),
    ({"person", "credit card number"},       "Confidential", "name + PAN (still Confidential, not Highly)"),
    ({"person", "medical condition"},        "Confidential", "name only + health: no combination -> not Highly"),

    # --- Highly Confidential: sensitive info + Confidential customer PII ---
    ({"person", "phone number", "medical condition"}, "Highly Confidential", "sensitive + (name+phone)"),
    ({"person", "address", "medical condition"},      "Highly Confidential", "sensitive + (name+address)"),
]


def test_levels_match_girp():
    for labels, expected, why in CASES:
        got = classify_elements(labels)
        assert got == expected, f"{sorted(labels)} -> {got!r}, expected {expected!r}  ({why})"


def test_monotonic_rank_ordering():
    # Adding elements must never lower the classification.
    base = {"phone number"}
    assert RANK[classify_elements(base)] <= RANK[classify_elements(base | {"person"})]
    assert RANK[classify_elements({"person", "phone number"})] <= \
           RANK[classify_elements({"person", "phone number", "medical condition"})]


def test_explain_reports_level_and_reasons():
    out = explain({"person", "phone number", "medical condition"})
    assert out["level"] == "Highly Confidential"
    assert out["rank"] == RANK["Highly Confidential"]
    assert out["reasons"], "explain() must give at least one reason"


def test_person_stopwords_filtered():
    # Pronouns mis-tagged as a person must be dropped (not real identifiers).
    assert found_labels({"person": ["me"], "phone number": ["+61 400 000 111"]}) == {"phone number"}
    assert found_labels({"person": ["me"]}) == set()
    assert found_labels({"person": ["John Smith"], "phone number": ["x"]}) == {"person", "phone number"}
    assert found_labels({"person": [], "email address": ["a@b.com"]}) == {"email address"}


def test_stopword_yields_correct_girp():
    # "call me on <phone>" -> phone in isolation -> Private (not Confidential).
    assert classify_elements(found_labels({"person": ["me"], "phone number": ["x"]})) == "Private"
    # A real name + phone is still Confidential.
    assert classify_elements(found_labels({"person": ["Jane Roe"], "phone number": ["x"]})) == "Confidential"


def _run():
    passed = 0
    for labels, expected, why in CASES:
        got = classify_elements(labels)
        ok = got == expected
        passed += ok
        print(f"{'OK ' if ok else 'XX '} {expected:18s} <- {sorted(labels)}")
        if not ok:
            print(f"      ^ GOT {got!r} (rule: {why})")
    test_monotonic_rank_ordering()
    test_explain_reports_level_and_reasons()
    test_person_stopwords_filtered()
    test_stopword_yields_correct_girp()
    print(f"\nGIRP rule alignment: {passed}/{len(CASES)} cases passed; ordering + explain() + filtering OK.")
    if passed != len(CASES):
        raise SystemExit(1)
    print("All GIRP tests passed.")


if __name__ == "__main__":
    _run()
