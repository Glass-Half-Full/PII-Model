"""Deterministic tests that the GIRP rule engine matches the four-tier classification rules.

Run directly (no pytest needed):   python test_girp.py
Or under pytest:                    pytest test_girp.py
"""
from girp import classify_elements, explain, found_labels, is_valid_entity, LEVELS, RANK

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

    # --- Highly Confidential: sensitive + a Confidential-isolation identifier (no name needed) ---
    ({"medical condition", "tax file number"},         "Highly Confidential", "health + TFN (gov-id) -> Highly"),
    ({"medical condition", "medicare number"},         "Highly Confidential", "health + Medicare -> Highly"),
    ({"medical condition", "credit card number"},      "Highly Confidential", "health + card (PAN) -> Highly"),
    ({"medical condition", "passport number"},         "Highly Confidential", "health + passport -> Highly"),
    ({"medical condition", "driver's licence number"}, "Highly Confidential", "health + licence -> Highly"),
    # bank account is Private-in-isolation (not Confidential), so it does NOT escalate to Highly:
    ({"medical condition", "bank account number"},     "Confidential", "health + bank acct (Private-iso) -> Confidential"),
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
    assert found_labels({"person": ["John Smith"], "phone number": ["0412 345 678"]}) == {"person", "phone number"}
    assert found_labels({"person": [], "email address": ["a@b.com"]}) == {"email address"}


def test_stopword_yields_correct_girp():
    # "call me on <phone>" -> phone in isolation -> Private (not Confidential).
    assert classify_elements(found_labels({"person": ["me"], "phone number": ["0412 345 678"]})) == "Private"
    # A real name + phone is still Confidential.
    assert classify_elements(found_labels({"person": ["Jane Roe"], "phone number": ["0412 345 678"]})) == "Confidential"


def test_format_validation_removes_false_positives():
    # 'office' is not an address (no number, no street word) -> dropped.
    assert found_labels({"address": ["office"]}) == set()
    assert found_labels({"address": ["12 King Street"]}) == {"address"}   # has a number
    assert found_labels({"address": ["King Street"]}) == {"address"}      # street word
    # stray numbers are not cards / TFNs of the right size.
    assert found_labels({"credit card number": ["order 12345"]}) == set()
    assert found_labels({"credit card number": ["4111 1111 1111 1111"]}) == {"credit card number"}
    assert found_labels({"tax file number": ["unit 12"]}) == set()
    assert found_labels({"tax file number": ["123 456 789"]}) == {"tax file number"}
    # email/phone need the right shape.
    assert found_labels({"email address": ["contact us"]}) == set()
    assert found_labels({"email address": ["a@b.com"]}) == {"email address"}
    assert found_labels({"phone number": ["call the desk"]}) == set()
    assert found_labels({"phone number": ["02 9000 0000"]}) == {"phone number"}


def test_is_valid_entity_direct():
    assert is_valid_entity("credit card number", "4111 1111 1111 1111")
    assert not is_valid_entity("credit card number", "order 12345")
    assert is_valid_entity("address", "10 Oak Avenue")
    assert not is_valid_entity("address", "the office")
    assert is_valid_entity("person", "Jane Roe")
    assert not is_valid_entity("person", "me")
    assert is_valid_entity("email address", "x@y.org")
    assert not is_valid_entity("email address", "no at sign")


def test_regex_backstop():
    from girp import regex_elements
    # Luhn-valid card is caught even if the model misses it; a non-Luhn number is not.
    assert "credit card number" in regex_elements("paid with 4111 1111 1111 1111 today")
    assert "credit card number" not in regex_elements("ref 4111 1111 1111 1112")
    assert "email address" in regex_elements("write to a.b@example.com please")
    assert "phone number" in regex_elements("call +61 434 649 757 to confirm")
    assert regex_elements("order number 12345 shipped Tuesday") == set()


def test_robust_batch_extract_recovers_from_oom():
    # Adaptive batch sizing must recover from a CUDA OOM without a real GPU.
    from girp import _robust_batch_extract

    class FakeModel:
        def __init__(self):
            self.calls = 0
            self.on_cpu = False

        def batch_extract_entities(self, texts, labels, batch_size, threshold):
            self.calls += 1
            if batch_size > 1:                      # simulate OOM until batch_size drops to 1
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return [{"entities": {}} for _ in texts]

        def to(self, device):
            self.on_cpu = (device == "cpu")

    m = FakeModel()
    res = _robust_batch_extract(m, ["a", "b", "c"], ["person"], 0.5, 8, progress=False, desc="t")
    assert len(res) == 3            # all rows processed despite OOM
    assert m.calls >= 2            # it retried with smaller batches


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
    test_format_validation_removes_false_positives()
    test_is_valid_entity_direct()
    test_regex_backstop()
    test_robust_batch_extract_recovers_from_oom()
    print(f"\nGIRP rule alignment: {passed}/{len(CASES)} cases passed; ordering + explain() + "
          "validation + OOM-recovery OK.")
    if passed != len(CASES):
        raise SystemExit(1)
    print("All GIRP tests passed.")


if __name__ == "__main__":
    _run()
