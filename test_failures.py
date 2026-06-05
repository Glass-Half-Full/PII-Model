"""Failure-mode regression suite — deterministic, no model or extras required.

Guards the precision/validation + GIRP-rule logic against the false-positive bait and edge cases
that broke (or could break) earlier: numeric IDs mis-tagged as phone/card/address, titles/pronouns
as names, empty/long text, and the GIRP combination rules (incl. health + Confidential-isolation).

Run:  python test_failures.py   (or)   pytest test_failures.py
"""
from girp import found_labels, classify_elements, is_valid_entity

# --- numeric / structured false-positive bait must NOT validate as the wrong element ----------
PHONE_BAIT = ["06-184755-866851-3", "192.168.1.100", "0x1ff90b9ec7fa013d7fadc6ae", "0984477344415390"]
ADDRESS_BAIT = ["-38.9302,113.5422", "250.59.196.86", "99578", "34nfuko7AiX7MXkz6syvQ38"]
PERSON_BAIT = ["me", "you", "students", "Male", "Female", "the office",
               "Investor Program Supervisor", "Human Group Coordinator"]


def test_phone_bait_rejected():
    for v in PHONE_BAIT:
        assert not is_valid_entity("phone number", v), v
    assert is_valid_entity("phone number", "02 9000 0000")      # a real phone still validates
    assert is_valid_entity("phone number", "+61 412 345 678")


def test_address_bait_rejected():
    for v in ADDRESS_BAIT:
        assert not is_valid_entity("address", v), v
    assert is_valid_entity("address", "10 Oak Avenue")          # real address validates
    assert is_valid_entity("address", "King Street")


def test_person_bait_rejected():
    for v in PERSON_BAIT:
        assert not is_valid_entity("person", v), v
    assert is_valid_entity("person", "Jane Roe")                # a real name validates


def test_found_labels_drops_bait_in_context():
    # numbers the model might tag as phone/address are filtered out -> stays Public
    ents = {"phone number": ["192.168.1.100"], "address": ["99578"], "person": ["the office"]}
    assert found_labels(ents) == set()
    assert classify_elements(found_labels(ents)) == "Public"


def test_empty_and_long_text_safe():
    assert found_labels({}) == set()
    assert found_labels({"person": [], "phone number": [""]}) == set()
    assert not is_valid_entity("person", "")
    assert not is_valid_entity("phone number", "x")
    long_val = "word " * 5000
    is_valid_entity("person", long_val)                       # must not raise on very long input
    assert found_labels({"person": [long_val]}) <= {"person"}  # handles long input without crashing


def test_health_combination_rules():
    # health + Confidential-isolation identifier -> Highly Confidential (A1 invariant)
    for conf in ("tax file number", "medicare number", "credit card number",
                 "passport number", "driver's licence number"):
        assert classify_elements({"medical condition", conf}) == "Highly Confidential", conf
    # health alone or with only a name (no combination) stays Confidential
    assert classify_elements({"medical condition"}) == "Confidential"
    assert classify_elements({"person", "medical condition"}) == "Confidential"
    # health + bank account (Private-in-isolation) does NOT escalate
    assert classify_elements({"medical condition", "bank account number"}) == "Confidential"


def test_name_plus_combination_is_confidential():
    assert classify_elements({"person", "phone number"}) == "Confidential"
    assert classify_elements({"person"}) == "Private"
    assert classify_elements(set()) == "Public"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\nAll {len(fns)} failure-mode tests passed.")


if __name__ == "__main__":
    _run()
