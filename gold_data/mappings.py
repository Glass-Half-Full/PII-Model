"""Source PII label -> GIRP element label mappings (offline, no model).

Each source dataset uses its own label vocabulary. We map those to the GIRP element
vocabulary (girp.GIRP_PII_LABELS) so a single gold schema and a single evaluator work
across all of them. A value of ``None`` means "this label has no clean GIRP equivalent —
drop the span from scoring." Dropping (rather than guessing) avoids the over-classification
inflation noted in BASELINE.md, where mapping coarse/ambiguous labels created phantom PII.

Design choices worth knowing:
  * SSN / SOCIALNUM -> "tax file number". GIRP has no SSN element and the production
    Presidio mapping (aupii.PRESIDIO2GIRP) already maps US_SSN -> "tax file number", so the
    MODEL emits "tax file number" for an SSN. Gold must agree or we'd score spurious FPs.
    (Type conflation for per-entity span-F1 is a known, documented limitation.)
  * IBAN / BBAN / ACCOUNTNUMBER -> "bank account number" (Presidio maps these the same way).
  * Generic/coarse labels (CITY, ZIPCODE, DATE, AGE, IDCARDNUM, CVV, username, ...) -> None.
"""
from __future__ import annotations

# Reuse the legacy ai4privacy/pii-masking-200k map already defined for the old benchmark.
from benchmark import GOLD2GIRP as AI4PRIVACY_200K


def _person(*keys):
    return {k: "person" for k in keys}


# ai4privacy/open-pii-masking-500k (verified label vocab: GIVENNAME, SURNAME, TELEPHONENUM,
# EMAIL, STREET, BUILDINGNUM, TAXNUM, SOCIALNUM, PASSPORTNUM, DRIVERLICENSENUM, CREDITCARDNUMBER,
# CITY, ZIPCODE, DATE, TIME, AGE, IDCARDNUM, TITLE, GENDER, SEX, ... + finance labels).
AI4PRIVACY_500K = {
    **_person("GIVENNAME", "SURNAME", "MIDDLENAME", "FIRSTNAME", "LASTNAME", "PREFIX"),
    "EMAIL": "email address",
    "TELEPHONENUM": "phone number", "PHONENUMBER": "phone number",
    "STREET": "address", "BUILDINGNUM": "address", "SECONDARYADDRESS": "address",
    "ACCOUNTNUMBER": "bank account number", "IBAN": "bank account number",
    "CREDITCARDNUMBER": "credit card number",
    "TAXNUM": "tax file number", "SOCIALNUM": "tax file number",
    "PASSPORTNUM": "passport number",
    "DRIVERLICENSENUM": "driver's licence number",
    "DOB": "date of birth", "DATEOFBIRTH": "date of birth",
    # known-but-dropped (coarse, ambiguous, or non-personal under GIRP):
    "CITY": None, "STATE": None, "ZIPCODE": None, "COUNTY": None, "NEARBYGPSCOORDINATE": None,
    "DATE": None, "TIME": None, "AGE": None, "GENDER": None, "SEX": None, "EYECOLOR": None,
    "HEIGHT": None, "TITLE": None, "JOBTITLE": None, "JOBAREA": None, "JOBTYPE": None,
    "IDCARDNUM": None, "BIC": None, "ACCOUNTNAME": None, "CREDITCARDCVV": None,
    "CREDITCARDISSUER": None, "MASKEDNUMBER": None, "PIN": None, "PASSWORD": None,
    "USERNAME": None, "URL": None, "IP": None, "IPV4": None, "IPV6": None, "MAC": None,
    "PHONEIMEI": None, "VEHICLEVIN": None, "VEHICLEVRM": None, "BITCOINADDRESS": None,
    "ETHEREUMADDRESS": None, "AMOUNT": None, "CURRENCY": None, "CURRENCYCODE": None,
    "CURRENCYNAME": None, "CURRENCYSYMBOL": None, "ORDINALDIRECTION": None, "SUFFIX": None,
    "COMPANYNAME": None, "ORGANIZATION": None,
}

# gretelai/synthetic_pii_finance_multilingual (verified lowercase snake_case vocab).
GRETEL_FINANCE = {
    "name": "person", "first_name": "person", "last_name": "person",
    "email": "email address",
    "phone_number": "phone number",
    "street_address": "address",
    "iban": "bank account number", "bban": "bank account number",
    "credit_card_number": "credit card number",
    "ssn": "tax file number",
    "driver_license_number": "driver's licence number",
    "date_of_birth": "date of birth",
    # known-but-dropped:
    "company": None, "date": None, "time": None, "date_time": None, "customer_id": None,
    "employee_id": None, "user_name": None, "password": None, "api_key": None,
    "account_pin": None, "credit_card_security_code": None, "swift_bic_code": None,
    "ipv4": None, "ipv6": None, "url": None, "passport_number": "passport number",
}

# Kaggle "PII Data Detection" / PIILO student essays (verified 7-label BIO vocab).
KAGGLE_PII = {
    "NAME_STUDENT": "person",
    "EMAIL": "email address",
    "PHONE_NUM": "phone number",
    "STREET_ADDRESS": "address",
    # known-but-dropped:
    "USERNAME": None, "URL_PERSONAL": None, "ID_NUM": None,
}

REGISTRY = {
    "ai4privacy": AI4PRIVACY_500K,
    "ai4privacy-200k": dict(AI4PRIVACY_200K),
    "gretel-finance": GRETEL_FINANCE,
    "kaggle-pii": KAGGLE_PII,
}


def map_label(source: str, raw: str):
    """Return the GIRP element for a source's raw label, or None to drop it.

    Raises KeyError if ``source`` is unknown (caller must register a mapping first).
    """
    if source not in REGISTRY:
        raise KeyError(f"no label mapping registered for source {source!r}")
    return REGISTRY[source].get(raw)


def is_known(source: str, raw: str) -> bool:
    """True if ``raw`` is explicitly handled (mapped or deliberately dropped) for ``source``.

    False means the label is genuinely unrecognised and should be logged for review rather
    than silently dropped (so new label types in updated dataset releases are not missed).
    """
    return source in REGISTRY and raw in REGISTRY[source]
