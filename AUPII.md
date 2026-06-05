# Australian-ready hybrid PII system (`aupii.py`)

A "battle-hardened" combined-assessment pipeline that uses each engine where it is strongest:

```
            ┌─ gliner2 (zero-shot)  ── names, phone, address, DOB, birthplace, health, passport/licence
 text ──▶  ─┤                                                   (recall)
            └─ Presidio (checksum/regex) ── email, credit card (Luhn), IBAN/bank, SSN,
                 + Australian recognizers ── TFN, Medicare, ABN, ACN, BSB  (precision)
                          │
                          ▼
              GIRP rules (girp.classify_elements) ──▶ Public / Private / Confidential / Highly Confidential
```

**Why hybrid:** a bigger model gave ~0 gain (see `PRODUCTION.md`). The real win is letting the
zero-shot model do *recall* and a checksum engine do *precision* — which removes the
account→card / numeric-ID false positives that drove over-classification, and adds reliable,
checksum-validated **Australian** identifiers.

## Results (same 2,000-row benchmark + synthetic AU)

| Setup | Micro-F1 | GIRP acc | Over-class | Speed | Notes |
|---|---|---|---|---|---|
| gliner2 alone @0.7 | 77.3% | 62.7% | 29.5% | 17 rows/s | baseline |
| **gliner2 + Presidio @0.7** | ~77% | **~65%** | **~26%** | ~18–30 rows/s | bank 63→83; AU checksums; **best accuracy** |
| gliner-pii-small + Presidio @0.3 | ~72% | ~59% | ~30% | **~35–51 rows/s** | 330 MB; **best for RTX 2050 / CPU** |
| **Hybrid on synthetic AU** | — | **99–100%** | 0% | — | TFN/Medicare/card via checksums |

Card F1 looks low on the *public* benchmark only because 85% of its cards are non-Luhn synthetic;
on **real** (Luhn-valid) cards Presidio is near-perfect and removes account→card false positives.

## Australian coverage (checksum-validated)
- **Tax File Number (TFN)**, **Medicare**, **ABN**, **ACN** — Presidio AU recognizers (checksum).
- **BSB** — custom recognizer (context-gated).
- **Credit card** — Luhn. **IBAN/bank** — checksum/length. Passport & driver's licence — gliner2
  zero-shot (handles AU formats Presidio's US recognizers miss).

## Choosing the ML engine for your hardware
- **`gliner2-base` (default, 833 MB)** — best accuracy; fp16 on CUDA fits your 4 GB RTX 2050.
- **`knowledgator/gliner-pii-small-v1.0` (330 MB)** — smaller + faster (~2–3× throughput), ideal
  for RTX 2050 / CPU at a small accuracy cost. Load via `aupii.load_gliner_pii()`.

## Install (offline-capable)
```
python -m pip install -r requirements.txt -r requirements-hybrid.txt
python -m spacy download en_core_web_sm    # one-time; the hybrid uses Presidio's pattern/checksum
                                           # recognisers + gliner2 for NER, so the small model suffices
```
After this, everything runs **locally/offline** (gliner2 weights + spaCy model + regex; no API calls).

## Usage
```python
import pandas as pd
from aupii import load_hybrid, classify_columns_hybrid

model, analyzer, device = load_hybrid()          # gliner2 + Presidio (AU recognizers), local
df = pd.read_csv("your_file.csv")
result = classify_columns_hybrid(model, analyzer, df, ["notes", "comments"])
# -> <col>_girp_level, <col>_girp_elements, and overall girp_level per row
```
Lightweight RTX 2050 variant:
```python
from aupii import load_gliner_pii, build_analyzer, classify_columns_hybrid
model = load_gliner_pii("knowledgator/gliner-pii-small-v1.0")   # 330 MB
analyzer = build_analyzer()
result = classify_columns_hybrid(model, analyzer, df, ["notes"], threshold=0.3)
```

Validate: `python test_aupii.py`.
