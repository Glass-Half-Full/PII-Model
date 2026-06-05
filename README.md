# Local PII & GIRP Information Classifier

Detects PII and classifies text into the four-tier **GIRP** sensitivity scheme
(**Public → Private → Confidential → Highly Confidential**) — running **100% locally**.
No internet, no Hugging Face Hub, no API calls: the model and all config load from files in
this repository. Built on a local 205M-parameter GLiNER2 model (Apache-2.0).

## What it does
- Tag the **GIRP sensitivity level** of any text column(s) in a pandas DataFrame
- Identify **PII** (names, emails, phones, SSNs, addresses, cards, government IDs, …)
- Classify text (sentiment / intent / topic)
- Redact PII from text

## Requirements
- **Python 3.13** (3.11+ works)
- Packages in `requirements.txt` (installed with pip / your internal mirror)
- Model weights `model.safetensors` (~800 MB) — **already included in this repo**
- CPU works fine; an NVIDIA GPU (e.g. RTX 2050) is used automatically if a CUDA build of PyTorch is installed

> Everything needed to run the model is in this folder. After cloning, **no downloads are required at run time.**

## Get started — 3 steps

**1. Install dependencies**

Windows:
```
setup.bat
```
macOS / Linux:
```
bash setup.sh
```
or directly, on any OS:
```
python -m pip install -r requirements.txt
```

**2. Open the notebook**
```
python -m jupyter lab gliner2_pii_demo.ipynb
```

**3. Point it at your data**

In the **“Scan your columns”** cell, set your DataFrame and the columns to scan, then run it:
```python
df = pd.read_csv("your_file.csv")                 # your data
COLUMNS = ["customer_note", "agent_comment"]      # the columns to scan
result = classify_columns(model, df, COLUMNS)
```
`result` gains a `<col>_girp_level` (and `<col>_girp_elements`) for each scanned column, plus an
overall **`girp_level`** per row — the most sensitive level found across those columns.

## Use it in a script (no notebook)
```python
import pandas as pd
from girp import load_local_model, classify_columns

model, device = load_local_model()                # local files only — no internet
df = pd.read_csv("your_file.csv")
result = classify_columns(model, df, ["notes", "comments"])
result.to_csv("classified.csv", index=False)
```

## Australian-ready hybrid (recommended for best accuracy)
For the strongest, Australian-tuned setup, use `aupii.py`: **gliner2** (recall) + **Microsoft
Presidio** checksum recognizers (Luhn cards, IBAN/bank, SSN) + **Australian** recognizers
(**TFN, Medicare, ABN, ACN, BSB**) + the GIRP rules. It raises GIRP accuracy, cuts
over-classification vs the model alone, and validates AU identifiers by checksum. A 330 MB PII
model option (`gliner-pii-small`) suits a 4 GB RTX 2050 / CPU. Install `requirements-hybrid.txt`;
full guide in [`AUPII.md`](AUPII.md).

## GIRP levels

| Level | Applies when… |
|---|---|
| **Public** | No personal information present |
| **Private** | A personal detail *in isolation* — phone, address, birthplace, mother's maiden name, or bank account |
| **Confidential** | A **full name + any** of {phone, DOB, address, birthplace, mother's maiden name}; **or** a card number (PAN), government identifier (TFN / Medicare / passport / licence), or health information on its own |
| **Highly Confidential** | Health / sensitive information **together with** Confidential-level customer PII |

"GIRP" here denotes a generic four-tier information-classification scheme; all rules are defined in
this repository (`girp.py`) and cover the personal-information rules that can be derived from text.
Document-type rules (board papers, audit reports, etc.) need document-level context and are out of scope.

## Validate
```
python test_girp.py
```
Checks the GIRP rules (deterministic, no model needed).

## Performance baseline
See [`BASELINE.md`](BASELINE.md) for measured precision/recall/F1 on a public labeled PII dataset —
email F1 100%, phone/name/address ~80%+, micro-avg F1 ~78%. The main error mode is *over*-classification
on PII-dense text (tunable via `threshold`). Reproduce with `python benchmark.py` (needs internet).

## Files
| File | Purpose |
|---|---|
| `gliner2_pii_demo.ipynb` | Main notebook (setup → scan columns → PII → classification → redaction → synthetic validation) |
| `girp.py` | GIRP rules + format validation + regex/Luhn backstop + OOM-safe local loader + DataFrame helpers |
| `test_girp.py` | GIRP rule, validation, regex & OOM-recovery tests |
| `synthetic.py` | Synthetic data generator for validation (all tiers + false-positive bait) |
| `benchmark.py`, `BASELINE.md` | Performance baseline on a public labeled PII set (`benchmark.py` needs internet) |
| `PRODUCTION.md` | Model-comparison findings + production roadmap (fine-tuning, calibration, monitoring) |
| `aupii.py`, `test_aupii.py`, `AUPII.md` | **Australian-ready hybrid** (gliner2 + Presidio checksums + AU TFN/Medicare/ABN/ACN/BSB) |
| `requirements-hybrid.txt` | Extra deps for the hybrid (Presidio, spaCy, gliner) |
| `requirements.txt`, `setup.bat`, `setup.sh` | Dependency install (no venv/conda) |
| `model.safetensors`, `config.json`, `tokenizer*.json`, `added_tokens.json`, `special_tokens_map.json`, `encoder_config/` | The local model + tokenizer + config |

## Notes
- **Offline:** loads with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. Copy this folder to an air-gapped machine and it runs unchanged.
- **Device:** automatic CUDA → CPU. On CUDA it uses fp16 (~0.4 GB VRAM — fits a 4 GB card). If you hit out-of-memory, lower `batch_size`.
- **Tuning:** `threshold` (default `0.7` — best precision/recall balance; use `0.5` for max recall, `0.85+` for fewest false flags) and the GIRP element label sets in `girp.py` are editable.
- **Improving further:** see [`PRODUCTION.md`](PRODUCTION.md) — a bigger model gives ~0% gain for 3× cost; the real lever is fine-tuning on labeled domain data.
- **Robust:** format validation removes false positives (a "card" needs 13–19 digits, an "address" a number/street word, pronouns aren't names); a Luhn-checked regex backstop catches structured PII the model misses (cards, emails, international phones); CUDA out-of-memory self-recovers (batch halving → CPU fallback). Validated on synthetic data across all four tiers (`synthetic.py`).

---
Model: GLiNER2 (Apache-2.0), run entirely locally by this repository.
