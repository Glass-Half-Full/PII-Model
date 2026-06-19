# Australian-Ready Hybrid PII System

`pii_model/aupii.py` is the recommended runtime path for Australian PII/GIRP classification.

It combines:

```text
text
  -> GLiNER2 fine-tuned checkpoint: names, phone, address, DOB, health, passport/licence
  -> Presidio pattern/checksum recognizers: email, credit card, bank IDs, TFN, Medicare, ABN, ACN, BSB
  -> GIRP rules: Public / Private / Confidential / Highly Confidential
```

## Why Hybrid

The model gives recall for contextual/fuzzy PII. Presidio gives precision for structured identifiers
where regex/checksums are stronger than zero-shot NER. This reduces numeric false positives while
adding reliable Australian identifier coverage.

## No spaCy Model Download

The hybrid does **not** require:

```text
python -m spacy download en_core_web_sm
```

Presidio still has a Python package dependency chain which includes the `spacy` package, so the
Windows wheelhouse includes a spaCy wheel. But the runtime does not download or load a spaCy language
model. `build_analyzer()` supplies a no-op NLP engine because GLiNER2 handles NER and Presidio is used
for pattern/checksum recognition.

## Windows Offline Install

Use the GitHub-only PowerShell path in `README.md`:

```powershell
$ScriptUrl = "https://github.com/Glass-Half-Full/PII-Model/raw/pii-model-offline-2026-06-19/scripts/download_windows.ps1"
Invoke-WebRequest -Uri $ScriptUrl -OutFile "download_windows.ps1"
powershell -ExecutionPolicy Bypass -File .\download_windows.ps1 -Install
```

This downloads `models/`, `pii_model/`, and `wheelhouse/` from exact GitHub raw links.

## Usage

```python
import pandas as pd
from pii_model.aupii import load_hybrid, classify_columns_hybrid

model, analyzer, device = load_hybrid("models/fine-tuned")

df = pd.read_csv("your_file.csv")
result = classify_columns_hybrid(
    model,
    analyzer,
    df,
    ["notes", "comments"],
    threshold=0.7,
    progress=False,
)
```

The output adds:

- `<column>_girp_level`
- `<column>_girp_elements`
- `<column>_needs_review`
- row-level `girp_level`
- row-level `needs_review`

## Australian Coverage

- TFN, Medicare, ABN, ACN: Presidio Australian recognizers.
- BSB: custom recognizer for `NNN-NNN` bank-state-branch values.
- Credit card: Luhn-checked Presidio recognizer.
- Email and bank identifiers: Presidio pattern/checksum recognizers.
- Names, addresses, phones, DOB, health terms, passport/licence: GLiNER2.

## Current Baseline

Verified local/offline 10,000-row run:

| Metric | Result |
|---|---:|
| Runtime network access | None |
| CPU throughput | ~15 rows/s |
| Micro-average PII F1 | 78.6% |
| GIRP level accuracy | 64.3% |
| Over-classification | 26.5% |
| Under-classification | 9.2% |

Label-level highlights:

| Label | F1 |
|---|---:|
| Email | 99.8 |
| Bank | 86.9 |
| Person | 81.9 |
| Address | 78.8 |
| Phone | 72.9 |

The card score on public benchmarks is depressed by non-Luhn synthetic values. On real Luhn-valid
cards, the Presidio checksum recognizer is the high-precision path.

## Human Review

Rows are flagged `needs_review` when:

- The row is classified as `Highly Confidential`.
- The GLiNER2 and Presidio paths disagree on the GIRP level.

This is the production safety band for high-stakes or uncertain rows.
