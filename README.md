# Local PII & GIRP Information Classifier

Detects PII and classifies text into the four-tier **GIRP** sensitivity scheme:
**Public -> Private -> Confidential -> Highly Confidential**.

The shipped runtime is local-first: model files load from `models/`, and inference does not call
Hugging Face, external APIs, or the internet.

## Clean Directory Layout

| Path | Purpose |
|---|---|
| `pii_model/` | Python package with the GIRP rules, model loader, DataFrame helpers, and Australian hybrid layer |
| `models/base/` | Base GLiNER2 checkpoint and tokenizer/config files |
| `models/fine-tuned/` | Recommended fine-tuned checkpoint to use by default |
| `models/lora-adapter/` | LoRA adapter/training artifact retained for provenance/reuse |
| `wheelhouse/` | Windows Python 3.12 wheels for offline install from GitHub-downloaded files |
| `scripts/download_windows.ps1` | One-file Windows downloader using exact GitHub `/raw/` links |
| `requirements.txt`, `requirements-hybrid.txt` | Runtime dependency lists |
| `setup.bat`, `setup.sh` | Local install helpers |
| `AUPII.md` | Australian hybrid details and baseline performance |

Ignored local development/evaluation files such as `data/`, `reports/`, caches, and `.venv/` are not
part of the plug-and-play GitHub package.

## Windows Download: GitHub Links Only

Prerequisite: Windows with **Python 3.12** available as `py -3.12`.

Create an empty folder, for example `C:\PII-Model`, open **PowerShell** in that folder, then run:

```powershell
$ScriptUrl = "https://github.com/Glass-Half-Full/PII-Model/raw/pii-model-offline-2026-06-19/scripts/download_windows.ps1"
Invoke-WebRequest -Uri $ScriptUrl -OutFile "download_windows.ps1"
powershell -ExecutionPolicy Bypass -File .\download_windows.ps1 -Install
```

That script downloads the runtime files, both model checkpoints, the LoRA adapter artifact, and all
Windows Python wheels from GitHub raw URLs only. It does **not** use Brew, Git LFS, codeload ZIPs,
Hugging Face, PyPI, or `python -m spacy download`.

If you want to download first and install separately:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_windows.ps1
py -3.12 -m pip install --no-index --find-links wheelhouse -r requirements.txt -r requirements-hybrid.txt
```

Key direct model links:

- [Base model weights](https://github.com/Glass-Half-Full/PII-Model/raw/pii-model-offline-2026-06-19/models/base/model.safetensors)
- [Fine-tuned model weights](https://github.com/Glass-Half-Full/PII-Model/raw/pii-model-offline-2026-06-19/models/fine-tuned/model.safetensors)
- [LoRA adapter weights](https://github.com/Glass-Half-Full/PII-Model/raw/pii-model-offline-2026-06-19/models/lora-adapter/adapter_model.safetensors)
- [Windows downloader script](https://github.com/Glass-Half-Full/PII-Model/raw/pii-model-offline-2026-06-19/scripts/download_windows.ps1)

Expected model sizes:

| File | Bytes |
|---|---:|
| `models/base/model.safetensors` | `833938108` |
| `models/fine-tuned/model.safetensors` | `833938108` |
| `models/lora-adapter/adapter_model.safetensors` | `5329152` |

Expected SHA-256 hashes:

```text
845fc4bd93c525b86124c58ab4f56c9eacf8587953086b14c501fab25957c007  models/base/model.safetensors
1ff2a86d7470057cc200f94f1c7fd078c2ace437065a8c14c77d1b80a345fa92  models/fine-tuned/model.safetensors
eca4d810c9480a59a621d11ba2d5ab56a409cb349f9fc03e3bc9c9012355b73c  models/lora-adapter/adapter_model.safetensors
```

## DataFrame Use

Recommended Australian hybrid path:

```python
import pandas as pd
from pii_model.aupii import load_hybrid, classify_columns_hybrid

model, analyzer, device = load_hybrid("models/fine-tuned")

df = pd.DataFrame({
    "customer_note": [
        "Call Sarah Lee on 02 9000 0000 about the refund.",
        "The quarterly report is ready for review.",
    ],
    "agent_comment": [
        "Email sarah.lee@example.com before 5pm.",
        "No personal information in this row.",
    ],
})

result = classify_columns_hybrid(
    model,
    analyzer,
    df,
    ["customer_note", "agent_comment"],
    threshold=0.7,
    progress=False,
)

print(result[[
    "girp_level",
    "needs_review",
    "customer_note_girp_level",
    "customer_note_girp_elements",
    "agent_comment_girp_level",
    "agent_comment_girp_elements",
]])
```

CSV example:

```python
import pandas as pd
from pii_model.aupii import load_hybrid, classify_columns_hybrid

model, analyzer, device = load_hybrid("models/fine-tuned")

df = pd.read_csv("your_file.csv")
result = classify_columns_hybrid(
    model,
    analyzer,
    df,
    ["customer_note", "agent_comment"],
    threshold=0.7,
    progress=False,
)
result.to_csv("classified.csv", index=False)
```

The output adds per-column fields such as `<column>_girp_level`, `<column>_girp_elements`, and
`<column>_needs_review`, plus row-level `girp_level` and `needs_review`.

Base-model fallback without Presidio:

```python
import pandas as pd
from pii_model.girp import load_local_model, classify_columns

model, device = load_local_model("models/fine-tuned")
df = pd.read_csv("your_file.csv")
result = classify_columns(model, df, ["customer_note", "agent_comment"], progress=False)
```

## Australian Hybrid

`pii_model/aupii.py` combines:

- GLiNER2 for fuzzy/contextual PII such as names, phones, addresses, DOB, and health terms.
- Microsoft Presidio pattern/checksum recognizers for email, credit card, IBAN/bank IDs, TFN,
  Medicare, ABN, ACN, and custom BSB.
- The deterministic GIRP rules in `pii_model/girp.py`.

No spaCy language-model download is needed. Presidio is supplied with a no-op NLP engine because this
package uses Presidio for pattern/checksum recognition and GLiNER2 for NER.

## Current Baseline

Verified local/offline run over a 10,000-row benchmark with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`:

- Finished error-free with no runtime network access.
- CPU throughput: about 15 rows/s.
- Micro-average PII F1: 78.6%.
- GIRP level accuracy: 64.3%.
- Strongest labels: email 99.8 F1, bank 86.9, person 81.9, address 78.8, phone 72.9.

Card F1 is lower on the public benchmark because many benchmark cards are non-Luhn synthetic values;
for real Luhn-valid cards, the Presidio checksum recognizer is the intended high-precision path.

## GIRP Levels

| Level | Applies when |
|---|---|
| **Public** | No personal information detected |
| **Private** | A personal detail in isolation: phone, email, address, birthplace, mother's maiden name, bank account, or name |
| **Confidential** | Full name plus a combination element, or a confidential identifier such as card, TFN, Medicare, passport, licence, or health information |
| **Highly Confidential** | Health/sensitive information together with Confidential-level customer PII |

## Smoke Check

After install, run:

```python
import pandas as pd
from pii_model.aupii import load_hybrid, classify_columns_hybrid

model, analyzer, device = load_hybrid("models/fine-tuned")
df = pd.DataFrame({"text": ["Call Sarah Lee on 02 9000 0000.", "No PII here."]})
result = classify_columns_hybrid(model, analyzer, df, ["text"], progress=False)
print(result[["text_girp_level", "text_girp_elements", "girp_level"]])
```

Expected behavior: the first row is PII-bearing and the second row remains `Public`.
