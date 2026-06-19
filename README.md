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
- Model weights `model.safetensors` and `model-finetuned/final/model.safetensors` — **already included in this repo via Git LFS**
- CPU works fine; an NVIDIA GPU (e.g. RTX 2050) is used automatically if a CUDA build of PyTorch is installed

> Everything needed to run the model is in this folder. After cloning, **no downloads are required at run time.**

## Download on Windows

Use these direct links if you do not have Git, Git LFS, or Homebrew.

1. Download the source package:

[Download PII-Model-main.zip](https://github.com/Glass-Half-Full/PII-Model/archive/refs/heads/main.zip)

2. Extract the ZIP, then open **PowerShell** inside the extracted `PII-Model-main` folder.

3. Download the real model files into the extracted folder. This replaces any small Git LFS pointer
files from the ZIP:

```powershell
$ProgressPreference = "SilentlyContinue"

New-Item -ItemType Directory -Force -Path "model-finetuned\final" | Out-Null
New-Item -ItemType Directory -Force -Path "model-finetuned\_trainer\final" | Out-Null

Invoke-WebRequest `
  -Uri "https://media.githubusercontent.com/media/Glass-Half-Full/PII-Model/bd03fabdcb81fe3e77cad377c8bc2e7db60a02f5/model.safetensors" `
  -OutFile "model.safetensors"

Invoke-WebRequest `
  -Uri "https://media.githubusercontent.com/media/Glass-Half-Full/PII-Model/bd03fabdcb81fe3e77cad377c8bc2e7db60a02f5/model-finetuned/final/model.safetensors" `
  -OutFile "model-finetuned\final\model.safetensors"

Invoke-WebRequest `
  -Uri "https://media.githubusercontent.com/media/Glass-Half-Full/PII-Model/bd03fabdcb81fe3e77cad377c8bc2e7db60a02f5/model-finetuned/_trainer/final/adapter_model.safetensors" `
  -OutFile "model-finetuned\_trainer\final\adapter_model.safetensors"
```

Direct model links if you prefer clicking and saving manually:

- [Base model.safetensors](https://media.githubusercontent.com/media/Glass-Half-Full/PII-Model/bd03fabdcb81fe3e77cad377c8bc2e7db60a02f5/model.safetensors) -> save as `model.safetensors`
- [Fine-tuned model.safetensors](https://media.githubusercontent.com/media/Glass-Half-Full/PII-Model/bd03fabdcb81fe3e77cad377c8bc2e7db60a02f5/model-finetuned/final/model.safetensors) -> save as `model-finetuned\final\model.safetensors`
- [LoRA adapter_model.safetensors](https://media.githubusercontent.com/media/Glass-Half-Full/PII-Model/bd03fabdcb81fe3e77cad377c8bc2e7db60a02f5/model-finetuned/_trainer/final/adapter_model.safetensors) -> save as `model-finetuned\_trainer\final\adapter_model.safetensors`

Verify the model files are real downloads, not pointer stubs:

```powershell
Get-Item `
  "model.safetensors", `
  "model-finetuned\final\model.safetensors", `
  "model-finetuned\_trainer\final\adapter_model.safetensors" |
  Select-Object Name, Length

Get-FileHash "model.safetensors" -Algorithm SHA256
Get-FileHash "model-finetuned\final\model.safetensors" -Algorithm SHA256
Get-FileHash "model-finetuned\_trainer\final\adapter_model.safetensors" -Algorithm SHA256
```

Expected sizes:

- `model.safetensors`: `833938108` bytes
- `model-finetuned\final\model.safetensors`: `833938108` bytes
- `model-finetuned\_trainer\final\adapter_model.safetensors`: `5329152` bytes

Expected SHA-256 hashes:
```
845fc4bd93c525b86124c58ab4f56c9eacf8587953086b14c501fab25957c007  model.safetensors
1ff2a86d7470057cc200f94f1c7fd078c2ace437065a8c14c77d1b80a345fa92  model-finetuned/final/model.safetensors
eca4d810c9480a59a621d11ba2d5ab56a409cb349f9fc03e3bc9c9012355b73c  model-finetuned/_trainer/final/adapter_model.safetensors
```

If any `.safetensors` file is only a few hundred bytes, it is still a Git LFS pointer file. Download
that file again from the direct link above.

Optional Git LFS path for machines that already have Git LFS:

```powershell
git clone https://github.com/Glass-Half-Full/PII-Model.git
cd PII-Model
git lfs pull
```

The root `model.safetensors` is the base local model. The accepted fine-tuned checkpoint is in
`model-finetuned/final/`; load it with:
```python
from aupii import load_hybrid
model, analyzer, device = load_hybrid("model-finetuned/final")
```

## Plug-and-play DataFrame use

Install the runtime dependencies once:

Windows:
```
setup.bat
```
macOS / Linux:
```
bash setup.sh
```

For the recommended Australian hybrid path, also install the hybrid extras:
```
python -m pip install -r requirements.txt -r requirements-hybrid.txt
python -m spacy download en_core_web_sm
```

After setup, inference runs locally from the downloaded repo files. No Hub/API call is required at
classification time.

Classify an existing pandas DataFrame:
```python
import pandas as pd
from aupii import load_hybrid, classify_columns_hybrid

model, analyzer, device = load_hybrid("model-finetuned/final")

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

Classify a CSV and write the tagged output:
```python
import pandas as pd
from aupii import load_hybrid, classify_columns_hybrid

model, analyzer, device = load_hybrid("model-finetuned/final")

df = pd.read_csv("your_file.csv")
columns_to_scan = ["customer_note", "agent_comment"]

result = classify_columns_hybrid(
    model,
    analyzer,
    df,
    columns_to_scan,
    threshold=0.7,
    progress=False,
)

result.to_csv("classified.csv", index=False)
```

The output adds per-column fields such as `<column>_girp_level`, `<column>_girp_elements`, and
`<column>_needs_review`, plus overall row fields `girp_level` and `needs_review`.

Base-model fallback without the hybrid extras:
```python
import pandas as pd
from girp import load_local_model, classify_columns

model, device = load_local_model("model-finetuned/final")
df = pd.read_csv("your_file.csv")
result = classify_columns(model, df, ["customer_note", "agent_comment"], progress=False)
```

## Australian-ready hybrid (recommended for best accuracy)
For the strongest, Australian-tuned setup, use `aupii.py`: **gliner2** (recall) + **Microsoft
Presidio** checksum recognizers (Luhn cards, IBAN/bank, SSN) + **Australian** recognizers
(**TFN, Medicare, ABN, ACN, BSB**) + the GIRP rules. It raises GIRP accuracy, cuts
over-classification vs the model alone, and validates AU identifiers by checksum. A 330 MB PII
model option (`gliner-pii-small`) suits a 4 GB RTX 2050 / CPU.

Install: `python -m pip install -r requirements.txt -r requirements-hybrid.txt` then
`python -m spacy download en_core_web_sm` — after that it runs **100% offline**. Full guide in [`AUPII.md`](AUPII.md).

```python
from aupii import load_hybrid, classify_columns_hybrid
model, analyzer, _ = load_hybrid()          # gliner2 + Presidio (AU recognizers), local
result = classify_columns_hybrid(model, analyzer, df, ["notes", "comments"])
```

## Runs locally, verified at scale ✅
The Australian hybrid was run over a **10,000-row** dataset with networking disabled
(`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`) and finished **error-free with no network access**:

- **Fully local / no external API calls** — gliner2 weights load from this folder, spaCy
  `en_core_web_sm` is a local package, and Presidio is pure regex/checksums. Nothing reaches out at run time.
- **Throughput** ~15 rows/s on CPU (834 MB resident); faster on the RTX 2050 GPU, or with the 330 MB `gliner-pii-small` engine.
- **Accuracy at scale** (vs gold labels): micro-avg **F1 78.6%** — email 99.8, bank 86.9, person 81.9,
  address 78.8, phone 72.9. GIRP level accuracy 64.3% (over-classification 26.5%, under 9.2% — tune with `threshold`).
- Card F1 looks low *only* on this public benchmark (its cards are non-Luhn synthetic); on real
  Luhn-valid cards Presidio's checksum makes it near-exact and removes account→card false positives.

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

## Smoke check
Run this after setup to confirm the fine-tuned model loads and classifies a small DataFrame:
```python
import pandas as pd
from aupii import load_hybrid, classify_columns_hybrid

model, analyzer, device = load_hybrid("model-finetuned/final")
df = pd.DataFrame({"text": ["Call Sarah Lee on 02 9000 0000.", "No PII here."]})
result = classify_columns_hybrid(model, analyzer, df, ["text"], progress=False)
print(result[["text_girp_level", "text_girp_elements", "girp_level"]])
```

Expected behavior: the first row is flagged as PII-bearing and the second row remains `Public`.

## Files
| File | Purpose |
|---|---|
| `girp.py` | Local model loader, GIRP rules, validation, regex backstops, and DataFrame helpers |
| `aupii.py`, `AUPII.md` | Australian-ready hybrid layer: gliner2 + Presidio checksum/regex recognizers |
| `requirements.txt`, `requirements-hybrid.txt` | Runtime dependencies |
| `setup.bat`, `setup.sh` | Basic dependency install helpers |
| `model.safetensors` | Base local GLiNER2 model weights |
| `model-finetuned/final/` | Accepted fine-tuned checkpoint to use by default |
| `model-finetuned/_trainer/final/` | LoRA adapter artifact retained for provenance/reuse |
| `config.json`, `tokenizer*.json`, `added_tokens.json`, `special_tokens_map.json`, `encoder_config/` | Base model tokenizer/config files |

## Notes
- **Offline:** loads with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. Copy this folder to an air-gapped machine and it runs unchanged.
- **Device:** automatic CUDA → CPU. On CUDA it uses fp16 (~0.4 GB VRAM — fits a 4 GB card). If you hit out-of-memory, lower `batch_size`.
- **Tuning:** `threshold` (default `0.7` — best precision/recall balance; use `0.5` for max recall, `0.85+` for fewest false flags) and the GIRP element label sets in `girp.py` are editable.
- **Robust:** format validation removes false positives (a "card" needs 13–19 digits, an "address" a number/street word, pronouns aren't names); a Luhn-checked regex backstop catches structured PII the model misses (cards, emails, international phones); CUDA out-of-memory self-recovers (batch halving → CPU fallback).

---
Model: GLiNER2 (Apache-2.0), run entirely locally by this repository.
