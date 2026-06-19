param(
  [switch]$Install
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Commit = "pii-model-offline-2026-06-19"
$BaseUrl = "https://github.com/Glass-Half-Full/PII-Model/raw/$Commit"

$RuntimeFiles = @(
  "README.md",
  "AUPII.md",
  "requirements.txt",
  "requirements-hybrid.txt",
  "setup.bat",
  "pii_model/__init__.py",
  "pii_model/girp.py",
  "pii_model/aupii.py",
  "models/base/added_tokens.json",
  "models/base/config.json",
  "models/base/encoder_config/config.json",
  "models/base/special_tokens_map.json",
  "models/base/tokenizer.json",
  "models/base/tokenizer_config.json",
  "models/base/model.safetensors",
  "models/fine-tuned/added_tokens.json",
  "models/fine-tuned/config.json",
  "models/fine-tuned/encoder_config/config.json",
  "models/fine-tuned/special_tokens_map.json",
  "models/fine-tuned/tokenizer.json",
  "models/fine-tuned/tokenizer_config.json",
  "models/fine-tuned/model.safetensors",
  "models/lora-adapter/README.md",
  "models/lora-adapter/adapter_config.json",
  "models/lora-adapter/adapter_model.safetensors",
  "models/lora-adapter/training_config.json"
)

$WheelFiles = @(
  "accelerate-1.14.0-py3-none-any.whl",
  "annotated_doc-0.0.4-py3-none-any.whl",
  "annotated_types-0.7.0-py3-none-any.whl",
  "anyio-4.14.0-py3-none-any.whl",
  "blis-1.3.3-cp312-cp312-win_amd64.whl",
  "catalogue-2.0.10-py3-none-any.whl",
  "certifi-2026.6.17-py3-none-any.whl",
  "charset_normalizer-3.4.7-cp312-cp312-win_amd64.whl",
  "click-8.4.1-py3-none-any.whl",
  "cloudpathlib-0.24.0-py3-none-any.whl",
  "confection-1.3.3-py3-none-any.whl",
  "cymem-2.0.13-cp312-cp312-win_amd64.whl",
  "filelock-3.29.4-py3-none-any.whl",
  "flatbuffers-25.12.19-py2.py3-none-any.whl",
  "fsspec-2026.6.0-py3-none-any.whl",
  "gliner-0.2.27-py3-none-any.whl",
  "gliner2-1.3.1-py3-none-any.whl",
  "h11-0.16.0-py3-none-any.whl",
  "hf_xet-1.5.1-cp37-abi3-win_amd64.whl",
  "httpcore-1.0.9-py3-none-any.whl",
  "httpx-0.28.1-py3-none-any.whl",
  "huggingface_hub-1.20.1-py3-none-any.whl",
  "idna-3.18-py3-none-any.whl",
  "jinja2-3.1.6-py3-none-any.whl",
  "markdown_it_py-4.2.0-py3-none-any.whl",
  "markupsafe-3.0.3-cp312-cp312-win_amd64.whl",
  "mdurl-0.1.2-py3-none-any.whl",
  "mpmath-1.3.0-py3-none-any.whl",
  "murmurhash-1.0.15-cp312-cp312-win_amd64.whl",
  "networkx-3.6.1-py3-none-any.whl",
  "numpy-2.4.6-cp312-cp312-win_amd64.whl",
  "onnxruntime-1.27.0-cp312-cp312-win_amd64.whl",
  "packaging-26.2-py3-none-any.whl",
  "pandas-3.0.3-cp312-cp312-win_amd64.whl",
  "peft-0.19.1-py3-none-any.whl",
  "phonenumbers-9.0.32-py2.py3-none-any.whl",
  "preshed-3.0.13-cp312-cp312-win_amd64.whl",
  "presidio_analyzer-2.2.362-py3-none-any.whl",
  "protobuf-7.35.1-cp310-abi3-win_amd64.whl",
  "psutil-7.2.2-cp37-abi3-win_amd64.whl",
  "pydantic-2.13.4-py3-none-any.whl",
  "pydantic_core-2.46.4-cp312-cp312-win_amd64.whl",
  "pygments-2.20.0-py3-none-any.whl",
  "python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
  "pyyaml-6.0.3-cp312-cp312-win_amd64.whl",
  "regex-2026.5.9-cp312-cp312-win_amd64.whl",
  "requests-2.34.2-py3-none-any.whl",
  "requests_file-3.0.1-py2.py3-none-any.whl",
  "rich-15.0.0-py3-none-any.whl",
  "safetensors-0.8.0-cp310-abi3-win_amd64.whl",
  "sentencepiece-0.2.1-cp312-cp312-win_amd64.whl",
  "setuptools-81.0.0-py3-none-any.whl",
  "shellingham-1.5.4-py2.py3-none-any.whl",
  "six-1.17.0-py2.py3-none-any.whl",
  "smart_open-7.6.1-py3-none-any.whl",
  "spacy-3.8.14-cp312-cp312-win_amd64.whl",
  "spacy_legacy-3.0.12-py2.py3-none-any.whl",
  "spacy_loggers-1.0.5-py3-none-any.whl",
  "srsly-2.5.3-cp312-cp312-win_amd64.whl",
  "sympy-1.14.0-py3-none-any.whl",
  "thinc-8.3.13-cp312-cp312-win_amd64.whl",
  "tldextract-5.3.1-py3-none-any.whl",
  "tokenizers-0.22.2-cp39-abi3-win_amd64.whl",
  "torch-2.12.1-cp312-cp312-win_amd64.whl",
  "tqdm-4.68.3-py3-none-any.whl",
  "transformers-5.6.2-py3-none-any.whl",
  "typer-0.25.1-py3-none-any.whl",
  "typing_extensions-4.15.0-py3-none-any.whl",
  "typing_inspection-0.4.2-py3-none-any.whl",
  "urllib3-2.7.0-py3-none-any.whl",
  "wasabi-1.1.3-py3-none-any.whl",
  "weasel-1.0.0-py3-none-any.whl",
  "wrapt-2.2.1-cp312-cp312-win_amd64.whl"
)

function Get-GitHubFile {
  param([string]$Path)

  $UrlPath = $Path -replace "\\", "/"
  $Url = "$BaseUrl/$UrlPath"
  $OutPath = Join-Path (Get-Location) ($Path -replace "/", [IO.Path]::DirectorySeparatorChar)
  $OutDir = Split-Path $OutPath
  if ($OutDir) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
  }
  Write-Host "Downloading $Path"
  Invoke-WebRequest -Uri $Url -OutFile $OutPath
}

foreach ($Path in $RuntimeFiles) {
  Get-GitHubFile $Path
}

foreach ($Wheel in $WheelFiles) {
  Get-GitHubFile "wheelhouse/$Wheel"
}

$ExpectedSizes = @{
  "models/base/model.safetensors" = 833938108
  "models/fine-tuned/model.safetensors" = 833938108
  "models/lora-adapter/adapter_model.safetensors" = 5329152
}

$ExpectedHashes = @{
  "models/base/model.safetensors" = "845fc4bd93c525b86124c58ab4f56c9eacf8587953086b14c501fab25957c007"
  "models/fine-tuned/model.safetensors" = "1ff2a86d7470057cc200f94f1c7fd078c2ace437065a8c14c77d1b80a345fa92"
  "models/lora-adapter/adapter_model.safetensors" = "eca4d810c9480a59a621d11ba2d5ab56a409cb349f9fc03e3bc9c9012355b73c"
}

foreach ($Path in $ExpectedSizes.Keys) {
  $File = Get-Item $Path
  if ($File.Length -ne $ExpectedSizes[$Path]) {
    throw "Unexpected size for $Path. Got $($File.Length), expected $($ExpectedSizes[$Path])."
  }
  $Hash = (Get-FileHash $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($Hash -ne $ExpectedHashes[$Path]) {
    throw "Unexpected SHA256 for $Path. Got $Hash, expected $($ExpectedHashes[$Path])."
  }
}

Write-Host ""
Write-Host "Download complete. Install with:"
Write-Host "  py -3.12 -m pip install --no-index --find-links wheelhouse -r requirements.txt -r requirements-hybrid.txt"
Write-Host ""
Write-Host "Then run the DataFrame example in README.md."

if ($Install) {
  py -3.12 -m pip install --no-index --find-links wheelhouse -r requirements.txt -r requirements-hybrid.txt
}
