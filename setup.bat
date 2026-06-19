@echo off
REM Setup for the local PII & GIRP classifier (Windows, Python 3.12).
REM Usage:  setup.bat        (or set PYTHON=py -3.12 first)

if "%PYTHON%"=="" set PYTHON=python
%PYTHON% --version

if exist wheelhouse\*.whl (
  echo Installing from local wheelhouse only...
  %PYTHON% -m pip install --no-index --find-links wheelhouse -r requirements.txt -r requirements-hybrid.txt
) else (
  echo Installing from package indexes because wheelhouse was not found...
  %PYTHON% -m pip install -r requirements.txt -r requirements-hybrid.txt
)

echo.
echo Done. See README.md for the DataFrame and CSV examples.
echo.
echo NVIDIA GPU: ensure a CUDA build of torch is installed from your mirror; otherwise CPU is used.
echo No spaCy language model download is required. The model itself loads from local files.
