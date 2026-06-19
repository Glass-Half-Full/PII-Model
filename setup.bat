@echo off
REM Setup for the local PII & GIRP classifier (Windows, Python 3.13).
REM No venv/conda - installs into your current Python via pip / your internal mirror.
REM Usage:  setup.bat        (or set PYTHON=py -3.13 first)

if "%PYTHON%"=="" set PYTHON=python
%PYTHON% --version

%PYTHON% -m pip install -r requirements.txt

echo.
echo Done. See README.md for the DataFrame and CSV examples.
echo.
echo NVIDIA GPU: ensure a CUDA build of torch is installed from your mirror; otherwise CPU is used.
echo The model itself loads from local files - no internet required at run time.
