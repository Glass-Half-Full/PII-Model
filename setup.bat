@echo off
REM Setup for the local PII & GIRP notebook (Windows, Python 3.13).
REM No venv/conda - installs into your current Python via pip / your internal mirror.
REM Usage:  setup.bat        (or set PYTHON=py -3.13 first)

if "%PYTHON%"=="" set PYTHON=python
%PYTHON% --version

%PYTHON% -m pip install -r requirements.txt

echo.
echo Done. Launch the notebook with:
echo   %PYTHON% -m jupyter lab gliner2_pii_demo.ipynb
echo.
echo NVIDIA GPU: ensure a CUDA build of torch is installed from your mirror; otherwise CPU is used.
echo The model itself loads from local files - no internet required at run time.
