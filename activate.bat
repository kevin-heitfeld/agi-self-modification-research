@echo off
REM Quick activation script for AGI Self-Modification Research environment
REM Run as: activate.bat

call venv\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo ERROR: Could not activate virtual environment
    echo Have you run setup.bat yet?
    pause
    exit /b 1
)

REM Set temp directories and pip cache to D: drive to avoid C: drive space issues
if not exist D:\temp mkdir D:\temp
set TMPDIR=D:\temp
set TEMP=D:\temp
set TMP=D:\temp
set PIP_CACHE_DIR=D:\temp\pip-cache

echo.
echo ========================================
echo AGI Self-Modification Research
echo Virtual Environment Activated
echo ========================================
echo.
echo Python: 
python --version
echo.
echo Quick commands:
echo   jupyter notebook           - Start Jupyter notebooks
echo   jupyter lab                - Start JupyterLab (alternative)
echo   pytest                     - Run all tests
echo   pytest tests/              - Run tests in tests/ directory
echo   python verify_installation.py - Verify installation
echo   python -m src.experiments.baseline - Run baseline experiments
echo.
echo To deactivate: deactivate
echo.
