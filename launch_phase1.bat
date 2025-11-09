@echo off
REM Launch Phase 1 Introspection
REM Activates venv and sets PYTHONPATH

echo ================================================================================
echo PHASE 1: READ-ONLY INTROSPECTION - LAUNCH SCRIPT
echo ================================================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Set PYTHONPATH to include src/
set PYTHONPATH=%CD%

REM Run Phase 1 - Default is Phase 1a (no heritage baseline)
python scripts\experiments\phase1a_no_heritage.py

REM Deactivate venv
deactivate
