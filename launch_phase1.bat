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

REM Run Phase 1
python scripts\experiments\phase1_introspection.py

REM Deactivate venv
deactivate
