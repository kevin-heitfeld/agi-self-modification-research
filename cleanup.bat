@echo off
REM Cleanup script for AGI Self-Modification Research
REM This script removes the virtual environment and cached files
REM Run as: cleanup.bat

echo ========================================
echo AGI Self-Modification Research Cleanup
echo ========================================
echo.
echo This will remove:
echo   - Virtual environment (venv/)
echo   - Python cache files (__pycache__, *.pyc)
echo   - Pytest cache (.pytest_cache/)
echo   - Coverage reports (.coverage, htmlcov/)
echo   - Jupyter checkpoints (.ipynb_checkpoints/)
echo.
echo This will NOT remove:
echo   - Source code
echo   - Documentation
echo   - Model checkpoints
echo   - Experimental data
echo.

set /p confirm="Are you sure? [y/N]: "
if /i not "%confirm%"=="y" (
    echo Cleanup cancelled
    pause
    exit /b 0
)

echo.
echo Cleaning up...

REM Remove virtual environment
if exist venv (
    echo Removing virtual environment...
    rmdir /s /q venv
    echo   venv/ removed
)

REM Remove Python cache
echo Removing Python cache files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
echo   Python cache removed

REM Remove pytest cache
if exist .pytest_cache (
    echo Removing pytest cache...
    rmdir /s /q .pytest_cache
    echo   .pytest_cache/ removed
)

REM Remove coverage reports
if exist .coverage (
    echo Removing coverage reports...
    del /q .coverage
    echo   .coverage removed
)
if exist htmlcov (
    rmdir /s /q htmlcov
    echo   htmlcov/ removed
)

REM Remove Jupyter checkpoints
echo Removing Jupyter checkpoints...
for /d /r . %%d in (.ipynb_checkpoints) do @if exist "%%d" rd /s /q "%%d"
echo   .ipynb_checkpoints/ removed

REM Remove mypy cache
if exist .mypy_cache (
    echo Removing mypy cache...
    rmdir /s /q .mypy_cache
    echo   .mypy_cache/ removed
)

echo.
echo ========================================
echo Cleanup complete!
echo ========================================
echo.
echo To reinstall the environment:
echo   setup.bat
echo.
pause
