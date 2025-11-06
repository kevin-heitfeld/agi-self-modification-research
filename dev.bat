@echo off
REM Development helper script with common commands
REM Run as: dev.bat [command]

if "%1"=="" goto help

if /i "%1"=="test" goto test
if /i "%1"=="test-cov" goto test_cov
if /i "%1"=="format" goto format
if /i "%1"=="lint" goto lint
if /i "%1"=="type-check" goto type_check
if /i "%1"=="notebook" goto notebook
if /i "%1"=="lab" goto lab
if /i "%1"=="verify" goto verify
if /i "%1"=="clean" goto clean

echo Unknown command: %1
goto help

:help
echo ========================================
echo AGI Self-Modification Research - Dev Tools
echo ========================================
echo.
echo Usage: dev.bat [command]
echo.
echo Commands:
echo   test         - Run all tests
echo   test-cov     - Run tests with coverage report
echo   format       - Format code with black and isort
echo   lint         - Check code with flake8
echo   type-check   - Check types with mypy
echo   notebook     - Start Jupyter notebook
echo   lab          - Start JupyterLab
echo   verify       - Verify installation
echo   clean        - Clean cache files (safe)
echo.
echo Examples:
echo   dev.bat test
echo   dev.bat format
echo   dev.bat notebook
goto end

:test
echo Running tests...
call venv\Scripts\activate.bat
pytest tests/ -v
goto end

:test_cov
echo Running tests with coverage...
call venv\Scripts\activate.bat
pytest tests/ -v --cov=src --cov-report=html --cov-report=term
echo.
echo Coverage report generated: htmlcov\index.html
goto end

:format
echo Formatting code...
call venv\Scripts\activate.bat
echo Running black...
black src/ tests/
echo Running isort...
isort src/ tests/
echo Code formatted!
goto end

:lint
echo Checking code style...
call venv\Scripts\activate.bat
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
goto end

:type_check
echo Checking types...
call venv\Scripts\activate.bat
mypy src/ --ignore-missing-imports
goto end

:notebook
echo Starting Jupyter notebook...
call venv\Scripts\activate.bat
jupyter notebook
goto end

:lab
echo Starting JupyterLab...
call venv\Scripts\activate.bat
jupyter lab
goto end

:verify
echo Verifying installation...
call venv\Scripts\activate.bat
python verify_installation.py
goto end

:clean
echo Cleaning cache files...
echo Removing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
echo Removing pytest cache...
if exist .pytest_cache rmdir /s /q .pytest_cache
echo Removing coverage files...
if exist .coverage del /q .coverage
if exist htmlcov rmdir /s /q htmlcov
echo Removing Jupyter checkpoints...
for /d /r . %%d in (.ipynb_checkpoints) do @if exist "%%d" rd /s /q "%%d"
echo Cache cleaned!
goto end

:end
