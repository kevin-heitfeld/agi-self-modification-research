@echo off
REM AGI Self-Modification Research - Installation Script for Windows
REM This script sets up the complete development environment
REM Run as: setup.bat

echo ========================================
echo AGI Self-Modification Research Setup
echo ========================================
echo.

REM Check Python version
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo.

REM Check if Python version is appropriate
python -c "import sys; exit(0 if sys.version_info >= (3, 10) and sys.version_info < (3, 12) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Python 3.10 or 3.11 is recommended
    echo You are using a different version - proceed with caution
    echo.
    pause
)

REM Check NVIDIA GPU and CUDA
echo [2/8] Checking NVIDIA GPU and CUDA...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: nvidia-smi not found - GPU may not be available
    echo This project requires a CUDA-capable NVIDIA GPU
    echo.
    set CUDA_VERSION=cpu
) else (
    echo GPU detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo.
    echo CUDA version:
    nvidia-smi | findstr "CUDA Version"
    echo.
    echo Please confirm your CUDA version:
    echo   [1] CUDA 11.8 (cu118)
    echo   [2] CUDA 12.4 (cu124) - Recommended (works with CUDA 12.x and 13.x)
    echo   [3] CUDA 12.1 (cu121)
    echo   [4] CPU only (not recommended)
    echo.
    set /p cuda_choice="Enter choice [1-4]: "

    if "%cuda_choice%"=="1" set CUDA_VERSION=cu118
    if "%cuda_choice%"=="2" set CUDA_VERSION=cu124
    if "%cuda_choice%"=="3" set CUDA_VERSION=cu121
    if "%cuda_choice%"=="4" set CUDA_VERSION=cpu

    if not defined CUDA_VERSION (
        echo Invalid choice, using cu124 (CUDA 12.4)
        set CUDA_VERSION=cu124
    )
)

echo Using CUDA version: %CUDA_VERSION%
echo.

REM Create virtual environment
echo [3/8] Creating virtual environment...
if exist venv (
    echo WARNING: venv directory already exists
    set /p overwrite="Delete and recreate? [y/N]: "
    if /i "%overwrite%"=="y" (
        echo Removing existing venv...
        rmdir /s /q venv
    ) else (
        echo Keeping existing venv and continuing...
        goto activate_venv
    )
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created successfully
echo.

:activate_venv
REM Activate virtual environment
echo [4/8] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated
echo.

REM Set temp directories and pip cache to D: drive to avoid C: drive space issues
echo Setting temporary directories and pip cache to D:\temp to avoid disk space issues...
if not exist D:\temp mkdir D:\temp
set TMPDIR=D:\temp
set TEMP=D:\temp
set TMP=D:\temp
set PIP_CACHE_DIR=D:\temp\pip-cache
echo Temp directories: D:\temp
echo Pip cache: D:\temp\pip-cache
echo.

REM Upgrade pip
echo [5/8] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch with appropriate CUDA version
echo [6/8] Installing PyTorch (this may take several minutes^)...
if "%CUDA_VERSION%"=="cpu" (
    echo Installing CPU-only version (not recommended for this project^)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else if "%CUDA_VERSION%"=="cu121" (
    echo Installing PyTorch 2.5.1 with CUDA 12.1...
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
) else if "%CUDA_VERSION%"=="cu124" (
    echo Installing PyTorch with CUDA 12.4...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
) else (
    echo Installing PyTorch with CUDA %CUDA_VERSION%...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CUDA_VERSION%
)

if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    echo Please check your internet connection and try again
    pause
    exit /b 1
)
echo PyTorch installed successfully
echo.

REM Install remaining requirements
echo [7/8] Installing remaining packages (this may take several minutes^)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    echo Check requirements.txt and try again
    pause
    exit /b 1
)
echo All packages installed successfully
echo.

REM Verify installation
echo [8/8] Verifying installation...
python verify_installation.py
if %errorlevel% neq 0 (
    echo WARNING: Some verification checks failed
    echo Review the output above for details
    echo.
) else (
    echo.
    echo ========================================
    echo Installation completed successfully!
    echo ========================================
    echo.
)

echo Next steps:
echo   1. Review the verification output above
echo   2. Activate the environment: venv\Scripts\activate
echo   3. Start Jupyter: jupyter notebook
echo   4. Or run tests: pytest tests/
echo.

REM Create activation helper script
echo @echo off > activate.bat
echo call venv\Scripts\activate.bat >> activate.bat
echo echo Virtual environment activated >> activate.bat
echo echo. >> activate.bat
echo echo Quick commands: >> activate.bat
echo echo   jupyter notebook  - Start Jupyter >> activate.bat
echo echo   pytest            - Run tests >> activate.bat
echo echo   python verify_installation.py - Verify setup >> activate.bat

echo Helper script created: activate.bat
echo Run 'activate.bat' to quickly activate the environment
echo.

pause
