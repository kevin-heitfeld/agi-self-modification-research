@echo off
REM Update existing virtual environment with new dependencies
REM This script updates packages without recreating the venv

echo ========================================
echo AGI Self-Modification - Update venv
echo ========================================
echo.
echo This will update your existing virtual environment with:
echo - Latest PyTorch 2.5.1+cu121
echo - Latest Transformers 4.57.1+
echo - HQQ quantization library
echo.
echo Note: Make sure you have at least 500 MB free on C: drive
echo.

REM Check if venv exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to create the environment.
    pause
    exit /b 1
)

REM Don't rely on activation - use venv Python directly
echo [1/5] Checking virtual environment...
if not exist venv\Scripts\python.exe (
    echo ERROR: Virtual environment Python not found!
    echo Expected: venv\Scripts\python.exe
    pause
    exit /b 1
)
echo Virtual environment found
echo Using: %CD%\venv\Scripts\python.exe
echo.

REM Set temp directories and pip cache to D: drive
echo [2/5] Setting temporary directories to D:\temp...
if not exist D:\temp mkdir D:\temp
set TMPDIR=D:\temp
set TEMP=D:\temp
set TMP=D:\temp
set PIP_CACHE_DIR=D:\temp\pip-cache
echo Temp directories: D:\temp
echo Pip cache: D:\temp\pip-cache
echo.

REM Upgrade pip
echo [3/5] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip
echo.

REM Install/upgrade PyTorch 2.5.1+cu121
echo [4/5] Installing PyTorch 2.5.1+cu121 (this may take several minutes)...
echo Note: Using D:\temp for downloads and NO cache to avoid C: drive issues
venv\Scripts\python.exe -m pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    echo Please check your internet connection and try again
    pause
    exit /b 1
)
echo PyTorch installed successfully
echo.

REM Install/upgrade remaining requirements including hqq
echo [5/5] Installing/upgrading all requirements (this may take several minutes)...
echo This includes:
echo - Transformers 4.57.1+
echo - HQQ quantization library
echo - All other dependencies from requirements.txt
echo.
venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt --upgrade

if %errorlevel% neq 0 (
    echo WARNING: Some packages may have failed to install
    echo Check the output above for details
    echo.
) else (
    echo.
    echo ========================================
    echo Update completed successfully!
    echo ========================================
    echo.
)

REM Verify key packages
echo Verifying key packages:
venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}')"
venv\Scripts\python.exe -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo.
echo Checking HQQ quantization support:
venv\Scripts\python.exe -c "try:
    from transformers.cache_utils import QuantizedCache
    print('✓ HQQ Quantized Cache: Available (new API - 75%% memory savings)')
except ImportError:
    try:
        from transformers.cache_utils import HQQQuantizedCache
        print('✓ HQQ Quantized Cache: Available (deprecated API)')
    except ImportError:
        print('✗ HQQ Quantized Cache: Not available')
"

echo.
echo Update complete! You can now run your experiments.
echo.
echo Next steps:
echo   1. Test the installation: python verify_installation.py
echo   2. Run Phase 1a experiment on Colab or locally
echo.

pause
