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

REM Check if venv exists
if not exist venv (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to create the environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/5] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated
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
python -m pip install --upgrade pip
echo.

REM Install/upgrade PyTorch 2.5.1+cu121
echo [4/5] Installing PyTorch 2.5.1+cu121 (this may take several minutes)...
echo Note: Using D:\temp for downloads to avoid C: drive space issues
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

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
python -m pip install -r requirements.txt --upgrade

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
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo.
echo Checking HQQ quantization support:
python -c "try:
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
