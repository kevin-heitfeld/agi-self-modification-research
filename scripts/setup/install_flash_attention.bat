@echo off
REM Install Flash Attention 2 for memory optimization
REM This script handles the special installation requirements

echo ================================================================
echo Installing Flash Attention 2
echo ================================================================
echo.
echo Flash Attention 2 provides:
echo - Memory-efficient attention (O(n) instead of O(n^2))
echo - 2-4x faster generation
echo - 1-2 GB memory savings during inference
echo.
echo Requirements:
echo - CUDA 11.6 or higher
echo - Compatible GPU (T4, A100, etc.)
echo - ~10 minutes compilation time
echo.

REM Check if in virtual environment
if not defined VIRTUAL_ENV (
    echo ERROR: Virtual environment not activated!
    echo Please run: activate.bat
    pause
    exit /b 1
)

REM Set temp directories and pip cache to D: drive
echo Setting temporary directories to D:\temp to avoid disk space issues...
if not exist D:\temp mkdir D:\temp
set TMPDIR=D:\temp
set TEMP=D:\temp
set TMP=D:\temp
set PIP_CACHE_DIR=D:\temp\pip-cache
echo.

echo [1/3] Installing build dependencies...
pip install ninja packaging wheel

echo.
echo [2/3] Installing Flash Attention 2 (this may take 5-10 minutes)...
echo Note: The build process will compile CUDA kernels
pip install flash-attn --no-build-isolation

if errorlevel 1 (
    echo.
    echo ================================================================
    echo WARNING: Flash Attention installation failed
    echo ================================================================
    echo.
    echo This is OK - the system will automatically fallback to eager attention.
    echo.
    echo Common causes:
    echo - CUDA version too old (need 11.6+^)
    echo - Incompatible GPU
    echo - Missing build tools
    echo.
    echo The experiments will still work, just without Flash Attention optimization.
    echo.
    pause
    exit /b 0
)

echo.
echo [3/3] Verifying installation...
python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"

if errorlevel 1 (
    echo WARNING: Flash Attention import failed
    echo System will fallback to eager attention
) else (
    echo.
    echo ================================================================
    echo SUCCESS: Flash Attention 2 installed!
    echo ================================================================
    echo.
    echo Your experiments will now use:
    echo - Flash Attention 2 for memory-efficient attention
    echo - INT8 KV cache quantization for 50%% cache memory savings
    echo.
    echo Expected memory savings: 4-6 GB during generation
    echo Expected speedup: 2-4x faster
    echo.
)

pause
