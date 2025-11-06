@echo off
REM Helper script when model is already copied from another environment
REM Just verifies the model exists in the correct location

echo ========================================
echo Model Location Verification
echo ========================================
echo.

echo Expected model location:
echo models\models--Qwen--Qwen2.5-3B-Instruct\
echo.

if exist "models\models--Qwen--Qwen2.5-3B-Instruct\" (
    echo [OK] Model directory found!
    echo.
    echo Model files:
    dir /b "models\models--Qwen--Qwen2.5-3B-Instruct\snapshots\*"
    echo.
    echo ========================================
    echo Model is ready to use!
    echo ========================================
    echo.
    echo Next steps:
    echo   1. Run baseline benchmarks: python scripts\run_benchmarks.py
    echo   2. Start building introspection APIs
    echo.
) else (
    echo [ERROR] Model directory not found!
    echo.
    echo Please copy the model directory from your original environment:
    echo   Source: [original]\models\models--Qwen--Qwen2.5-3B-Instruct\
    echo   Target: D:\agi\agi-self-modification-research\models\
    echo.
)

pause
