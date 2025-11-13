@echo off
REM Quick C: drive cleanup helper
REM This script helps identify and clean up space on C: drive

echo ========================================
echo C: Drive Cleanup Helper
echo ========================================
echo.

echo Current C: drive space:
for /f "tokens=3" %%a in ('dir C:\ ^| find "bytes free"') do set C_FREE=%%a
echo Free: %C_FREE% bytes (%.2f MB)
echo.

echo Common locations that can be cleaned:
echo.

echo 1. Python pip cache: C:\Users\%USERNAME%\AppData\Local\pip\cache
if exist "C:\Users\%USERNAME%\AppData\Local\pip\cache" (
    for /f "tokens=3" %%a in ('dir "C:\Users\%USERNAME%\AppData\Local\pip\cache" /s ^| find "bytes"') do set PIP_SIZE=%%a
    echo    Size: %PIP_SIZE% bytes
    set /p clean_pip="   Clean pip cache? [y/N]: "
    if /i "!clean_pip!"=="y" (
        echo    Cleaning pip cache...
        rmdir /s /q "C:\Users\%USERNAME%\AppData\Local\pip\cache"
        mkdir "C:\Users\%USERNAME%\AppData\Local\pip\cache"
        echo    Done!
    )
) else (
    echo    Not found or already clean
)
echo.

echo 2. Temp files: C:\Users\%USERNAME%\AppData\Local\Temp
if exist "C:\Users\%USERNAME%\AppData\Local\Temp" (
    for /f "tokens=3" %%a in ('dir "C:\Users\%USERNAME%\AppData\Local\Temp" /s ^| find "bytes"') do set TEMP_SIZE=%%a
    echo    Size: %TEMP_SIZE% bytes
    set /p clean_temp="   Clean temp files? [y/N]: "
    if /i "!clean_temp!"=="y" (
        echo    Cleaning temp files...
        del /q /s "C:\Users\%USERNAME%\AppData\Local\Temp\*.*" 2>nul
        echo    Done!
    )
) else (
    echo    Not found
)
echo.

echo 3. Windows Temp: C:\Windows\Temp (requires admin)
echo    Run: cleanmgr.exe (Disk Cleanup tool)
echo.

echo 4. VS Code extensions cache
if exist "C:\Users\%USERNAME%\.vscode\extensions" (
    for /f "tokens=3" %%a in ('dir "C:\Users\%USERNAME%\.vscode\extensions" /s ^| find "bytes"') do set VSCODE_SIZE=%%a
    echo    Size: %VSCODE_SIZE% bytes
    echo    Location: C:\Users\%USERNAME%\.vscode\extensions
    echo    Manually review and delete unused extensions
)
echo.

echo 5. npm cache (if Node.js installed)
if exist "C:\Users\%USERNAME%\AppData\Roaming\npm-cache" (
    echo    Run: npm cache clean --force
)
echo.

echo After cleanup, free space:
for /f "tokens=3" %%a in ('dir C:\ ^| find "bytes free"') do set C_FREE_AFTER=%%a
echo Free: %C_FREE_AFTER% bytes
echo.

echo ========================================
echo ALTERNATIVE: Use Colab Instead!
echo ========================================
echo.
echo Your code is already on GitHub. You can:
echo 1. Open notebooks/Phase1_Colab.ipynb on GitHub
echo 2. Click "Open in Colab" badge at the top
echo 3. Run all cells - everything installs automatically
echo 4. No local disk space needed!
echo.

pause
