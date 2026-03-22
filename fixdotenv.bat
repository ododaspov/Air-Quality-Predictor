@echo off
echo ============================================
echo  Deep-fix: dotenv conflict cleaner
echo ============================================
echo.

cd /d "%~dp0"

call venv\Scripts\activate.bat

echo [1/5] Uninstalling all dotenv variants...
pip uninstall dotenv -y 2>nul
pip uninstall python-dotenv -y 2>nul
pip uninstall dotenv-python -y 2>nul

echo [2/5] Manually deleting stale dotenv folders from venv...
set SITE=venv\lib\site-packages

if exist "%SITE%\dotenv" (
    rmdir /s /q "%SITE%\dotenv"
    echo    Deleted: %SITE%\dotenv\
)
if exist "%SITE%\dotenv-0*" (
    rmdir /s /q "%SITE%\dotenv-0*"
)
for /d %%i in ("%SITE%\python_dotenv*") do (
    rmdir /s /q "%%i"
    echo    Deleted: %%i
)
for /d %%i in ("%SITE%\dotenv*") do (
    rmdir /s /q "%%i"
    echo    Deleted: %%i
)
for %%i in ("%SITE%\dotenv*") do (
    del /q "%%i"
    echo    Deleted: %%i
)

echo [3/5] Clearing pip cache...
pip cache purge

echo [4/5] Reinstalling python-dotenv fresh...
pip install python-dotenv --no-cache-dir

echo [5/5] Verifying...
python -c "from dotenv import load_dotenv; print('SUCCESS - load_dotenv is working!')"

echo.
echo ============================================
echo  Done! Restart your Jupyter kernel now.
echo  Kernel menu: Kernel -> Restart Kernel
echo ============================================
pause