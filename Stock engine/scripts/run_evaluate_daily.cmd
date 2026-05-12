@echo off
setlocal

set "PROJECT_DIR=%~dp0..\\"
set "PYTHON_EXE=%PROJECT_DIR%market_env\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  set "PYTHON_EXE=python"
)

if not exist "%PROJECT_DIR%logs" (
  mkdir "%PROJECT_DIR%logs"
)

cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" "%PROJECT_DIR%app.py" evaluate >> "%PROJECT_DIR%logs\evaluate.log" 2>&1

endlocal
