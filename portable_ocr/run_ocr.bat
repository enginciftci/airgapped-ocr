@echo off
setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "DO_NOT_TRACK=1"
set "ANONYMIZED_TELEMETRY=False"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PATH=%ROOT_DIR%\runtime\python;%ROOT_DIR%\runtime\python\DLLs;%ROOT_DIR%\runtime\llama;%SystemRoot%\system32;%SystemRoot%"

"%ROOT_DIR%\runtime\python\python.exe" "%ROOT_DIR%\src\cli.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
