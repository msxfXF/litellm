@echo off
setlocal enabledelayedexpansion

REM One-click startup for LiteLLM Proxy (Windows).
REM Uses the repo-local config in this directory.

cd /d "%~dp0"

set "CONFIG=%~dp0proxy_iflow_config.yaml"
if "%LITELLM_CONFIG%" NEQ "" set "CONFIG=%LITELLM_CONFIG%"

set "HOST=127.0.0.1"
if "%LITELLM_HOST%" NEQ "" set "HOST=%LITELLM_HOST%"

set "PORT=4000"
if "%LITELLM_PORT%" NEQ "" set "PORT=%LITELLM_PORT%"

REM Resolve litellm.exe from the active Python install (most reliable on Windows).
for /f "usebackq delims=" %%I in (`python -c "import sysconfig, os; print(os.path.join(sysconfig.get_path('scripts'), 'litellm.exe'))"`) do set "LITELLM_EXE=%%I"

if exist "%LITELLM_EXE%" (
  echo Starting LiteLLM Proxy: "%LITELLM_EXE%" --config "%CONFIG%" --host "%HOST%" --port "%PORT%"
  "%LITELLM_EXE%" --config "%CONFIG%" --host "%HOST%" --port "%PORT%"
) else (
  echo Starting LiteLLM Proxy: litellm --config "%CONFIG%" --host "%HOST%" --port "%PORT%"
  litellm --config "%CONFIG%" --host "%HOST%" --port "%PORT%"
)

if errorlevel 1 (
  echo.
  echo LiteLLM exited with errorlevel %errorlevel%.
  pause
)

