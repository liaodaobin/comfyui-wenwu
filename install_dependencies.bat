@echo off
setlocal
chcp 65001 >nul

echo ============================================================
echo comfyui-wenwu dependency installer
echo ============================================================
echo.

set "NODE_DIR=%~dp0"
set "COMFY_DIR=%NODE_DIR%..\.."
set "ROOT_DIR=%NODE_DIR%..\..\.."
set "PYTHON_EXE="

if exist "%ROOT_DIR%\python\python.exe" set "PYTHON_EXE=%ROOT_DIR%\python\python.exe"
if not defined PYTHON_EXE if exist "%ROOT_DIR%\python_embeded\python.exe" set "PYTHON_EXE=%ROOT_DIR%\python_embeded\python.exe"
if not defined PYTHON_EXE if exist "%COMFY_DIR%\python_embeded\python.exe" set "PYTHON_EXE=%COMFY_DIR%\python_embeded\python.exe"
if not defined PYTHON_EXE for %%P in (python.exe) do set "PYTHON_EXE=%%~$PATH:P"

if not defined PYTHON_EXE (
    echo [ERROR] Python was not found.
    echo Please run this file from a ComfyUI package that contains python\python.exe,
    echo or install dependencies manually with:
    echo python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    pause
    exit /b 1
)

echo [INFO] Python: %PYTHON_EXE%
echo [INFO] Installing Python packages from requirements.txt ...
"%PYTHON_EXE%" -m pip install -r "%NODE_DIR%requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo Try another mirror manually, for example:
    echo "%PYTHON_EXE%" -m pip install -r "%NODE_DIR%requirements.txt" -i https://mirrors.aliyun.com/pypi/simple
    pause
    exit /b 1
)

echo.
if exist "%COMFY_DIR%\custom_nodes\ComfyUI-llama-cpp\nodes.py" (
    echo [OK] ComfyUI-llama-cpp was found.
) else (
    echo [WARN] ComfyUI-llama-cpp was NOT found.
    echo WenWuPromptGenerator needs this folder:
    echo %COMFY_DIR%\custom_nodes\ComfyUI-llama-cpp
)

echo.
echo [DONE] Please restart ComfyUI.
pause
