@echo off
setlocal
REM Script para ejecutar la interfaz grafica usando el entorno virtual del proyecto si existe

set SCRIPT_DIR=%~dp0
set VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe

if exist "%VENV_PYTHON%" (
    set PYTHON_CMD="%VENV_PYTHON%"
) else (
    echo Verificando Python...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: Python no esta instalado o no se encuentra en el PATH
        echo.
        echo SOLUCION: Ejecuta primero install.bat
        echo.
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
)

echo.
echo Iniciando interfaz grafica...
cd /d "%SCRIPT_DIR%src"
%PYTHON_CMD% color_classification_gui.py
if errorlevel 1 (
    echo.
    echo ERROR: No se pudieron cargar las dependencias
    echo Por favor ejecuta install.bat primero
    echo.
)
pause
