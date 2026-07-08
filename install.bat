@echo off
REM Script de instalacion unica
REM Instala todas las dependencias necesarias para ejecutar el programa

cls
echo ========================================
echo  Instalador - Color Recognition
echo ========================================
echo.

echo Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python no esta instalado o no se encuentra en el PATH
    echo.
    echo SOLUCION:
    echo 1. Descarga Python desde https://www.python.org/downloads/
    echo 2. Durante la instalacion, MARCA la opcion: "Add Python to PATH"
    echo 3. Vuelve a ejecutar este script
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo   %PYTHON_VERSION% encontrado

echo.
echo Instalando dependencias...
echo   - opencv-python
echo   - numpy
echo   - Pillow
echo.

python -m pip install -q --upgrade pip
python -m pip install -q opencv-python>=4.5 numpy>=1.19 Pillow>=9.0

if errorlevel 1 (
    echo.
    echo ERROR: No se pudieron instalar las dependencias
    pause
    exit /b 1
)

cls
echo.
echo ========================================
echo  Instalacion completada!
echo ========================================
echo.
echo Ya puedes ejecutar el programa con:
echo   - run_gui.bat (interfaz grafica)
echo.
pause
