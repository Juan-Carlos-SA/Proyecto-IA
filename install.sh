#!/bin/bash
# Script de instalacion unica
# Instala todas las dependencias necesarias para ejecutar el programa

clear
echo "========================================"
echo "  Instalador - Color Recognition"
echo "========================================"
echo ""

echo "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "ERROR: Python no esta instalado"
    echo ""
    echo "SOLUCION:"
    echo "  macOS:"
    echo "    brew install python3"
    echo ""
    echo "  Linux (Ubuntu/Debian):"
    echo "    sudo apt-get update"
    echo "    sudo apt-get install python3 python3-pip"
    echo ""
    exit 1
fi

python3 --version

echo ""
echo "Instalando dependencias..."
echo "  - opencv-python"
echo "  - numpy"
echo "  - Pillow"
echo ""

python3 -m pip install -q --upgrade pip
python3 -m pip install -q opencv-python>=4.5 numpy>=1.19 Pillow>=9.0

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: No se pudieron instalar las dependencias"
    exit 1
fi

clear
echo ""
echo "========================================"
echo "  Instalacion completada!"
echo "========================================"
echo ""
echo "Ya puedes ejecutar el programa con:"
echo "  bash run_gui.sh (interfaz grafica)"
echo ""
