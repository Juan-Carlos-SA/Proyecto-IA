#!/bin/bash
# Script para ejecutar la interfaz grafica sin necesidad de entorno virtual

echo "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "ERROR: Python no esta instalado o no se encuentra en el PATH"
    echo ""
    echo "SOLUCION: Ejecuta primero install.sh"
    echo ""
    exit 1
fi

echo ""
echo "Iniciando interfaz grafica..."
cd src
python3 color_classification_gui.py
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: No se pudieron cargar las dependencias"
    echo "Por favor ejecuta install.sh primero"
    echo ""
fi
