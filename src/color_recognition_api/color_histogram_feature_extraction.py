#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Original author : Ahmet Ozlu
# --- Mejoras         : robustez ante iluminacion / enfoque
# ----------------------------------------------
#
# CAMBIOS PRINCIPALES RESPECTO A LA VERSION ORIGINAL
# ----------------------------------------------------
# 1) Se trabaja en espacio de color HSV en lugar de RGB puro. El canal H (tono)
#    es en gran medida independiente del brillo, por lo que un mismo objeto
#    bajo luz fuerte, tenue o con sombra sigue dando un tono H similar, cosa
#    que NO ocurre con R, G, B (que cambian mucho al cambiar la iluminacion).
#
# 2) Antes de calcular el histograma se descartan los pixeles "no fiables":
#      - Sombras muy oscuras (V bajo)
#      - Brillos quemados / reflejos (V muy alto y S muy bajo)
#    Estos pixeles no representan el color real del objeto, solo el efecto
#    de la luz, y eran una causa directa de clasificaciones erroneas.
#
# 3) Se aplica CLAHE (ecualizacion adaptativa de contraste) sobre el canal de
#    luminancia para atenuar diferencias fuertes de exposicion/sombra dentro
#    de la misma imagen antes de analizar el color.
#
# 4) El histograma de cada canal se suaviza levemente antes de tomar el pico,
#    para que un solo grupo de pixeles ruidosos (ruido de sensor / compresion
#    JPEG) no desvie la deteccion.
#
# 5) En el video en vivo, solo se analiza una region central (ROI) en vez de
#    todo el frame, para que el fondo no contamine la lectura de color del
#    objeto que el usuario esta mostrando a la camara.
# ----------------------------------------------

import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------

def _normalize_illumination(bgr_image):
    """Atenua sombras/exposicion desigual con CLAHE sobre el canal L (LAB)."""
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _denoise(bgr_image):
    """Desenfoque leve para reducir ruido de sensor antes del histograma."""
    return cv2.GaussianBlur(bgr_image, (5, 5), 0)


def get_center_roi(image, roi_ratio=0.35):
    """
    Devuelve (x1, y1, x2, y2) de un recuadro cuadrado centrado que cubre
    `roi_ratio` de la dimension mas pequena de la imagen. Se usa para que en
    la webcam solo se analice el objeto mostrado frente a la camara y no el
    fondo de la habitacion.
    """
    h, w = image.shape[:2]
    size = int(min(h, w) * roi_ratio)
    size = max(size, 10)
    cx, cy = w // 2, h // 2
    x1, y1 = max(cx - size // 2, 0), max(cy - size // 2, 0)
    x2, y2 = min(cx + size // 2, w), min(cy + size // 2, h)
    return x1, y1, x2, y2


def _valid_pixel_mask(hsv_image):
    """
    Descarta:
      - Sombra profunda: V bajo (< 40)
      - Brillo quemado / reflejo especular: V muy alto (> 250) y S bajo (< 30)
    Si casi todo queda descartado (p. ej. una imagen totalmente oscura o muy
    clara), se usan todos los pixeles como respaldo para no dejar la mascara
    vacia.
    """
    h, s, v = cv2.split(hsv_image)
    shadow_mask = v < 40
    highlight_mask = (v > 250) & (s < 30)
    invalid = shadow_mask | highlight_mask
    mask = np.uint8((~invalid) * 255)
    if cv2.countNonZero(mask) < 0.05 * mask.size:
        mask = np.full(mask.shape, 255, dtype=np.uint8)
    return mask


def _smooth_hist(hist):
    kernel = np.array([1, 2, 4, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(hist.flatten(), kernel, mode='same')


def _dominant_hsv(bgr_region):
    """
    Calcula un descriptor (H, S, V) robusto ante iluminacion para una region
    BGR: quita ruido, normaliza exposicion, enmascara sombras/brillos y toma
    el pico suavizado del histograma de cada canal.
    """
    region = _denoise(bgr_region)
    region = _normalize_illumination(region)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = _valid_pixel_mask(hsv)

    h_hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], mask, [256], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], mask, [256], [0, 256])

    h_peak = int(np.argmax(_smooth_hist(h_hist)))
    s_peak = int(np.argmax(_smooth_hist(s_hist)))
    v_peak = int(np.argmax(_smooth_hist(v_hist)))

    return h_peak, s_peak, v_peak


# El brillo (V) cambia mucho con la iluminacion (una misma superficie se ve
# con V muy distinto bajo poca luz vs mucha luz), mientras que el tono
# ponderado por saturacion (x, y) es mucho mas estable. Si V pesara igual
# que x/y, una foto oscura de un objeto de color podia terminar mas cerca
# de los ejemplos de entrenamiento de "black" (que tambien tienen V bajo)
# que de los ejemplos de su propio color. Bajarle el peso deja que el tono
# siga siendo el factor principal, y V solo termina de inclinar la balanza
# para distinguir negro/gris/blanco (que es donde realmente hace falta).
V_WEIGHT = 0.4


def _hue_to_xy(h_peak, s_peak):
    """
    Convierte (H, S) a un punto cartesiano (x, y) sobre el circulo de color:
        x = S * cos(theta),  y = S * sin(theta)
    Esto resuelve dos problemas del H crudo usado antes:
      1. El tono es una magnitud CIRCULAR (H=0 y H=179 son casi el mismo
         color, el rojo), pero comparar los numeros crudos con distancia
         euclidiana los trataba como opuestos. En (x, y) quedan juntos.
      2. Cuando la saturacion es baja (grises, negro, blanco casi puro) el
         tono es en la practica ruido: pequenas variaciones de luz pueden
         mover el H medido sin que el color realmente cambie. Al multiplicar
         por S, esos casos colapsan cerca del origen (x=0, y=0) sin importar
         cual haya sido el H medido, en vez de "contar" como una diferencia
         de tono grande frente a un color muy saturado del mismo tono.
    Para colores muy saturados (rojo, naranja, azul, etc.) el punto queda
    lejos del origen y ahi si el tono pesa fuerte en la distancia, que es
    justamente donde mas importa distinguir bien un color de otro.
    """
    theta = h_peak * (2.0 * np.pi / 180.0)  # H de OpenCV: 0-179 ~ 0-358 grados
    x = s_peak * np.cos(theta)
    y = s_peak * np.sin(theta)
    return x, y


def _dominant_features(bgr_region):
    h_peak, s_peak, v_peak = _dominant_hsv(bgr_region)
    x, y = _hue_to_xy(h_peak, s_peak)
    return x, y, v_peak * V_WEIGHT


# ---------------------------------------------------------------------------
# API publica (mismos nombres que la version original para no romper nada
# que ya llame a estas funciones)
# ---------------------------------------------------------------------------

def color_histogram_of_test_image(test_src_image, use_center_roi=True):
    image = test_src_image

    if use_center_roi:
        x1, y1, x2, y2 = get_center_roi(image)
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            region = image
    else:
        region = image

    x, y, v = _dominant_features(region)
    feature_data = f'{x},{y},{v}'

    with open('test.data', 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_training_image(img_name):
    # La etiqueta es el nombre de la carpeta que contiene la imagen. Asi,
    # cualquier carpeta nueva dentro de training_dataset (p. ej. "pink",
    # "grey", o cualquier otro color que agregues despues) se usa
    # automaticamente como una clase mas, sin tener que tocar el codigo.
    data_source = os.path.basename(os.path.dirname(img_name))

    image = cv2.imread(img_name)
    if image is None:
        return

    x, y, v = _dominant_features(image)
    feature_data = f'{x},{y},{v}'

    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():
    # Get the directory of this module and navigate to training_dataset
    module_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(module_dir)
    training_dir = os.path.join(src_dir, 'training_dataset')

    if not os.path.isdir(training_dir):
        return

    # Recorre TODAS las subcarpetas de training_dataset, sin necesidad de
    # mantener una lista fija de nombres de color en el codigo.
    for color in sorted(os.listdir(training_dir)):
        color_dir = os.path.join(training_dir, color)
        if not os.path.isdir(color_dir):
            continue
        for f in os.listdir(color_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                color_histogram_of_training_image(os.path.join(color_dir, f))
