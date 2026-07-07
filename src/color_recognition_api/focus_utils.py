#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# Utilidades de enfoque, compartidas por color_classification_webcam.py y
# color_classification_gui.py para no duplicar la misma logica dos veces.
# ----------------------------------------------
#
# QUE HACE ESTE MODULO
# ----------------------------------------------------
# 1) Mide que tan nitida esta una region de la imagen (varianza del
#    Laplaciano: cuanto mas alta, mas bordes definidos = mas nitida).
# 2) Convierte esa medida en un porcentaje 0-100 facil de mostrar en
#    pantalla, para que el usuario vea en vivo si esta bien enfocado.
# 3) Ofrece una recalibracion de enfoque manual (autofocus_sweep): muchas
#    camaras USB / DroidCam no reaccionan a CAP_PROP_AUTOFOCUS=1 (lo
#    aceptan pero no hacen nada), asi que como respaldo se prueban varios
#    valores de foco manual y se deja el que da la imagen mas nitida.
# ----------------------------------------------

import cv2


def sharpness_score(gray_region):
    """Varianza del Laplaciano: mayor valor = imagen mas nitida."""
    if gray_region is None or gray_region.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_region, cv2.CV_64F).var())


def sharpness_percent(gray_region, reference=250.0):
    """
    Convierte la nitidez cruda a un porcentaje 0-100 aproximado, usando
    `reference` como el valor que se considera "totalmente nitido" (100%).
    Es solo para mostrar en pantalla, no cambia la logica de deteccion.
    """
    score = sharpness_score(gray_region)
    return max(0.0, min(100.0, (score / reference) * 100.0))


def is_blurry(gray_region, threshold=60, min_contrast=8):
    """
    True si la region esta demasiado borrosa para confiar en su color.
    Una superficie lisa de un solo color (poco contraste) no permite medir
    el enfoque de forma confiable con el Laplaciano, asi que en ese caso se
    asume que esta bien enfocada en vez de marcarla como borrosa por error.
    """
    if gray_region is None or gray_region.size == 0:
        return True
    if gray_region.std() < min_contrast:
        return False
    return sharpness_score(gray_region) < threshold


def autofocus_sweep(cap, roi_gray_fn, focus_values=range(0, 256, 16), settle_frames=2):
    """
    Recalibra el enfoque manualmente probando distintos valores de
    CAP_PROP_FOCUS y quedandose con el que da mayor nitidez en la region que
    devuelve `roi_gray_fn(frame)` (una funcion que recibe un frame BGR y
    devuelve la region en escala de grises a evaluar).

    Sirve de respaldo cuando el autoenfoque automatico de la camara no esta
    disponible o no reacciona. Si la camara no soporta control manual de
    foco, no rompe nada: simplemente devuelve False sin cambiar el estado
    de la camara.

    Devuelve True si se pudo recalibrar, False si el dispositivo no soporta
    foco manual o no se pudo leer ningun frame valido durante el barrido.
    """
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception:
        pass

    best_value, best_score = None, -1.0
    read_any_frame = False

    for value in focus_values:
        try:
            cap.set(cv2.CAP_PROP_FOCUS, value)
        except Exception:
            break

        # deja unos frames para que la camara aplique el nuevo foco antes
        # de medir la nitidez, si no la medicion queda desfasada
        for _ in range(settle_frames):
            cap.read()

        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        read_any_frame = True

        gray = roi_gray_fn(frame)
        score = sharpness_score(gray)
        if score > best_score:
            best_score, best_value = score, value

    if not read_any_frame or best_value is None:
        return False

    try:
        cap.set(cv2.CAP_PROP_FOCUS, best_value)
    except Exception:
        return False
    return True


def try_enable_autofocus(cap):
    """Intenta activar el autoenfoque continuo de la camara. No todas las
    camaras/drivers lo soportan desde OpenCV; si falla, se ignora sin
    afectar el resto del programa."""
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    except Exception:
        pass
