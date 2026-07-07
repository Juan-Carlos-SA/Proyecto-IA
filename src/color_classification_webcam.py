#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Original author : Ahmet Ozlu
# --- Mejoras         : recalibracion de enfoque, deteccion de desenfoque
# ---                   por porcentaje, analisis solo del recuadro central,
# ---                   confianza de la prediccion, suavizado temporal e
# ---                   instrucciones de uso en pantalla.
# ----------------------------------------------
#
# CAMBIOS PRINCIPALES RESPECTO A LA VERSION ANTERIOR
# ----------------------------------------------------
# 1) Enfoque: al iniciar se intenta activar el autoenfoque continuo de la
#    camara y, ademas, se puede forzar una recalibracion manual en cualquier
#    momento presionando "f" (util con camaras/DroidCam que no reaccionan al
#    autoenfoque). Se muestra en pantalla un porcentaje de nitidez en vivo,
#    no solo un aviso de "borroso" cuando ya es demasiado tarde.
# 2) Deteccion de color: ahora solo se analiza el recuadro central que se ve
#    en pantalla (no el frame completo). Antes se mezclaba el fondo, la mano
#    que sostiene el objeto, etc. en el mismo histograma que el objeto, lo
#    que hacia el resultado mucho menos confiable.
# 3) Se muestra la confianza (%) de cada prediccion junto al color, para que
#    el usuario sepa si la lectura es solida o dudosa.
# 4) Instrucciones de uso visibles en la propia ventana de la camara.
# ----------------------------------------------

import cv2
import os
from collections import deque, Counter
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
from color_recognition_api import focus_utils

# Change to src directory to ensure correct file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# --- Parametros ajustables ---
BLUR_THRESHOLD = 60            # mas bajo = mas tolerante a imagenes borrosas
MIN_CONTRAST = 8               # por debajo de esto no se puede medir enfoque de forma confiable
SMOOTHING_WINDOW = 7            # cuantas predicciones recientes se combinan para estabilizar el resultado
PROCESS_EVERY_N_FRAMES = 2      # analizar 1 de cada N frames (rendimiento)
MIN_CONFIDENCE = 40             # por debajo de este % de confianza, la lectura no se toma en cuenta
ROI_RATIO = 0.45                # fraccion de la dimension menor de la imagen que cubre el recuadro central

# Try to find an available camera
cap = None
for i in range(10):  # Try indices 0-9
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret and frame is not None:
            cap = test_cap
            print(f"Camera found at index {i}")
            break
        test_cap.release()

if cap is None:
    print("Error: Could not open any camera. Please check your camera/DroidCam connection.")
    exit()

# Se intenta activar el autoenfoque continuo. Si la camara/driver no lo
# soporta, no pasa nada: el usuario siempre puede presionar "f" para forzar
# una recalibracion manual del enfoque.
focus_utils.try_enable_autofocus(cap)

# checking whether the training data is ready
PATH = './training.data'

training_data_valid = False
if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    if os.path.getsize(PATH) > 0:
        print('training data is ready, classifier is loading...')
        training_data_valid = True
    else:
        print('training data file is empty, regenerating...')

if not training_data_valid:
    print('training data is being created...')
    if os.path.isfile(PATH):
        os.remove(PATH)
    color_histogram_feature_extraction.training()
    print('training data is ready, classifier is loading...')


def roi_gray_of(frame):
    """Devuelve en escala de grises solo el recuadro central del frame."""
    x1, y1, x2, y2 = color_histogram_feature_extraction.get_center_roi(frame, roi_ratio=ROI_RATIO)
    region = frame[y1:y2, x1:x2]
    return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)


def draw_instructions(frame):
    """Barra de instrucciones de uso en la parte inferior de la ventana."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 34), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame,
        'Objeto dentro del recuadro  |  f = recalibrar enfoque  |  q = salir',
        (10, h - 11),
        cv2.FONT_HERSHEY_PLAIN,
        1.1,
        (255, 255, 255),
        1,
    )


def draw_sharpness_bar(frame, percent, x, y, width=160, height=14):
    """Barra horizontal que muestra el % de nitidez en vivo."""
    color = (0, 200, 0) if percent >= 55 else ((0, 165, 255) if percent >= 30 else (0, 0, 255))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (90, 90, 90), 1)
    fill_w = int(width * (percent / 100.0))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color, -1)
    cv2.putText(frame, f'Enfoque: {int(percent)}%', (x, y - 6), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1)


recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
stable_prediction = 'n.a.'
last_confidence = 0.0
frame_count = 0
recalibrating_message_ttl = 0

print('Instrucciones: coloca el objeto dentro del recuadro central.')
print("Presiona 'f' para recalibrar el enfoque, 'q' para salir.")

while True:

    # Capture frame-by-frame
    (ret, frame) = cap.read()

    # Skip frame if read failed or frame is empty
    if not ret or frame is None or frame.size == 0:
        print("Warning: Invalid frame, skipping...")
        continue

    frame_count += 1

    x1, y1, x2, y2 = color_histogram_feature_extraction.get_center_roi(frame, roi_ratio=ROI_RATIO)
    roi = frame[y1:y2, x1:x2]

    status_text = stable_prediction
    text_color = (255, 255, 255)
    box_color = (0, 200, 0)  # verde = ok
    sharpness_pct = 0.0

    if roi.size > 0:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness_pct = focus_utils.sharpness_percent(gray_roi)

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            if focus_utils.is_blurry(gray_roi, threshold=BLUR_THRESHOLD, min_contrast=MIN_CONTRAST):
                status_text = 'Desenfocado - ajusta el enfoque o presiona f'
                text_color = (0, 0, 255)  # rojo = advertencia
                box_color = (0, 0, 255)
            else:
                color_histogram_feature_extraction.color_histogram_of_test_image(roi, use_center_roi=False)
                prediction, confidence = knn_classifier.main('training.data', 'test.data')
                if confidence >= MIN_CONFIDENCE:
                    recent_predictions.append(prediction)
                    last_confidence = confidence
                if recent_predictions:
                    stable_prediction = Counter(recent_predictions).most_common(1)[0][0]
                status_text = stable_prediction

    # Recuadro central donde debe colocarse el objeto
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    label = f'Prediction: {status_text}'
    if status_text not in ('n.a.', 'Desenfocado - ajusta el enfoque o presiona f'):
        label += f' ({int(last_confidence)}%)'

    cv2.putText(frame, label, (15, 45), cv2.FONT_HERSHEY_PLAIN, 3, text_color, 2)
    draw_sharpness_bar(frame, sharpness_pct, 15, 75)

    if recalibrating_message_ttl > 0:
        cv2.putText(frame, 'Recalibrando enfoque...', (15, 105), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 200, 255), 1)
        recalibrating_message_ttl -= 1

    draw_instructions(frame)

    # Display the resulting frame
    cv2.imshow('color classifier', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        print('Recalibrando enfoque...')
        recalibrated = focus_utils.autofocus_sweep(cap, roi_gray_of)
        if recalibrated:
            print('Enfoque recalibrado.')
        else:
            print('Esta camara no soporta enfoque manual controlable desde el programa.')
        recalibrating_message_ttl = 15
        recent_predictions.clear()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
