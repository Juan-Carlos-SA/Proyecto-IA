#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Original author : Ahmet Ozlu
# --- Mejoras         : ROI central, deteccion de desenfoque,
# ---                   suavizado temporal de la prediccion
# ----------------------------------------------
#
# CAMBIOS RESPECTO A LA VERSION ORIGINAL
# ----------------------------------------------
# 1) ROI central: solo se analiza el recuadro que se dibuja en pantalla, no
#    el frame completo. Antes, si habia varios objetos/colores de fondo en
#    camara, el histograma mezclaba todo y el resultado era practicamente
#    aleatorio.
# 2) Deteccion de desenfoque (varianza del Laplaciano): si el recuadro esta
#    borroso (camara sin foco, movimiento), se avisa en pantalla en vez de
#    devolver una prediccion inventada con datos poco confiables.
# 3) Suavizado temporal: se guarda un historial de las ultimas N
#    predicciones y se muestra la mas votada, para eliminar el "parpadeo"
#    de la prediccion entre frames.
# ----------------------------------------------

import cv2
import os
from collections import deque, Counter
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier

# Change to src directory to ensure correct file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# --- Parametros ajustables ---
BLUR_THRESHOLD = 60          # mas bajo = mas tolerante a imagenes borrosas
MIN_CONTRAST = 8              # por debajo de esto, la superficie es demasiado
                               # lisa/uniforme (mouse, pared, tela) como para
                               # medir enfoque de forma confiable
SMOOTHING_WINDOW = 15         # cuantas predicciones recientes se promedian
PROCESS_EVERY_N_FRAMES = 2    # analizar 1 de cada N frames (rendimiento)

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


def is_blurry(gray_roi, threshold=BLUR_THRESHOLD, min_contrast=MIN_CONTRAST):
    # Un objeto de color solido y liso (ej. un mouse negro) tiene pocos
    # bordes de por si, incluso perfectamente enfocado, asi que la varianza
    # del Laplaciano ahi es baja de forma natural. Si la superficie ya es
    # casi uniforme (poco contraste), no tiene sentido evaluar el enfoque:
    # simplemente se asume que esta bien.
    if gray_roi.std() < min_contrast:
        return False
    return cv2.Laplacian(gray_roi, cv2.CV_64F).var() < threshold


recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
stable_prediction = 'n.a.'
frame_count = 0

while True:

    # Capture frame-by-frame
    (ret, frame) = cap.read()

    # Skip frame if read failed or frame is empty
    if not ret or frame is None or frame.size == 0:
        print("Warning: Invalid frame, skipping...")
        continue

    frame_count += 1
    x1, y1, x2, y2 = color_histogram_feature_extraction.get_center_roi(frame)
    roi = frame[y1:y2, x1:x2]

    status_text = stable_prediction
    box_color = (0, 200, 0)  # verde = ok

    if roi.size > 0 and frame_count % PROCESS_EVERY_N_FRAMES == 0:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if is_blurry(gray_roi):
            status_text = 'Desenfocado - ajusta el enfoque'
            box_color = (0, 0, 255)  # rojo = advertencia
        else:
            color_histogram_feature_extraction.color_histogram_of_test_image(frame)
            prediction = knn_classifier.main('training.data', 'test.data')
            recent_predictions.append(prediction)
            stable_prediction = Counter(recent_predictions).most_common(1)[0][0]
            status_text = stable_prediction

    # Guia visual: recuadro donde debe colocarse el objeto
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(
        frame,
        'Coloca el objeto dentro del recuadro',
        (15, 25),
        cv2.FONT_HERSHEY_PLAIN,
        1.2,
        (255, 255, 255),
        1,
        )
    cv2.putText(
        frame,
        'Prediction: ' + status_text,
        (15, 60),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 255, 255),
        2,
        )

    # Display the resulting frame
    cv2.imshow('color classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
