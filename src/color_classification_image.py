#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Original author : Ahmet Ozlu
# -------------------------------------------------------------------------
#

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import platform
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)


def _has_display():
    """
    En Windows/Mac se asume que hay entorno grafico. En Linux se revisa la
    variable DISPLAY: si no existe (por ejemplo al correr por SSH o en un
    servidor), evitamos llamar a cv2.selectROI/cv2.imshow, porque OpenCV
    aborta el proceso (no lanza una excepcion capturable) cuando no
    encuentra un servidor grafico.
    """
    if platform.system() in ('Windows', 'Darwin'):
        return True
    return bool(os.environ.get('DISPLAY'))


HAS_DISPLAY = _has_display()

# read the test image
image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, 'nose.jpg')
source_image = cv2.imread(image_path)
if source_image is None:
    print(f"Error: no se pudo leer la imagen '{image_path}'")
    sys.exit(1)

prediction = 'n.a.'

# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.path.getsize(PATH) > 0:
    print('training data is ready, classifier is loading...')
else:
    print('training data is being created...')
    if os.path.isfile(PATH):
        os.remove(PATH)
    color_histogram_feature_extraction.training()
    print('training data is ready, classifier is loading...')


region_to_analyze = source_image
if HAS_DISPLAY:
    try:
        window_name = 'Selecciona el objeto (ENTER=confirmar, ESC=usar toda la imagen)'
        roi = cv2.selectROI(window_name, source_image, showCrosshair=True)
        cv2.destroyWindow(window_name)
        x, y, w, h = roi
        if w > 0 and h > 0:
            region_to_analyze = source_image[y:y + h, x:x + w]
    except cv2.error:
        print('No se pudo abrir la ventana de seleccion: se analizara la imagen completa.')
else:
    print('Sin entorno grafico disponible: se analizara la imagen completa.')

# get the prediction
color_histogram_feature_extraction.color_histogram_of_test_image(region_to_analyze, use_center_roi=False)
prediction = knn_classifier.main('training.data', 'test.data')
print('Detected color is:', prediction)

cv2.putText(
    source_image,
    'Prediction: ' + prediction,
    (15, 45),
    cv2.FONT_HERSHEY_PLAIN,
    3,
    (255, 255, 255),
    2,
    )

# Display the resulting frame
if HAS_DISPLAY:
    try:
        cv2.imshow('color classifier', source_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        out_path = os.path.join(base_dir, 'resultado.jpg')
        cv2.imwrite(out_path, source_image)
        print(f'No se pudo mostrar la ventana: resultado guardado en {out_path}')
else:
    out_path = os.path.join(base_dir, 'resultado.jpg')
    cv2.imwrite(out_path, source_image)
    print(f'Sin entorno grafico: resultado guardado en {out_path}')
