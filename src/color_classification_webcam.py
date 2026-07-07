#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Original author : Ahmet Ozlu
# --- Mejoras         : deteccion de desenfoque, suavizado temporal
# ---                   de la prediccion
# ----------------------------------------------
#
import cv2
import os
from collections import deque, Counter
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier

# Change to src directory to ensure correct file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# --- Parametros ajustables ---
BLUR_THRESHOLD = 60          
MIN_CONTRAST = 8              
SMOOTHING_WINDOW = 7           
PROCESS_EVERY_N_FRAMES = 2    

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

    status_text = stable_prediction
    text_color = (255, 255, 255)

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_blurry(gray_frame):
            status_text = 'Desenfocado - ajusta el enfoque'
            text_color = (0, 0, 255)  # rojo = advertencia
        else:
            color_histogram_feature_extraction.color_histogram_of_test_image(frame, use_center_roi=False)
            prediction = knn_classifier.main('training.data', 'test.data')
            recent_predictions.append(prediction)
            stable_prediction = Counter(recent_predictions).most_common(1)[0][0]
            status_text = stable_prediction

    cv2.putText(
        frame,
        'Prediction: ' + status_text,
        (15, 45),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        text_color,
        2,
        )

    # Display the resulting frame
    cv2.imshow('color classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
