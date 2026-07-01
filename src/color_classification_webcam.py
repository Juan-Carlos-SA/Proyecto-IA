#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path

# Change to src directory to ensure correct file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

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

prediction = 'n.a.'

# checking whether the training data is ready
PATH = './training.data'

# Check if training data exists and has content
training_data_valid = False
if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    # Check if file has content
    if os.path.getsize(PATH) > 0:
        print ('training data is ready, classifier is loading...')
        training_data_valid = True
    else:
        print ('training data file is empty, regenerating...')

if not training_data_valid:
    print ('training data is being created...')
    # Delete empty file if it exists
    if os.path.isfile(PATH):
        os.remove(PATH)
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')

while True:

    # Capture frame-by-frame
    (ret, frame) = cap.read()
    
    # Skip frame if read failed or frame is empty
    if not ret or frame is None or frame.size == 0:
        print("Warning: Invalid frame, skipping...")
        continue

    cv2.putText(
        frame,
        'Prediction: ' + prediction,
        (15, 45),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        200,
        )

    # Display the resulting frame
    cv2.imshow('color classifier', frame)

    color_histogram_feature_extraction.color_histogram_of_test_image(frame)

    prediction = knn_classifier.main('training.data', 'test.data')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()		
