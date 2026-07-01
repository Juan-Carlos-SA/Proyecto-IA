#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from color_recognition_api import knn_classifier as knn_classifier


def color_histogram_of_test_image(test_src_image):

    # load the image
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            # print(feature_data)

    with open('test.data', 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_training_image(img_name):

    # detect image color by using image file name to label training data
    if 'red' in img_name:
        data_source = 'red'
    elif 'yellow' in img_name:
        data_source = 'yellow'
    elif 'green' in img_name:
        data_source = 'green'
    elif 'orange' in img_name:
        data_source = 'orange'
    elif 'white' in img_name:
        data_source = 'white'
    elif 'black' in img_name:
        data_source = 'black'
    elif 'blue' in img_name:
        data_source = 'blue'
    elif 'violet' in img_name:
        data_source = 'violet'

    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():
    # Get the directory of this module and navigate to training_dataset
    module_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(module_dir)
    training_dir = os.path.join(src_dir, 'training_dataset')

    # red color training images
    red_dir = os.path.join(training_dir, 'red')
    if os.path.isdir(red_dir):
        for f in os.listdir(red_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                color_histogram_of_training_image(os.path.join(red_dir, f))

    # yellow color training images
    yellow_dir = os.path.join(training_dir, 'yellow')
    if os.path.isdir(yellow_dir):
        for f in os.listdir(yellow_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                color_histogram_of_training_image(os.path.join(yellow_dir, f))

    # green color training images
    green_dir = os.path.join(training_dir, 'green')
    if os.path.isdir(green_dir):
        for f in os.listdir(green_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                color_histogram_of_training_image(os.path.join(green_dir, f))

    # orange color training images
    orange_dir = os.path.join(training_dir, 'orange')
    if os.path.isdir(orange_dir):
        for f in os.listdir(orange_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                color_histogram_of_training_image(os.path.join(orange_dir, f))

    # white color training images
    white_dir = os.path.join(training_dir, 'white')
    if os.path.isdir(white_dir):
        for f in os.listdir(white_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                color_histogram_of_training_image(os.path.join(white_dir, f))

    # black color training images
    black_dir = os.path.join(training_dir, 'black')
    if os.path.isdir(black_dir):
        for f in os.listdir(black_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                color_histogram_of_training_image(os.path.join(black_dir, f))

    # blue color training images
    blue_dir = os.path.join(training_dir, 'blue')
    if os.path.isdir(blue_dir):
        for f in os.listdir(blue_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                color_histogram_of_training_image(os.path.join(blue_dir, f))
    
    # violet color training images
    violet_dir = os.path.join(training_dir, 'violet')
    if os.path.isdir(violet_dir):
        for f in os.listdir(violet_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                color_histogram_of_training_image(os.path.join(violet_dir, f))
