#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Original author : Ahmet Ozlu
# --- Mejoras         : voto ponderado por distancia + k seguro
# ----------------------------------------------
#
# CAMBIOS RESPECTO A LA VERSION ORIGINAL
# ----------------------------------------------
# - responseOfNeighbors ahora pondera cada vecino por 1/distancia en vez de
#   contar todos los votos por igual. Con datasets de entrenamiento pequenos
#   (por ejemplo la carpeta "blue" solo tiene 6 imagenes) un vecino lejano ya
#   no puede empatar/ganarle a varios vecinos mucho mas cercanos.
# - k se ajusta automaticamente para nunca superar la cantidad de datos de
#   entrenamiento disponibles (evita errores si el dataset es muy chico).
# ----------------------------------------------

import csv
import math
import operator


def calculateEuclideanDistance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)
    return math.sqrt(distance)


def kNearestNeighbors(training_feature_vector, testInstance, k):
    """Devuelve los k vecinos mas cercanos como pares (fila, distancia)."""
    distances = []
    length = len(testInstance)
    for x in range(len(training_feature_vector)):
        dist = calculateEuclideanDistance(testInstance, training_feature_vector[x], length)
        distances.append((training_feature_vector[x], dist))
    distances.sort(key=operator.itemgetter(1))

    k_actual = min(k, len(distances))
    if k_actual == 0:
        raise ValueError("No training data available for classification")

    return distances[:k_actual]


def responseOfNeighbors(neighbors_with_dist):
    """Voto ponderado: los vecinos mas cercanos pesan mas que los lejanos."""
    votes = {}
    for (neighbor, dist) in neighbors_with_dist:
        label = neighbor[-1]
        weight = 1.0 / (dist + 1e-6)
        votes[label] = votes.get(label, 0.0) + weight
    return max(votes.items(), key=operator.itemgetter(1))[0]


def loadDataset(filename, filename2, training_feature_vector=None, test_feature_vector=None):
    if training_feature_vector is None:
        training_feature_vector = []
    if test_feature_vector is None:
        test_feature_vector = []

    with open(filename) as csvfile:
        dataset = list(csv.reader(csvfile))
        if not dataset:
            raise ValueError(f"Training data file '{filename}' is empty. Please ensure training images are in the training_dataset folder.")
        for row in dataset:
            for y in range(3):
                row[y] = float(row[y])
            training_feature_vector.append(row)

    with open(filename2) as csvfile:
        dataset = list(csv.reader(csvfile))
        if not dataset:
            raise ValueError(f"Test data file '{filename2}' is empty.")
        for row in dataset:
            for y in range(3):
                row[y] = float(row[y])
            test_feature_vector.append(row)

    return training_feature_vector, test_feature_vector


def main(training_data, test_data, k=5):
    training_feature_vector, test_feature_vector = loadDataset(training_data, test_data)
    # k nunca puede superar la cantidad de muestras de entrenamiento disponibles
    k = max(1, min(k, len(training_feature_vector)))

    predictions = []
    for instance in test_feature_vector:
        neighbors = kNearestNeighbors(training_feature_vector, instance, k)
        predictions.append(responseOfNeighbors(neighbors))
    return predictions[0]
