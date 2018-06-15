# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import random
from math import ceil, sqrt
from datetime import datetime

from validation import Attribute, CrossValidator, FoldSeparator, Measurer
from neural import NeuralNetwork


def checkFilesFrom(args):
    # Check number of files
    if len(args) < 4:
        print('Requires a network file, an initial weights file and a dataset file as arguments')
        exit(0)

    # Check if files exist
    filenames = []
    for i in range(1, 4):
        filename = args[i]
        if not os.path.exists(filename):
            print('File named {} does not exist'.format(filename))
            exit(0)
        else:
            filenames.append(filename)

    return filenames


def readNetworkFile(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]
    return lines


def readWeightsFile(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]
        layers = [layer.split(';') for layer in lines]
        weights = []
        for layer in layers:
            # Read and add weights for current layer
            newWeights = np.array([[float(weight) for weight in layerWeights.split(',')] for layerWeights in layer])
            weights.append(newWeights)
    return np.array(weights).transpose()


def readDatasetFile(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]

        # Gets the class names (they appear after a ';')
        classNames = [className.strip() for className in lines[0].split(';')[1].split(',')]

        # Split the values from readed lines in lists and gets the names of the headers
        lists = [[values.strip() for values in line.replace(';',',').split(',')] for line in lines]
        headers = lists[0]

        # Removes the header line and maps the values
        instances = [parseInstance(headers, values) for values in lists[1:]]
    return instances, classNames


def parseInstance(headers, values):
    instance = {}
    for header, value in zip(headers, values):
        instance[header] = Attribute(value)
    return instance


def normalize(instances, field):
    fields = [instance[field] for instance in instances]
    minimum = min(fields).value
    maximum = max(fields).value
    if maximum == minimum and maximum == 0:
        maximum = 1
    for instance in instances:
        if instance[field].type == Attribute.Numerical:
            instance[field].value = (instance[field].value - minimum) / (maximum - minimum)
    return instances


if __name__ == '__main__':
    start = datetime.now()

    # Read data from input files
    filenames = checkFilesFrom(sys.argv)

    networkFile = readNetworkFile(filenames[0])
    regulation = float(networkFile[0])
    configuration = [int(numOfNeurons) for numOfNeurons in networkFile[1:]]

    weights = readWeightsFile(filenames[1])

    instances, classNames = readDatasetFile(filenames[2])

    # Tests
    print('regulation = {}\n'.format(regulation))
    print('configuration = {}\n'.format(configuration))
    print('weights = {}\n'.format(weights))
    print('class names = {}\n'.format(classNames))

    # Set seed
    random.seed(0)

    # Initialize neural network
    neuralNetwork = NeuralNetwork(0, 0, configuration, regulation)
    neuralNetwork.weights = weights
    neuralNetwork.train(instances, classNames)

    # Apply cross validation and print results
    validator = CrossValidator(10, neuralNetwork)
    acc, f1 = validator.validate(instances, classNames)

    print('f1:', f1.average, f1.std_dev)
    print('duration: {}'.format((datetime.now() - start).total_seconds()))
