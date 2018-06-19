# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import random
from math import ceil, sqrt
from datetime import datetime

from validation import Attribute, CrossValidator, FoldSeparator, Measurer
from neural import NeuralNetwork


TRAINING_MODE = 'training'
NUMERICAL_VERIFICATION_MODE = 'numerical'
BACKPROPAGATION_MODE = 'backpropagation'


def checkFilesFrom(args):
    # Check number of files
    if len(args) < 5:
        print('Requires an execution mode, a network file, an initial weights file and a dataset file as arguments')
        exit(0)

    # Check if execution mode is valid
    executionMode = args[1]
    if not (executionMode == TRAINING_MODE or executionMode == NUMERICAL_VERIFICATION_MODE or executionMode == BACKPROPAGATION_MODE):
        print('Execution mode must be one of the follows: <{}>, <{}> or <{}>.'.format(TRAINING_MODE, NUMERICAL_VERIFICATION_MODE, BACKPROPAGATION_MODE))
        exit(0)

    # Check if files exist
    filenames = []
    for i in range(2, 5):
        filename = args[i]
        if not os.path.exists(filename):
            print('File named {} does not exist'.format(filename))
            exit(0)
        else:
            filenames.append(filename)

    return executionMode, filenames


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


def readTrainingDatasetFile(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]

        # Gets the names of the headers
        lists = [line.replace(';',',').split(',') for line in lines]
        headers = lists[0]

        # Get the class name
        className = headers[len(headers) - 1]

        # Removes the header line and maps the values
        instances = [parseInstance(headers, values) for values in lists[1:]]
    return instances, className


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


def executeTraining(self, neuralNetwork, dataFilename):
    # Read instances and class name
    instances, className = readTrainingDatasetFile(filenames[2])
    print('class names = {}\n'.format(className))

    # Apply cross validation and print results
    validator = CrossValidator(10, neuralNetwork)
    acc, f1 = validator.validate(instances, className)

    print('f1:', f1.average, f1.std_dev)


def executeNumericalVerification(self, neuralNetwork, dataFilename):
    pass


def executeBackpropagation(self, neuralNetwork, dataFilename):
    # Read instances and class names
    instances, classNames = readDatasetFile(filenames[2])
    print('class names = {}\n'.format(classNames))

    # Train neural network with backpropagation
    neuralNetwork.train(instances, classNames)


if __name__ == '__main__':
    start = datetime.now()

    # Set seed
    random.seed(0)

    # Read data from input files
    executionMode, filenames = checkFilesFrom(sys.argv)

    networkFile = readNetworkFile(filenames[0])
    regulation = float(networkFile[0])
    configuration = [int(numOfNeurons) for numOfNeurons in networkFile[1:]]
    
    weights = readWeightsFile(filenames[1])

    # Initialize neural network
    neuralNetwork = NeuralNetwork(0, 0, configuration, regulation)
    neuralNetwork.weights = weights

    # Tests
    print('regulation = {}\n'.format(regulation))
    print('configuration = {}\n'.format(configuration))
    print('weights = {}\n'.format(weights))

    # Execute algorithm
    if executionMode == TRAINING_MODE:
        self.executeTraining(neuralNetwork, dataFilename)

    elif executionMode == NUMERICAL_VERIFICATION_MODE:
        self.executeNumericalVerification(neuralNetwork, dataFilename)

    elif executionMode == BACKPROPAGATION_MODE:
        self.executeBackpropagation(neuralNetwork, dataFilename)

    print('duration: {}'.format((datetime.now() - start).total_seconds()))
