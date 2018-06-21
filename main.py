# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import random
from math import ceil, sqrt
from datetime import datetime

from validation import Attribute, CrossValidator, FoldSeparator, Measurer
from neural import NeuralNetwork
from neuralBatcher import NeuralNetworkBatcher


TRAINING_MODE = 'training'
NUMERICAL_VERIFICATION_MODE = 'numerical'
BACKPROPAGATION_MODE = 'backpropagation'


def checkFilesFrom(args):
    # Check number of files
    if len(args) < 2:
        print('Requires an execution mode as argument.')
        exit(0)

    # Check if execution mode is valid
    executionMode = args[1]
    if not (executionMode == TRAINING_MODE or executionMode == NUMERICAL_VERIFICATION_MODE or executionMode == BACKPROPAGATION_MODE):
        print('Execution mode must be one of the follows: <{}>, <{}> or <{}>.'.format(TRAINING_MODE, NUMERICAL_VERIFICATION_MODE, BACKPROPAGATION_MODE))
        exit(0)

    # Check if files exist
    filenames = []
    filenamesEndIndex = 5

    if executionMode == TRAINING_MODE:
        filenamesEndIndex = 4
        if len(args) != filenamesEndIndex:
            print('This execution mode requires a network file and a dataset file as arguments.')
            exit(0)
    else:
        if len(args) != filenamesEndIndex:
            print('This execution mode requires a network file, an initial weights file and a dataset file as arguments.')
            exit(0)

    for i in range(2, filenamesEndIndex):
        filename = args[i]
        if not os.path.exists(filename):
            print('File named {} does not exist.'.format(filename))
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

        # Join class attributes in one attribute
        className = 'class'
        for instance in instances:
            instance[className] = [instance[value] for value in classNames]
            for value in classNames:
                instance.pop(value, None)

    return instances, className


def readTrainingDatasetFile(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]

        # Gets the names of the headers
        lists = [line.replace(';',',').split(',') for line in lines]
        headers = lists[0]

        # Removes the header line and maps the values
        instances = [parseInstance(headers, values) for values in lists[1:]]

        # Normalize instances
        for field in headers:
            instances = normalize(instances, field)

        # Get the class name and values
        className = headers[len(headers) - 1]

        classValuesList = []
        for instance in instances:
            classValue = instance[className]
            classValue.type = Attribute.Categorical
            classValuesList.append(classValue)
        classValues = list(set(classValuesList))
        classValues.sort()

    return instances, className, classValues


def parseInstance(headers, values):
    instance = {}
    for header, value in zip(headers, values):
        instance[header] = Attribute(value)
    return instance


def normalize(instances, field):
    fields = [instance[field] for instance in instances]
    minimum = min(fields).value
    maximum = max(fields).value
    if maximum == minimum:
        maximum += 1
    for instance in instances:
        if instance[field].type == Attribute.Numerical:
            instance[field].value = (instance[field].value - minimum) / (maximum - minimum)
    return instances


def printWeights(weights):
    for i, thetas in enumerate(weights):
        print('Theta {} initial weights:'.format(i))
        for thetaWeights in thetas:
            print('\t{}'.format(thetaWeights))
        print()


def printTrainingSet(instances):
    print('Training set:')
    for i, instance in enumerate(instances):
        print('\tInstance {}:'.format(i))

        inputs = []
        outputs = []
        for attribute in instance.keys():
            attributeValue = instance[attribute]
            if isinstance(attributeValue, Attribute):
                inputs.append(attributeValue.value)
            else:
                for outputAttribute in attributeValue:
                    outputs.append(outputAttribute.value)

        print('\t\tx: {}'.format(inputs))
        print('\t\ty: {}'.format(outputs))
    print()


def printGradients(title, gradients):
    print('=======================================================================')
    print(title)
    for i, theta in enumerate(gradients):
        print('Theta {} gradients:'.format(i))
        for thetaGradients in theta:
            print('\t{}'.format(thetaGradients))
        print()


def printGradientErrors(gradientErrors):
    print('=======================================================================')
    print('Gradients error:\n')
    for i, gradientError in enumerate(gradientErrors):
        print('Theta {} gradients error: {}'.format(i, gradientError))
    print()


def createNeuralNetworkForTrainingFrom(filenames):
    # Read input data
    networkFile = readNetworkFile(filenames[0])
    regulation = float(networkFile[0])
    configuration = [int(numOfNeurons) for numOfNeurons in networkFile[1:]]

    instances, className, classValues = readTrainingDatasetFile(filenames[1])

    # Print input data
    print('Regulation: {}\n'.format(regulation))
    print('Configuration: {}\n'.format(configuration))
    print('Class name: {}\n'.format(className))
    print('Class values: {}\n'.format(classValues))

    # Initialize neural network
    neuralNetwork = NeuralNetwork(configuration, regulation, classValues)

    return neuralNetwork, instances, className


def createNeuralNetworkForVerificationFrom(filenames):
    # Read input data
    networkFile = readNetworkFile(filenames[0])
    regulation = float(networkFile[0])
    configuration = [int(numOfNeurons) for numOfNeurons in networkFile[1:]]

    weights = readWeightsFile(filenames[1])

    instances, className = readDatasetFile(filenames[2])

    # Print input data
    print('Regulation: {}\n'.format(regulation))
    print('Configuration: {}\n'.format(configuration))
    printWeights(weights)
    printTrainingSet(instances)

    # Initialize and train neural network
    print('Training neural network...\n')
    neuralNetwork = NeuralNetwork(configuration, regulation)
    neuralNetwork.weights = weights

    return neuralNetwork, instances, className


def executeTraining(filenames):
    neuralNetwork, instances, className = createNeuralNetworkForTrainingFrom(filenames)

    numOfBatches = 100
    maxErrorDiff = 0.1
    neuralNetworkBatches = NeuralNetworkBatcher(neuralNetwork, numOfBatches, maxErrorDiff)

    # Apply cross validation and print results
    validator = CrossValidator(10, neuralNetworkBatches)
    acc, f1 = validator.validate(instances, className)

    print('f1:', f1.average, f1.std_dev)


def executeNumericalVerification(filenames):
    neuralNetwork, instances, className = createNeuralNetworkForVerificationFrom(filenames)

    numericGradients = neuralNetwork.trainNumerically(0.0000010000, instances, className)
    backpropagationGradients = neuralNetwork.train(instances, className)
    gradientErrors = neuralNetwork.gradientErrors(numericGradients, backpropagationGradients)

    printGradients('Backpropagation calculated gradients:\n', backpropagationGradients)
    printGradients('Numerically calculated gradients:\n', numericGradients)
    printGradientErrors(gradientErrors)


def executeBackpropagation(filenames):
    neuralNetwork, instances, className = createNeuralNetworkForVerificationFrom(filenames)

    backpropagationGradients = neuralNetwork.train(instances, className)

    printGradients('Backpropagation calculated gradients:\n', backpropagationGradients)


if __name__ == '__main__':
    start = datetime.now()

    # Set seed
    random.seed(0)

    # Read execution mode and input filenames
    executionMode, filenames = checkFilesFrom(sys.argv)

    # Execute algorithm
    print('Executing {} mode...\n'.format(executionMode))

    if executionMode == TRAINING_MODE:
        executeTraining(filenames)

    elif executionMode == NUMERICAL_VERIFICATION_MODE:
        executeNumericalVerification(filenames)

    elif executionMode == BACKPROPAGATION_MODE:
        executeBackpropagation(filenames)

    print('Duration: {} seconds\n'.format((datetime.now() - start).total_seconds()))
