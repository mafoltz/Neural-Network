# -*- coding: utf-8 -*-
import sys
import os
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
        lines = [line.rstrip() for line in f]
    return lines


def readWeightsFile(filename):
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f]
        layers = [layer.split(';') for layer in lines]
        weights = [[layerWeights.split(',') for layerWeights in layer] for layer in layers]
    return weights


def readDatasetFile(filename):
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f]
        # Gets the names of the headers
        lists = [line.replace(';',',').split(',') for line in lines]
        headers = lists[0]
        # Removes the header line and maps the values
        instances = [parseInstance(headers, values) for values in lists[1:]]
    return instances


def parseInstance(headers, values):
    instance = {}
    for header, value in zip(headers, values):
        instance[header] = Attribute(value)
    return instance


def attributesAndClassNameFrom(instances):
    attributes = list(instances[0].keys())
    className = attributes[len(attributes) - 1]
    attributes.remove(className)
    for column in attributes:
        instances = normalize(instances, column)
    return attributes, className


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
    regulation = networkFile[0]
    configuration = networkFile[1:]

    weights = readWeightsFile(filenames[1])

    instances = readDatasetFile(filenames[2])
    attributes, className = attributesAndClassNameFrom(instances)

    # Set seed
    random.seed(0)

    # Initialize neural network
    neuralNetwork = NeuralNetwork(0, 0, configuration, regulation)
    neuralNetwork.weights = weights

    # Apply cross validation and print results
    validator = CrossValidator(10, neuralNetwork)
    acc, f1 = validator.validate(instances, className)

    print('f1:', f1.average, f1.std_dev)
    print('duration: {}'.format((datetime.now() - start).total_seconds()))
