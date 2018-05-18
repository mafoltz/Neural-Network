# -*- coding: utf-8 -*-
import sys
import os
import random
from math import ceil, sqrt
from datetime import datetime

from validation import Attribute, CrossValidator, FoldSeparator, Measurer


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


def parseInstance(headers, values):
    instance = {}
    for header, value in zip(headers, values):
        instance[header] = Attribute(value)
    return instance


def readCSV(filename, separator):
    with open(filename, 'r') as f:
        lines = [line.rstrip() for line in f]
        # Gets the names of the headers
        lists = [line.split(separator) for line in lines]
        headers = lists[0]
        # Removes the header line and maps the values
        instances = [parseInstance(headers, values) for values in lists[1:]]
    return instances


if __name__ == '__main__':
    start = datetime.now()
    if len(sys.argv) < 4:
        print('Requires a file name, separator and forest size as arguments')
        exit(0)
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print('File does not exist')
        exit(0)

    separator = sys.argv[2]
    instances = readCSV(filename, separator)

    attributes = list(instances[0].keys())
    className = attributes[len(attributes) - 1]
    attributes.remove(className)
    for column in attributes:
        instances = normalize(instances, column)

    random.seed(0)

    # TODO: Neural Network

    validator = CrossValidator(10, forest)
    acc, f1 = validator.validate(instances, className)

    print('f1:', f1.average, f1.std_dev)
    print('duration: {}'.format((datetime.now() - start).total_seconds()))
