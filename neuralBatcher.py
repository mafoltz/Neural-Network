# -*- coding: utf-8 -*-

import numpy as np


class NeuralNetworkBatcher():

    def __init__(self, neuralNetwork, numOfBatches, maxErrorDiff):
        self.neuralNetwork = neuralNetwork
        self.numOfBatches = numOfBatches
        self.maxErrorDiff = maxErrorDiff

    def train(self, instances, className, attributes=None):
        batches = np.array_split(instances, self.numOfBatches)

        oldError = 10000000000
        errorDif = 10000000000
        iteration = 0

        while errorDif > self.maxErrorDiff:
            error = 0

            # For each batch, add the batch error
            for batchIndex in range(0, len(batches)):
                self.neuralNetwork.train(batches[batchIndex], className, attributes)
                error += self.neuralNetwork.regularizedCost

            # Calculate error diff by comparing batches error average with previous error average
            error = error / len(batches)
            errorDif = abs(error - oldError)
            oldError = error

            print('Iteration {} trained with error: {}'.format(iteration, error))
            iteration += 1

    def evaluate(self, test, className):
        return self.neuralNetwork.evaluate(test, className)
