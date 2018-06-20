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
        batchIndex = 0

        while errorDif > self.maxErrorDiff:
            self.neuralNetwork.train(batches[batchIndex], className, attributes)
            
            error = self.neuralNetwork.regularizedCost
            errorDif = abs(error - oldError)
            oldError = error
            print('Batch {} trained with error: {}'.format(batchIndex, error))

            batchIndex = batchIndex + 1
            if batchIndex == len(batches):
                batchIndex = 0

    def evaluate(self, test, className):
        return self.neuralNetwork.evaluate(test, className)
