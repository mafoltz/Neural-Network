# -*- coding: utf-8 -*-

import numpy as np


class NeuralBatcher():

    def __init__(self, numBatches, maxIterations, neuralNetwork):
        self.numBatches = numBatches
        self.neuralNetwork = neuralNetwork
        self.maxIterations = maxIterations

    def train(self, instances, className, attributes=None):
        batches = np.array_split(instances, self.numBatches)

        oldError = 10000000000
        errorDif = 10000000000
        batchIndex = 0
        iteration = 0
        while errorDif > 0.002 and iteration < self.maxIterations:
            error = self.neuralNetwork.train(batches[batchIndex], className, attributes)
            errorDif = error - oldError
            batchIndex = batchIndex + 1
            if batchIndex == len(batches):
                batchIndex = 0
            iteration += 1

    def evaluate(self, test, className):
        return self.neuralNetwork.evaluate(test, className)
