# -*- coding: utf-8 -*-

import numpy as np
import random
from math import exp


class NeuralNetwork(object):

    """
    activations: Representa o valor de ativação de cada neurônio.
    É uma lista de listas, com cada lista representando uma camada diferente.
    Dentro de cada camada, o primeiro elemento da lista é a ativação do bias, e deve ser sempre 1.
    """
    activations = np.array([])
    """
    weights: Representa os pesos entre as camadas.
    É uma lista de matrizes, com cada matriz representando a transição entre duas camadas.
    Cada matriz é composta de n2 linhas e n1 + 1 colunas, onde n2 é o número de neurônios da camada destino,
    e n1 é o número de neurônios da camada fonte (mais 1 para o bias).
    """
    weights = np.array([])

    """Cria uma nova rede neural.
    Se 'configuration' não for especificada, cria uma rede com 'depth' camadas,
    cada uma com 'width' neurônios (mais os neurônios de bias).

    Caso 'configuration' seja especificado, este deve ser uma lista de inteiros,
    onde o tamanho da lista é usado como 'depth', e cada camada tem sua
    'width' específica."""
    def __init__(self, depth, width, configuration=None):

        if not configuration:
            configuration = [width] * depth

        self.numLayers = len(configuration)

        activations = []
        for width in configuration[:-1]:
            activations.append(np.array([1] + [0] * width))
        activations.append(np.array([0] * configuration[self.numLayers-1]))

        self.activations = np.array(activations)

        matrixes = []
        for n1, n2 in zip(configuration, configuration[1:]):
            layer = [[random.random() for i in range(n1 + 1)] for i in range(n2)]
            matrixes.append(layer)
        self.weights = np.array(matrixes)

    def sigmoide(self, value):
        return 1 / 1 + exp(-value)

    def propagate(self, inputs):
        newActivations = [np.array(inputs)]
        for layer in range(0, self.numLayers-1):
            layerValues = self.weights[layer] @ newActivations[layer]
            newActivations.append(np.array([1] + [self.sigmoide(a) for a in layerValues]))
        self.activations = np.array(newActivations)
        return self.activations[self.numLayers-1]

    def backpropagate(self, outputs, expecteds):
        deltas = [np.array([[f - y] for (f, y) in zip(outputs, expecteds)])]
        for layer in range(self.numLayers-2, -1, -1):
            delta = self.weights[layer].transpose() @ deltas[layer-(self.numLayers-2)]
            a = self.activations[layer]
            delta = np.multiply(delta.transpose(), a)
            delta = np.multiply(delta, (1 - a))
            delta = delta.transpose()
            deltas.append(np.array(delta[1:]))
        deltas = np.array(deltas)
        deltas = deltas[::-1]

        newWeights = []
        for layer in range(self.numLayers-2):
            layerWeights = self.weights[layer] + deltas[layer+1] @ self.activations[layer].transpose()
            newWeights.append(np.array(layerWeights))
        return newWeights

    def train(self, instances, className, attributes=None):
        pass

    def evaluate(self, test):
        pass
