# -*- coding: utf-8 -*-

import numpy as np
import random
from math import exp, log


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
    def __init__(self, depth, width, configuration=None, regulation=0):
        self.regulation = regulation
        self.alpha = 0.1

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
        return 1 / (1 + exp(-value))

    def propagate(self, inputs):
        newActivations = [np.array([1] + inputs)]
        print('input:', [1] + inputs)
        for layer in range(0, self.numLayers-1):
            layerValues = self.weights[layer] @ newActivations[layer]
            print('enter {}: {}'.format(layer+1, layerValues))

            if layer == self.numLayers-2:
                bias = []
            else:
                bias = [1]
            layerAct = np.array(bias + [self.sigmoide(a) for a in layerValues])
            print('layer {}: {}'.format(layer+1, layerAct))
            newActivations.append(layerAct)
        self.activations = np.array(newActivations)
        return self.activations[self.numLayers-1]

    def backpropagate(self, outputs, expecteds):
        deltas = [np.array([[f - y] for (f, y) in zip(outputs, expecteds)]).transpose()]
        print('\ndelta saida:', deltas[0])
        for layer in range(self.numLayers-2, 0, -1):
            delta = np.multiply(self.weights[layer].transpose(), deltas[layer-(self.numLayers-2)])
            delta = [sum(line) for line in delta]

            a = self.activations[layer]
            delta = np.multiply(delta, a)
            delta = np.multiply(delta, (1 - a))
            delta = np.array(delta[1:])
            print('delta {}: {}'.format(layer, delta))
            deltas.append(delta)
        # deltas = np.array(deltas)
        deltas = deltas[::-1]

        gradients = []
        for layer in range(self.numLayers-1):
            gradientDelta = deltas[layer] * np.array([self.activations[layer]]).transpose()
            print('gradient delta {}: {}'.format(layer, gradientDelta.transpose()))
            gradients.append(gradientDelta.transpose())
        return gradients

    def error(self, outputs, expectedOutputs):
        return sum([-y*log(fx) - (1-y)*log(1-fx) for fx, y in zip(outputs, expectedOutputs)])

    def train(self, instances, className, attributes=None):
        if not attributes:
            attributes = list(instances[0].keys())
            attributes.remove(className)

        outputs = []
        inputs = []
        for instance in instances:
            output = instance[className]
            if not isinstance(output, list):
                output = [output]
            outputs.append(output)
            attr = [v for k, v in instance.items() if k in attributes]
            if len(attr) == 1 and isinstance(attr[0], list):
                attr = attr[0]
            inputs.append(attr)

        print(outputs)
        print(inputs)

        gradients = None
        error = 0
        for i, (input, output) in enumerate(zip(inputs, outputs)):
            print('\nProcessando exemplo {}'.format(i))
            predictedOutput = self.propagate(input)
            print('\nSaída predita: {}'.format(predictedOutput))
            print('Saída esperada: {}'.format(output))
            error += self.error(predictedOutput, output)
            print('Erro:', error)
            gradientDelta = self.backpropagate(predictedOutput, output)
            if not gradients:
                gradients = gradientDelta
            else:
                gradients = np.add(gradients, gradientDelta)

        squared = self.weights ** 2
        weightSum = 0
        for layer in squared:
            weightSum += sum([sum(line[1:]) for line in layer])

        regulated = self.regulation * weightSum / (2*len(instances))

        error = error / len(instances) + regulated

        print('\nAccumulated error:', error)

        regulatedGradients = []
        for layer in range(self.numLayers-1):
            p = self.regulation * np.array(self.weights[layer])
            for i in range(len(p)):
                p[i][0] = 0

            layerGradient = np.add(gradients[layer], p) / len(instances)
            print('gradient for layer {}: {}'.format(layer, layerGradient))
            regulatedGradients.append(layerGradient)

        print()
        for layer in range(self.numLayers-1):
            layerValue = self.weights[layer] - self.alpha * regulatedGradients[layer]
            print('old weight for layer {}: {}'.format(layer, self.weights[layer]))
            print('new weight for layer {}: {}'.format(layer, layerValue))
            self.weights[layer] = layerValue

    def evaluate(self, test):
        pass


def example1():
    t = NeuralNetwork(0, 0, [1, 2, 1])

    # print(t.activations)
    t.weights[0] = np.array([[0.4, 0.1], [0.3, 0.2]])
    t.weights[1] = np.array([[0.7, 0.5, 0.6]])

    instance1 = {'x': 0.13, 'y': 0.9}
    instance2 = {'x': 0.42, 'y': 0.23}

    t.train([instance1, instance2], 'y')


def example2():
    t = NeuralNetwork(0, 0, [2, 4, 3, 2], regulation=0.25)

    # print(t.activations)
    t.weights[0] = np.array([[0.42, 0.15, 0.40],
                             [0.72, 0.10, 0.54],
                             [0.01, 0.19, 0.42],
                             [0.30, 0.35, 0.68]])

    t.weights[1] = np.array([[0.21, 0.67, 0.14, 0.96, 0.87],
                             [0.87, 0.42, 0.20, 0.32, 0.89],
                             [0.03, 0.56, 0.80, 0.69, 0.09]])

    t.weights[2] = np.array([[0.04, 0.87, 0.42, 0.53],
                             [0.17, 0.10, 0.95, 0.69]])

    instance1 = {'x': [0.32, 0.68], 'y': [0.75, 0.98]}
    instance2 = {'x': [0.83, 0.02], 'y': [0.75, 0.28]}

    t.train([instance1, instance2], 'y')


if __name__ == '__main__':
    example2()
