# -*- coding: utf-8 -*-

import numpy as np
import random
from math import exp, log
import copy


DEBUG = False

def printD(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


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
    def __init__(self, configuration, regulation=0, classValues=None):
        self.regulation = regulation
        self.alpha = 0.1
        self.classValues = classValues

        self.numLayers = len(configuration)

        activations = []
        for width in configuration[:-1]:
            activations.append(np.array([1] + [0] * width))
        activations.append(np.array([0] * configuration[self.numLayers-1]))

        self.activations = np.array(activations)

        matrixes = []
        for n1, n2 in zip(configuration, configuration[1:]):
            layer = [[random.random() for i in range(n1 + 1)] for i in range(n2)]
            matrixes.append(np.array(layer))
        self.originalWeights = np.array(matrixes)
        self.weights = copy.deepcopy(self.originalWeights)

    def reset(self):
        self.weights = copy.deepcopy(self.originalWeights)

    def sigmoide(self, value):
        return 1 / (1 + exp(-value))

    def propagate(self, inputs):
        newActivations = [np.array([1] + inputs)]
        printD('input:', [1] + inputs)
        for layer in range(0, self.numLayers-1):
            layerValues = self.weights[layer] @ newActivations[layer]
            printD('enter {}: {}'.format(layer+1, layerValues))

            if layer == self.numLayers-2:
                bias = []
            else:
                bias = [1]
            layerAct = np.array(bias + [self.sigmoide(a) for a in layerValues])
            printD('layer {}: {}'.format(layer+1, layerAct))
            newActivations.append(layerAct)
        self.activations = np.array(newActivations)
        return self.activations[self.numLayers-1]

    def backpropagate(self, outputs, expecteds):
        deltas = [np.array([[f - y] for (f, y) in zip(outputs, expecteds)]).transpose()]
        printD('\ndelta saida:', deltas[0])
        for layer in range(self.numLayers-2, 0, -1):
            delta = np.multiply(self.weights[layer].transpose(), deltas[layer-(self.numLayers-2)])
            delta = [sum(line) for line in delta]

            a = self.activations[layer]
            delta = np.multiply(delta, a)
            delta = np.multiply(delta, (1 - a))
            delta = np.array(delta[1:])
            printD('delta {}: {}'.format(layer, delta))
            deltas.append(delta)
        # deltas = np.array(deltas)
        deltas = deltas[::-1]

        gradients = []
        for layer in range(self.numLayers-1):
            gradientDelta = deltas[layer] * np.array([self.activations[layer]]).transpose()
            printD('gradient delta {}: {}'.format(layer, gradientDelta.transpose()))
            gradients.append(gradientDelta.transpose())
        return gradients

    def error(self, outputs, expectedOutputs):
        return sum([-y*log(fx) - (1-y)*log(1-fx) for fx, y in zip(outputs, expectedOutputs)])

    def networkInputsAndOutputsFrom(self, instances, className, attributes=None):
        if not attributes:
            attributes = list(instances[0].keys())
            attributes.remove(className)

        outputs = []
        inputs = []

        for instance in instances:
            instanceClass = instance[className]
            if not self.classValues:
                if isinstance(instanceClass, list):
                    output = [attr.value for attr in instanceClass]
                else:
                    output = [instanceClass.value]
            else:
                if instanceClass in self.classValues:
                    output = [1.0 if value == instanceClass else 0.0 for value in self.classValues]
                else:
                    print("Class {} not known to neural network".format(instanceClass))
                    exit(-1)
            outputs.append(output)

            attr = [item.value for attribute, item in instance.items() if attribute in attributes]
            if len(attr) == 1 and isinstance(attr[0], list):
                attr = attr[0]
            inputs.append(attr)

        return inputs, outputs

    def gradientsAndErrorFrom(self, inputs, outputs):
        gradients = None
        error = 0

        for i, (input, output) in enumerate(zip(inputs, outputs)):
            printD('\nProcessando exemplo {}'.format(i))

            predictedOutput = self.propagate(input)
            printD('\nSaída predita: {}'.format(predictedOutput))
            printD('Saída esperada: {}'.format(output))

            error += self.error(predictedOutput, output)
            printD('Erro:', error)

            gradientDelta = self.backpropagate(predictedOutput, output)
            if gradients == None:
                gradients = gradientDelta
            else:
                gradients = np.add(gradients, gradientDelta)

        # Average error
        error = error / (i + 1)
        
        return gradients, error

    def regularizedCostFrom(self, instances, error):
        squared = self.weights ** 2
        weightSum = 0
        for layer in squared:
            weightSum += sum([sum(line[1:]) for line in layer])

        regulated = self.regulation * weightSum / (2*len(instances))

        regularizedCost = error + regulated

        printD('\nAccumulated error:', regularizedCost)

        return regularizedCost

    def regulatedGradientsFrom(self, instances, gradients):
        regulatedGradients = []

        for layer in range(self.numLayers-1):
            p = self.regulation * np.array(self.weights[layer])
            for i in range(len(p)):
                p[i][0] = 0

            layerGradient = np.add(gradients[layer], p) / len(instances)
            printD('gradient for layer {}: {}'.format(layer, layerGradient))
            regulatedGradients.append(layerGradient)

        regulatedGradients = np.array([np.array(list) for list in regulatedGradients])
        return regulatedGradients

    def train(self, instances, className, attributes=None):
        inputs, outputs = self.networkInputsAndOutputsFrom(instances, className, attributes)

        gradients, error = self.gradientsAndErrorFrom(inputs, outputs)

        regularizedCost = self.regularizedCostFrom(instances, error)
        self.regularizedCost = regularizedCost

        regulatedGradients = self.regulatedGradientsFrom(instances, gradients)

        self.applyGradients(regulatedGradients)

        return regulatedGradients

    def trainNumerically(self, epsilon, instances, classNames, attributes=None):
        inputs, outputs = self.networkInputsAndOutputsFrom(instances, classNames, attributes)

        def singleErrorFor(layer, row, column):
            error = 0
            for i, (input, output) in enumerate(zip(inputs, outputs)):
                predictedOutput = self.propagate(input)
                error += self.error(predictedOutput, output)
            error = error / len(instances)
            error = self.regularizedCostFrom(instances, error)
            return error

        def errorFor(layer, row, column):
            self.weights[layer][row][column] += epsilon
            largerError = singleErrorFor(layer, row, column)
            self.weights[layer][row][column] -= 2*epsilon
            smallerError = singleErrorFor(layer, row, column)
            self.weights[layer][row][column] += epsilon
            return (largerError - smallerError) / (2*epsilon)

        gradients = []
        for layer, _ in enumerate(self.weights):
            gradients.append([])
            for row, _ in enumerate(self.weights[layer]):
                gradients[layer].append([])
                for column, _ in enumerate(self.weights[layer][row]):
                    gradients[layer][row].append([])
                    gradients[layer][row][column] = errorFor(layer, row, column)

        gradients = np.array([np.array(list) for list in gradients])
        return gradients

    def applyGradients(self, gradients):
        printD()
        for layer in range(self.numLayers-1):
            layerValue = self.weights[layer] - self.alpha * gradients[layer]
            printD('old weight for layer {}: {}'.format(layer, self.weights[layer]))
            printD('new weight for layer {}: {}'.format(layer, layerValue))
            self.weights[layer] = layerValue

    def evaluate(self, test, className):
        inputs, outputs = self.networkInputsAndOutputsFrom([test], className)

        predictedOutputs = self.propagate(inputs[0])

        maxValue = max(predictedOutputs)
        maxIndex = list(predictedOutputs).index(maxValue)

        return self.classValues[maxIndex]
