# -*- coding: utf-8 -*-

import numpy as np


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

        self.activations = np.array([[1] + [0] * depth for depth in configuration])

        matrixes = []
        for n1, n2 in zip(configuration, configuration[1:]):
            line = [0] * (n1 + 1)
            matrix = [line] * n2
            matrixes.append(matrix)
        self.weights = np.array(matrixes)

    def train(self, instances, className, attributes=None):
        pass

    def evaluate(self, test):
        pass
