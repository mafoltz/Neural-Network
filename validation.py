# -*- coding: utf-8 -*-
import random
from collections import Counter
from math import sqrt


class Attribute():

    Numerical = "numerical"
    Categorical = "categorical"

    def __init__(self, value):
        try:
            num = float(value)
            self.type = Attribute.Numerical
            self.value = num
        except ValueError:
            self.type = Attribute.Categorical
            self.value = value

    def __repr__(self):
        return '{0} ({1})'.format(self.value, self.type)

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.type == other.type and self.value == other.value

    def __hash__(self):
        return hash((self.type, self.value))

    def __getitem__(self, key):
        return self.value


class Bootstrapper():

    def __init__(self, instances):
        self.instances = instances

    def bootstrap(self, amount=None):
        if not amount:
            amount = len(self.instances)

        ret = []
        for i in range(amount):
            r = random.randint(0, amount - 1)
            ret.append(self.instances[r])

        return ret


class Measurer():

    def __init__(self, trainedAlgorithm, className):
        self.algorithm = trainedAlgorithm
        self.macroAverage = True
        self.className = className

    def meaure(self, tests):
        predicted = [self.algorithm.evaluate(test, self.className) for test in tests]
        actual = [test[self.className] for test in tests]

        zipped = list(zip(predicted, actual))

        results = [pred == real for (pred, real) in zipped]
        counted = Counter(results)
        totalCount = len(results)

        classes = list(Counter(actual).keys())
        precisions = []
        recalls = []
        for cls in classes:
            truePositive = sum([pred == actual == cls for (pred, actual) in zipped])
            # trueNegative = sum([pred == actual != cls for (pred, actual) in zipped])
            falsePositive = sum([pred == cls != actual for (pred, actual) in zipped])
            falseNegative = sum([actual == cls != pred for (pred, actual) in zipped])

            precisions.append((truePositive, (truePositive + falsePositive)))
            recalls.append((truePositive, (truePositive + falseNegative)))

        if self.macroAverage:

            ratioPrecisions = [right / total if total > 0 else 1 for (right, total) in precisions]
            precision = sum(ratioPrecisions) / len(precisions)
            ratioRecalls = [right / total if total > 0 else 1 for (right, total) in recalls]
            recall = sum(ratioRecalls) / len(recalls)
        else:
            precision = sum([right for (right, _) in precisions]) / sum([total for (_, total) in precisions])
            recall = sum([right for (right, _) in recalls]) / sum([total for (_, total) in recalls])

        dict = {
             'accuracy': (counted.get(True) or 0) / totalCount,
             'F1-measure': 2 * precision * recall / (precision + recall)}

        return dict


class Statistics():

    def __init__(self, values):
        self.values = values
        self.average = sum(values) / len(values)
        self.variance = sum([pow(value - self.average, 2) for value in values])
        self.std_dev = sqrt(self.variance)


class CrossValidator():

    def __init__(self, k, mLAlgorithm):
        self.separator = FoldSeparator(k)
        self.algorithm = mLAlgorithm

    def validate(self, instances, className):
        # Conta as classes de cada instancia
        self.separator.separateFolds(instances, className)

        results = []
        for i in range(len(self.separator.folds)):
            trainer, tester = self.separator.split(i)
            self.algorithm.reset()
            self.algorithm.train(trainer, className)

            measurer = Measurer(self.algorithm, className)
            result = measurer.meaure(tester)

            results.append(result)

        acc = Statistics([result['accuracy'] for result in results])
        f1s = Statistics([result['F1-measure'] for result in results])

        return acc, f1s


class FoldSeparator(object):

    def __init__(self, k):
        self.k = k
        self.folds = [[] for _ in range(k)]

    def separateFolds(self, instances, className):
        classes = [instance[className] for instance in instances]
        counter = Counter(classes)

        self.className = className
        self.classFolds = []

        # Separa as instancias por classes em arrays diferentes
        for classAttribute in counter:
            classInstances = [instance for instance in instances if instance[className] == classAttribute]
            self.classFolds.append(classInstances)

        # Insere uniformemente as instancias de cada classe nos k folds
        for classFold in self.classFolds:
            numOfInstancesPerFold = len(classFold) // self.k
            numOfExtraInstances = len(classFold) % self.k

            for k in range(self.k):
                for _ in range(numOfInstancesPerFold):
                    instance = classFold.pop(random.randint(0, len(classFold)-1))
                    self.folds[k].append(instance)

                # Caso haja n instancias excedentes, insere cada uma nos n primeiros folds
                if k < numOfExtraInstances:
                    instance = classFold.pop(random.randint(0, len(classFold)-1))
                    self.folds[k].append(instance)

    def split(self, atFold):
        tester = self.folds[atFold]
        trainer = []
        for i in range(len(self.folds)):
            if i != atFold:
                trainer += self.folds[i]
        return trainer, tester
