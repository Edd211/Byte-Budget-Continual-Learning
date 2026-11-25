import random
import numpy as np

class RandomSelection:
    def select(self, codes, labels, n, classMeans=None):
        indices = random.sample(range(len(codes)), n)
        return [codes[i] for i in indices], [labels[i] for i in indices]

class HerdingSelection:
    def select(self, codes, labels, n, classMeans=None):
        if classMeans is None:
            raise ValueError("Herding Selection needs class means")

        selectedCodes,selectedLabels = [], []
        uniqueClasses = set(labels)
        nPerClass = n // len(uniqueClasses)

        for c in uniqueClasses:
            cIndices = [i for i, l in enumerate(labels) if l == c]
            cCodes = [codes[i] for i in cIndices]

            mean = classMeans[c]
            dist = [np.linalg.norm(np.array(code) - np.array(mean)) for code in cCodes]

            closest = sorted(zip(dist, cIndices))[:nPerClass]
            for _, i in closest:
                selectedCodes.append(codes[i])
                selectedLabels.append(labels[i])

        return selectedCodes, selectedLabels