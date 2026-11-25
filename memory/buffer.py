import random

class MemoryBuffer():
    def __init__(self, byteBudget, codeSize, selectionStrategy):
        self.byteBudget = byteBudget
        self.codeSize = codeSize
        self.maxCodes = byteBudget // codeSize
        self.selectionStrategy = selectionStrategy
        self.codes = []
        self.labels = []
        self.classMeans = {}
        #additional stats below
        self.selectionCount = 0
        self.addCount = 0
        self.classDistribution = {}

    def updateClassMeans(self, classMeans):
        self.classMeans = classMeans

    #adds code & label, uses selection strategy if full
    def add(self, code, label):
        if self.isFull():
            self.codes, self.labels = self.selectionStrategy.select(self.codes, self.labels, self.maxCodes-1, classMeans=self.classMeans)
            self.selectionCount += 1
        self.codes.append(code)
        self.labels.append(label)
        self.addCount += 1

        self.classDistribution[label] = self.classDistribution.get(label, 0) + 1

    #gets random n codes & labels
    def retrieve(self, n):
        indices = random.sample(range(len(self.codes)), n)
        return [self.codes[i] for i in indices], [self.labels[i] for i in indices]

    def isFull(self):
        return len(self.codes) >= self.maxCodes

    def getStats(self):
        return {
            "byteBudget": self.byteBudget,
            "codeSize": self.codeSize,
            "maxCodes": self.maxCodes,
            "currentSize": len(self.codes),
            "selectionCount": self.selectionCount,
            "addCount": self.addCount,
            "classDistribution": self.classDistribution,
            "selection strategy": self.selectionStrategy.__class__.__name__,
        }

    def reset(self):
        self.codes = []
        self.labels = []
        self.classMeans = {}
        self.selectionCount = 0
        self.addCount = 0
        self.classDistribution = {}