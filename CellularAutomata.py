import numpy as np
import matplotlib.pyplot as plt
import random

class cellularAutomata:
    radius = 1
    rule = 0

    def __init__(self, radius, rule, envDims):
        self.radius = radius
        self.rule = rule
        self.environment = np.zeros(envDims, dtype=np.int8)

        tmp = [int(x) for x in bin(self.rule)[2:]]
        self.out = [0]* (2**(2*radius+1))
        self.out[0:len(tmp)] = tmp

    def apply_func(self, row):
        rowlen = len(row)
        result = []
        for i in range(0, rowlen):
            if i - self.radius < 0:
                firstPart = (i-self.radius) % (rowlen-1)
                secondPart = 2*self.radius + 1 - ((rowlen-1)-firstPart)
                num = np.concatenate((row[firstPart:(rowlen-1)],
                                      row[0:secondPart]), axis=0)
            elif i + self.radius > rowlen:

                secondPart = (i+self.radius) % (rowlen-1)
                num = np.concatenate((row[i:(rowlen-1)],
                                      row[0:secondPart]), axis=0)
            else:
                num = row[i-self.radius:i+self.radius]
            num = num.tolist()
            ind = int("".join(str(i) for i in num), 2)
            result.append(self.out[ind])
        return result

    def compute_step(self, row):
        if row > 0:
            self.environment[row] = self.apply_func(self.environment[row-1])

    def compute_all(self):
        for i in range(len(self.environment)):
            self.compute_step(i)

    def plot_all(self):
        plt.imshow(self.environment, cmap='Greys',  interpolation='nearest')
        plt.show()


size = (100,50)
a = cellularAutomata(1, 184, size)

rl = []
for i in range(size[1]):
    rl.append(random.randint(0,1))
a.environment[0] = rl


a.compute_all()
a.plot_all()
