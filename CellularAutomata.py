import random

import matplotlib.pyplot as plt
import numpy as np


class cellularAutomataHandler:
    """1D CA generator
    """

    def __init__(self, radius=1, rule=184, envDims=(100, 50)):
        """Initialize the 1D GA generator with the given parameters

        Keyword arguments:
        radius -- int, desired radius for the GA (default 1)
        rule -- int, desired rule to apply (default 184)
        envDims -- tuple (int, int), dimension of the environment
        """
        self.radius = radius
        self.envDims = envDims
        self.environment = np.zeros(envDims, dtype=np.int8)
        self.change_rule(rule)

    def change_rule(self, rule):
        """Change rule to run CA

        Keyword arguments:
        rule -- int, desired rule to apply (required)
        """
        self.rule = rule
        tmp = [int(x) for x in bin(self.rule)[2:]]
        self.out = [0] * (2**(2*self.radius+1))
        tmp = list(reversed(tmp))
        self.out[0:len(tmp)] = tmp

    def apply_func(self, row):
        """Apply global function one time.

        Keyword arguments:
        row -- int, desired row to update (required)

        Returns:
        output -- update CA's states after applying global rule
        """
        rowlen = len(row)
        result = []
        # Focusing on each cellular automata
        for i in range(0, rowlen):

            if i - self.radius < 0:
                # If there are no neighbours on the left,
                # create new ones

                how_many = abs(i-self.radius)
                secondPart = 2*self.radius + 1 - how_many
                new_data = []
                for i in range(how_many):
                    new_data.append(random.randint(0, 1))

                num = np.concatenate((new_data,
                                      row[0:secondPart]), axis=0)
            elif i + self.radius + 1 > rowlen:
                # If there are no neighbours on the right,
                # assume the state is 0 for the missing
                # neighbours

                secondPart = (i+self.radius) % (rowlen-1)
                new_data = [0]*secondPart
                num = np.concatenate((row[i-self.radius:rowlen],
                                      new_data), axis=0)
            else:
                num = row[i-self.radius:i+self.radius+1]

            num = num.tolist()
            ind = int("".join(str(i) for i in num), 2)
            result.append(self.out[ind])
        return result

    def compute_step(self, row):
        """Apply the global rule on the given row.

        Keyword arguments:
        row -- int, desired row to use to update (required)
        """
        if row > 0:
            self.environment[row] = self.apply_func(self.environment[row-1])

    def compute_all(self):
        """Compute all the steps using the desired rule.
        """
        for i in range(len(self.environment)):
            self.compute_step(i)

    def plot_all(self, path=None):
        """Plot or save the generated time representation of the evolution
        of the CA.

        Keyword arguments:
        path -- String, path where to save file, if None, the image
                will be shown instead (default None)
        """
        plt.imshow(self.environment, cmap='Greys',  interpolation='nearest')
        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=600)

    def generate_starting_point(self):
        """Randomly Generate the initial row
        for the CA.
        """
        rl = []
        for _ in range(self.envDims[1]):
            rl.append(random.randint(0, 1))
        self.environment[0] = rl

    def execute(self, path=None, generate_point=True):
        """Generate starting point and compute the time evolution
        of the CA, in the end, the resulting image representation
        will be saved or shown.

        Keyword arguments:
        path -- String, path where to save file, if None, the image
                will be shown instead (default None)
        generate_point -- Boolean, if True, the first true will
                          be randomly generated, otherwise the first
                          row will remain unchanged
        """
        if generate_point:
            self.generate_starting_point()
        self.compute_all()
        self.plot_all(path)


CA_hanlder = cellularAutomataHandler(envDims=(50, 100))
CA_hanlder.generate_starting_point()

for i in range(256):
    CA_hanlder.change_rule(i)
    CA_hanlder.execute("Images/test"+str(i)+".png", generate_point=False)
