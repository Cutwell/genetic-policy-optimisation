import cellpylib as cpl
import numpy as np
from statistics import mean
np.random.seed(0)

class AirflowRule(cpl.BaseRule):
    """ Simulate positive airflow from windows, updating velocities of air in cells.

    Assumptions:
        * Windows are either negative or positive sources of pressure
    
    """

    def __init__(self, matrix):
        """
        Attributes:
            matrix: map environment indicating walls, to prevent air flow from passing impassable boundaries.
        """
        
        self.matrix = matrix

    def __call__(self, n, c, t):
        """ 

        Inputs:
            n: the neighbourhood.

            c: the index of the current cell.

            t: the current timestep.

        Outputs:
            r: the new cell state.

        """

        x, y = c

        if not self.matrix[x][y] == 0:    # is window or is wall
            return n[1][1]  # return centre neighbour value == this cell value

        # remove neighbours if illegal (wraparound) or are walls
        for (i, j), neighbour in np.ndenumerate(n):
            if (x == 0 and i == 0) or (x == 9 and i == 2) or (y == 0 and j == 0) or (y == 9 and j == 2):
                n[i][j] = None  # reject cell

        # evaluate neighbourhood to calculate pressure level at current position
        filtered = [i for i in np.array(n).reshape([1, 9])[0] if i]
        r = sum([val/len(filtered) for val in filtered]) # mean of filtered list, removing None (rejected) neighbours

        # update cell to be mean of neighbourhood
        return r

cellular_automaton = cpl.init_simple2d(10, 10)

# set positive and negative air sources
cellular_automaton[:, [0], [1]] = 10000
cellular_automaton[:, [8], [9]] = -10000

matrix = [
    [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 2],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

ca = cpl.evolve2d(cellular_automaton, timesteps=cpl.until_fixed_point(), apply_rule=AirflowRule(matrix), neighbourhood="Moore")

print("Number of timesteps to reach fixed point: %s" % len(ca))
cpl.plot2d_animate(ca)
