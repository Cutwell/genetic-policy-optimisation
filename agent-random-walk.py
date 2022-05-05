from platypus import NSGAII, NSGAIII, Problem, Integer, Real, nondominated
from IPython.display import display, Markdown, HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cellpylib as cpl
import random
import logging

import cellpylibhack.functions2d as cplhack


def evolveComplex(layer1, timesteps, apply_rule1, r=1, neighbourhood="Moore"):

    von_neumann_mask = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
    for i in range(len(von_neumann_mask)):
        mask_size = np.absolute(r - i)
        von_neumann_mask[i][:mask_size] = 1
        if mask_size != 0:
            von_neumann_mask[i][-mask_size:] = 1

    _, rows, cols = layer1.shape
    neighbourhood_indices = cplhack._get_neighbourhood_indices(rows, cols, r)

    cell_indices = cplhack._get_cell_indices((rows, cols))
    cell_idx_to_neigh_idx = cplhack._get_cell_indices_to_neighbourhood_indices(
        cell_indices, rows, cols, r
    )

    # NOTE: to simplify total copied code, can only run this as fixed number of epochs.
    # TODO: add dynamic evolution.
    return _evolveComplex_fixed(
        layer1,
        timesteps,
        apply_rule1,
        neighbourhood,
        rows,
        cols,
        neighbourhood_indices,
        von_neumann_mask,
        cell_indices,
        cell_idx_to_neigh_idx,
    )


def _evolveComplex_fixed(
    layer1,
    timesteps,
    apply_rule1,
    neighbourhood,
    rows,
    cols,
    neighbourhood_indices,
    von_neumann_mask,
    cell_indices,
    cell_idx_to_neigh_idx,
):
    """
    Evolves the 2 layers of cellular automaton in parallel for a fixed of timesteps.


    """
    initial_conditions1 = layer1[-1]
    array1 = np.zeros((timesteps, rows, cols), dtype=layer1.dtype)
    array1[0] = initial_conditions1

    for t in range(1, timesteps):
        cell_layer1 = array1[t - 1]

        for row, cell_row1 in enumerate(cell_layer1):

            for col, cell1 in enumerate(cell_row1):

                n1 = cplhack._get_neighbourhood(
                    cell_layer1,
                    neighbourhood_indices,
                    row,
                    col,
                    neighbourhood,
                    von_neumann_mask,
                )

                # NOTE: allow rule to return set of coordinates and new state.
                # This allows agents to move in environment by triggering rule from current position then setting next self position in different place - movement.

                # array1 is agent layer, so needs infection layer neighbours for agent location
                coords, value = apply_rule1(n1, (row, col), t, array1[t])

                if value:   # if non-None value
                    x, y = coords
                    array1[t][x][y] = value

    return np.concatenate((layer1, array1[1:]), axis=0)


class AgentRule(cpl.BaseRule):
    """Simulate positive airflow from windows, updating velocities of air in cells.

    Assumptions:
        * Windows are either negative or positive sources of pressure

    """

    def __init__(self, matrix, social_distance, pathogen_info):
        """
        Attributes:
            matrix: map environment indicating walls, to prevent air flow from passing impassable boundaries.
            infection: map indicating airborne pathogens from air layer.
            social distance: target distance to encourage agents to stay away from each other.
            pathogen: infection object describing e.g.: chance of infection from contact.
        """

        self.matrix = matrix
        self.social_distance = social_distance
        self.pathogen_info = pathogen_info

    def __call__(
        self,
        neighbourhood,
        coords,
        timestamp,
        preview,
    ):
        """If called for an agent cell, update.

        Agent behaviour:
            1. Check surrounding cells for direct contact infected agent.
            2. Check infection air layer for pathogen on current cell.
            3. If infected, increase pathogen density level for current cell in infection air layer.
            4. Stay in same position, or select a neighbour cell that isn't a wall, doesn't have an agent occupying it, and won't have an agent occupying it next generation (via preview).

        Inputs:
            neighbourhood: the neighbourhood.

            infection neighbourhood: neighbourhood of agent in infection layer.

            coords: the index of the current cell.

            timestamp: the current timestep.

            preview: preview of currently constructed next generation.

            infection: current infection layer object.

        Outputs:
            r: the new cell state.

        """

        x, y = coords
        agent_value = neighbourhood[1][1]

        # ISSUE: air cells do not take action, but they will attempt to propogate their 0 value to the next generation.
        # This can cause agents to vanish, by propogating their 0 value to a cell an agent wants to occupy, removing the agent.
        # Returning None can flag the evolution function to ignore this cell output.  
        if agent_value == 0:  # if air, then skip
            return (x, y), None  # return None value, indicating should keep existing cell value

        # remove neighbours if illegal (wraparound), edge walls or interior walls
        valid_neighbours = np.array([])
        valid_neighbourhood = neighbourhood[:]
        for (i, j), neighbour in np.ndenumerate(neighbourhood):
            a, b = (x-1+i), (y-1+j)
            if not (
                (x == 0 and i == 0)
                or (x == 9 and i == 2)
                or (y == 0 and j == 0)
                or (y == 9 and j == 2)
            ) and self.matrix[a][b] == 0:
                valid_neighbours = np.append(valid_neighbours, neighbour)
            else:
                valid_neighbourhood[i][j] = -1

        if agent_value != 2:  # no use checking for infection if already infected
            # check for contact infection
            for neighbour in valid_neighbours:
                if neighbour == 2:  # if infected neighbour
                    agent_value = (
                        2
                        if random.random() <= self.pathogen_info.infection_from_contact
                        else agent_value
                    )  # infected if pass random chance check

        # assemble potential new movement coordinates
        movement_coords = np.empty((0, 2), dtype=int)

        # accept neighbour cell for movement if not occupied and not wall (== is air)
        for (i, j), neighbour in np.ndenumerate(valid_neighbourhood):
            a, b = (x-1+i), (y-1+j)
            if self.matrix[a][b] == 0 and neighbour == 0 and preview[a][b] == 0:  # if non-air now and in future
                # adjust coordinates from local to global
                movement_coords = np.append(movement_coords, [[a, b]], axis=0)

        # also add current cell as option
        movement_coords = np.append(movement_coords, [[x, y]], axis=0)

        # randomly select new coordinates to move agent to
        x_, y_ = random.choice(movement_coords)

        return (x_, y_), agent_value


class PathogenInfo:
    def __init__(
        self,
        infection_from_contact,
        infection_from_max_pollutants,
        max_pollutants,
        airborne_decay,
        pathogen_release_payload,
    ):
        """
        Attributes:
            infection from contact: chance of infection from contact with infected agent.
            infection from max pollutants: chance of infection from being exposed to maximum airborne pollutant concentration.
            max pollutants: airborne pollutants required to have full chance of infection_from_max_pollutants.
            airborne decay: rate of decay of pollutants from air.
            pathogen release payload: number of pollutants released by agent per epoch.

        """

        self.infection_from_contact = infection_from_contact
        self.infection_from_max_pollutants = infection_from_max_pollutants
        self.max_pollutants = max_pollutants
        self.airborne_decay = airborne_decay
        self.pathogen_release_payload = pathogen_release_payload


environment = np.array(
    [
        [1, 2, 1, 1, 1, 2, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 2],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [2, 0, 0, 0, 1, 0, 0, 0, 0, 2],
        [1, 1, 2, 1, 1, 2, 1, 2, 1, 1],
    ]
)  # encoded environment map (walls and windows) (air=0, walls=1, windows=2)

dimensions = environment.shape

agentPopSize = 10  # number of agents in simulation

init_infected_ratio = 0.1

pathogen_info = PathogenInfo(
    infection_from_contact=0.8,
    infection_from_max_pollutants=0.4,
    max_pollutants=50,
    airborne_decay=-2,
    pathogen_release_payload=10,
)

social_distance = 1

cellular_automaton = cpl.init_simple2d(*dimensions)

# get list of valid non-wall cells to spawn agents in
spawnpoints = np.empty((0, 2), dtype=int)
for (x, y), value in np.ndenumerate(environment):
    if value == 0:  # if air, then free for spawning
        spawnpoints = np.append(spawnpoints, [[x, y]], axis=0)

# random shuffle coords and extract popSize list of spawnpoints
np.random.shuffle(spawnpoints)

init_infected = int(agentPopSize * init_infected_ratio)
init_healthy = agentPopSize - init_infected

for x, y in [coord for coord in spawnpoints[:init_infected]]:
    cellular_automaton[:, x, y] = 2

for x, y in [coord for coord in spawnpoints[init_infected+1:init_healthy+1]]:
    cellular_automaton[:, x, y] = 1

agent_ca = evolveComplex(
    cellular_automaton,
    timesteps=50,
    apply_rule1=AgentRule(environment, social_distance, pathogen_info),
    neighbourhood="Moore",
)

cpl.plot2d_animate(agent_ca)