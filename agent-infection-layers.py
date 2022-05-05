from platypus import NSGAII, NSGAIII, Problem, Integer, Real, nondominated
from IPython.display import display, Markdown, HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cellpylib as cpl
import random
import logging

import cellpylibhack.functions2d as cplhack

def evolveParallel(
    layer1, layer2, timesteps, apply_rule1, apply_rule2, r=1, neighbourhood="Moore"
):
    """Evolves two cellular automaton in parallel. Generates new generation for first layer then uses the output to generate 2nd layer."""
    assert (
        layer1.shape == layer2.shape,
        "Cannot evolve different layer shapes in parallel.",
    )

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
    return _evolveParallel_fixed(
        layer1,
        layer2,
        timesteps,
        apply_rule1,
        apply_rule2,
        neighbourhood,
        rows,
        cols,
        neighbourhood_indices,
        von_neumann_mask,
        cell_indices,
        cell_idx_to_neigh_idx,
    )


def _evolveParallel_fixed(
    layer1,
    layer2,
    timesteps,
    apply_rule1,
    apply_rule2,
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

    initial_conditions2 = layer2[-1]
    array2 = np.zeros((timesteps, rows, cols), dtype=layer2.dtype)
    array2[0] = initial_conditions2

    for t in range(1, timesteps):
        cell_layer1 = array1[t - 1]
        cell_layer2 = array2[t - 1]

        for row, cell_row1 in enumerate(cell_layer1):
            cell_row2 = cell_layer2[row]

            for col, cell1 in enumerate(cell_row1):
                cell2 = cell_row2[col]

                n1 = cplhack._get_neighbourhood(
                    cell_layer1,
                    neighbourhood_indices,
                    row,
                    col,
                    neighbourhood,
                    von_neumann_mask,
                )

                n2 = cplhack._get_neighbourhood(
                    cell_layer2,
                    neighbourhood_indices,
                    row,
                    col,
                    neighbourhood,
                    von_neumann_mask,
                )

                # NOTE: allow rule to return set of coordinates and new state.
                # This allows agents to move in environment by triggering rule from current position then setting next self position in different place - movement.

                # array1 is agent layer, so needs infection layer neighbours for agent location
                coords, value = apply_rule1(n1, n2, (row, col), t, array1[t], array2[t])

                if value:   # if non-None value
                    x, y = coords
                    array1[t][x][y] = value

                # array2 is infection layer, so doesn't need infection layer neighbours for agent location, nor preview of next layer
                coords, value = apply_rule2(n2, (row, col), t)

                if value:   # if non-None value
                    x, y = coords
                    # add to new position, allow multiple cells to merge into a single point
                    array2[t][x][y] += value

    return np.concatenate((layer1, array1[1:]), axis=0), np.concatenate((layer2, array2[1:]), axis=0)


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
        infection_neighbourhood,
        coords,
        timestamp,
        preview,
        infection_layer,
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
                        if random.random() < self.pathogen_info.infection_from_contact
                        else agent_value
                    )  # infected if pass random chance check
            
            infection_in_cell = infection_neighbourhood[1][1]
            if infection_in_cell > 0:
                # infected if pass random chance, modified for % infection relative to max pollutants requried for max airborne infection chance
                agent_value = (
                    2
                    if random.random()
                    < (
                        self.pathogen_info.infection_from_max_pollutants
                        * (infection_in_cell / self.pathogen_info.max_pollutants)
                    )
                    else agent_value
                )
        
        else:  # agent is infected
            # pollute infection layer at current cell
            infection_layer[x][y] += self.pathogen_info.pathogen_release_payload

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

class InfectionRule(cpl.BaseRule):
    """Simulate positive airflow from windows, updating velocities of air in cells.

    Assumptions:
        * Windows are either negative or positive sources of pressure

    """

    def __init__(self, matrix, airflow, pathogen_info):
        """
        Attributes:
            matrix: map environment indicating walls, to prevent air flow from passing impassable boundaries.
        """

        self.matrix = matrix
        self.airflow = airflow
        self.pathogen_info = pathogen_info

    def __call__(self, neighbourhood, coords, timestamp):
        """

        Inputs:
            n: the neighbourhood.

            c: the index of the current cell.

            t: the current timestep.

        Outputs:
            r: the new cell state.

        """

        x, y = coords
        air_value = neighbourhood[1][1]

        if not self.matrix[x][y] == 0:  # is window or is wall
            return (x, y), None  # skip this value

        # remove neighbours if illegal (wraparound) or are walls
        valid_neighbour_coords = np.empty((0, 2), dtype=int)
        for (i, j), neighbour in np.ndenumerate(neighbourhood):
            a, b = (x-1+i), (y-1+j)
            if not (
                (x == 0 and i == 0)
                or (x == 9 and i == 2)
                or (y == 0 and j == 0)
                or (y == 9 and j == 2)
            ) and self.matrix[a][b] == 0:
                valid_neighbour_coords = np.append(
                    valid_neighbour_coords, [[a, b]], axis=0
                )

        # get coords of cell with greatest negative pressure
        x_, y_ = x, y

        # dominating search valid neighbours
        for a, b in valid_neighbour_coords:
            if (
                self.airflow[a][b] < self.airflow[x_][y_]
            ):  # if air pressure dominates (is lower) than current best, set as new best
                x_, y_ = a, b

        # reduce pathogen by decay rate
        air_value -= self.pathogen_info.airborne_decay

        if air_value < 0:
            air_value = 0

        # move airborne contents from current cell to new cell, following path of lowest pressure
        # subtracts air_value from current coords, adds air_value to new target coords.
        return (x_, y_), air_value


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

    def __call__(self, neighbourhood, coords, timestamp):
        """ 

        Inputs:
            neighbourhood: the neighbourhood.

            coords: the index of the current cell.

            timestamp: the current timestep.

        Outputs:
            airflow_value: the new cell state.

        """

        x, y = coords

        if not self.matrix[x][y] == 0:    # is window or is wall
            return neighbourhood[1][1]  # return centre neighbour value == this cell value

        # remove neighbours if illegal (wraparound) or are walls
        valid_neighbour_values = np.array([])
        for (i, j), neighbour in np.ndenumerate(neighbourhood):
            if not ((x == 0 and i == 0) or (x == 9 and i == 2) or (y == 0 and j == 0) or (y == 9 and j == 2)):
                valid_neighbour_values = np.append(
                    valid_neighbour_values, neighbour
                )

        # evaluate neighbourhood to calculate pressure level at current position
        airflow_value = sum([val/len(valid_neighbour_values) for val in valid_neighbour_values]) # mean of filtered list, removing None (rejected) neighbours

        # update cell to be mean of neighbourhood
        return airflow_value


environment = np.array(
    [
        [1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 2],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2],
        [1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1],
    ]
)  # encoded environment map (walls and windows) (air=0, walls=1, windows=2)

dimensions = environment.shape

agentPopSize = 20  # number of agents in simulation

init_infected_ratio = 0.1

pathogen_info = PathogenInfo(
    infection_from_contact=0.1,
    infection_from_max_pollutants=0.4,
    max_pollutants=20,
    airborne_decay=1,
    pathogen_release_payload=10,
)

social_distance = 1

# calculate room airflow
airflow = cpl.init_simple2d(*dimensions)

windowMap = np.empty((0, 2), dtype=int)
for (x, y), value in np.ndenumerate(environment):
    if value == 2:  # if window, add coords to window list
        windowMap = np.append(windowMap, [[x, y]], axis=0)

# set positive and negative air sources
for x, y in windowMap:
    if environment[x][y] == 2:
        if y < 5:  # positive incoming pressure for windows in top half
            pressure = 10000
        else:  # negative incoming pressure for windows in bottom half
            pressure = -10000

        airflow[:, x, y] = pressure

airflow_ca = cpl.evolve2d(airflow, timesteps=cpl.until_fixed_point(), apply_rule=AirflowRule(environment), neighbourhood="Moore")

airflow = airflow_ca[-1]

print("Number of timesteps to reach fixed point: %s" % len(airflow_ca))
cpl.plot2d_animate(airflow_ca, show_grid=True, title="Airflow", interval=50, colormap='Blues')

agent = cpl.init_simple2d(*dimensions)
infection = cpl.init_simple2d(*dimensions)

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
    agent[:, x, y] = 2

for x, y in [coord for coord in spawnpoints[init_infected+1:init_healthy+1]]:
    agent[:, x, y] = 1

agent_ca, infection_ca = evolveParallel(
    agent,
    infection,
    timesteps=200,
    apply_rule1=AgentRule(environment, social_distance, pathogen_info),
    apply_rule2=InfectionRule(environment, airflow, pathogen_info),
    neighbourhood="Moore",
)

cpl.plot2d_animate(agent_ca, show_grid=True, title="Agents", interval=100, colormap='Accent')
cpl.plot2d_animate(infection_ca, show_grid=True, title="Infection airflow", interval=100, colormap='YlGn')