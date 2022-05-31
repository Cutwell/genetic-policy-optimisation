import numpy as np
import cellpylib as cpl
import random

class AirflowRule(cpl.BaseRule):
    """Simulate positive airflow from windows, updating velocities of air in cells.

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

        if not self.matrix[x][y] == 0:  # if not air, pass
            return neighbourhood[1][
                1
            ]  # return centre neighbour value == this cell value

        # remove neighbours if illegal (wraparound) or are walls
        valid_neighbour_values = np.array([])
        for (i, j), neighbour in np.ndenumerate(neighbourhood):
            if not (
                (x == 0 and i == 0)
                or (x == 9 and i == 2)
                or (y == 0 and j == 0)
                or (y == 9 and j == 2)
            ):
                valid_neighbour_values = np.append(valid_neighbour_values, neighbour)

        # evaluate neighbourhood to calculate pressure level at current position
        airflow_value = sum(
            [val / len(valid_neighbour_values) for val in valid_neighbour_values]
        )  # mean of filtered list, removing None (rejected) neighbours

        # update cell to be mean of neighbourhood
        return airflow_value

class AgentRule(cpl.BaseRule):
    """Simulate positive airflow from windows, updating velocities of air in cells.

    Assumptions:
        * Windows are either negative or positive sources of pressure

    """

    def __init__(
        self, matrix, social_distance, pathogen_info, masking, mask_effectiveness
    ):
        """
        Attributes:
            matrix: map environment indicating walls, to prevent air flow from passing impassable boundaries.
            infection: map indicating airborne pathogens from air layer.
            social distance: target distance to encourage agents to stay away from each other.
            pathogen: infection object describing e.g.: chance of infection from contact.
            masking: if policy is to wear masks.
            mask_effectiveness: efficiency of masks to filter pathogen from entering environment.
        """

        self.matrix = matrix
        self.social_distance = social_distance
        self.pathogen_info = pathogen_info
        self.masking = True if masking == 1 else False
        self.mask_effectiveness = mask_effectiveness

        self.shape_x = self.matrix.shape[0]
        self.shape_y = self.matrix.shape[1]

    def __call__(
        self,
        neighbourhood,
        infection_neighbourhood,
        coords,
        timestamp,
        preview,
        infection_layer,
        agent_neighbours,
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
        # agent_value = neighbourhood[self.social_distance][self.social_distance]
        agent_value = neighbourhood[1][1]

        # ISSUE: air cells do not take action, but they will attempt to propogate their 0 value to the next generation.
        # This can cause agents to vanish, by propogating their 0 value to a cell an agent wants to occupy, removing the agent.
        # Returning None can flag the evolution function to ignore this cell output.
        if agent_value == 0:  # if air, then skip
            return (
                x,
                y,
            ), None  # return None value, indicating should keep existing cell value

        # remove neighbours if illegal (wraparound), edge walls or interior walls
        valid_neighbours = np.array([])
        valid_neighbourhood = neighbourhood[:]
        for (i, j), neighbour in np.ndenumerate(neighbourhood):
            a, b = (x - 1 + i), (y - 1 + j)
            if (
                (a >= 0 and a < self.shape_x and b >= 0 and b < self.shape_y)
                and not (  # bounds checks
                    (x == 0 and i == 0)
                    or (x == 9 and i == 2)
                    or (y == 0 and j == 0)
                    or (y == 9 and j == 2)
                )  # not edge wall or illegal wrap-around
                # and (abs(a-self.social_distance-1) + abs(b-self.social_distance-1)) == 1    # new square is 1 unit from current position
                and self.matrix[a][b] == 0  # not a wall
            ):
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
            if self.masking:
                payload = self.pathogen_info.pathogen_release_payload * (
                    1 - self.mask_effectiveness
                )
            else:
                payload = self.pathogen_info.pathogen_release_payload

            infection_layer[x][y] += payload

        # assemble potential new movement coordinates
        movement_coords = np.empty((0, 2), dtype=int)

        # accept neighbour cell for movement if not occupied and not wall (== is air)
        for (i, j), neighbour in np.ndenumerate(valid_neighbourhood):
            a, b = (x - 1 + i), (y - 1 + j)
            if (
                (a >= 0 and a < self.shape_x and b >= 0 and b < self.shape_y)
                and  # bounds checks
                # (abs(a-self.social_distance-1) + abs(b-self.social_distance-1)) == 1 and    # if in range
                self.matrix[a][b] == 0
                and neighbour == 0
                and preview[a][b]
                == 0  # square is not a wall, not occupied by another agent, won't be occupied by an agent in the current future.
            ):  # if non-air now and in future
                # adjust coordinates from local to global
                movement_coords = np.append(movement_coords, [[a, b]], axis=0)

        # also add current cell as option
        movement_coords = np.append(movement_coords, [[x, y]], axis=0)

        # x_, y_ = random.choice(movement_coords)

        if agent_neighbours.shape[0] == 0:  # if no agents nearby
            x_, y_ = random.choice(movement_coords)
        else:
            # calculate distances between agent and other agents for each movement option
            movement_options = np.empty((0, 3), dtype=int)
            for i, j in movement_coords:
                curr_score = np.Infinity
                # calculate this cell distance to all other agents, score is smallest value
                for i_, j_ in agent_neighbours:
                    dist = abs(i_ - self.social_distance - 1) + abs(
                        j_ - self.social_distance - 1
                    )
                    if curr_score > dist and dist > 0:
                        curr_score = dist

                movement_options = np.append(
                    movement_options, [[i, j, curr_score]], axis=0
                )
            # filter for movement that conforms to social distancing
            conforming_movement = [
                (i, j)
                for i, j, score in movement_options
                if score >= self.social_distance
            ]
            if len(conforming_movement) == 0:
                # find best worst movement
                best, best_score = None, 0
                for i, j, score in movement_options:
                    # if closest agent to current cell is further than current best, is dominating
                    if best_score < score:
                        best = (i, j)
                x_, y_ = best
            else:
                # select random conforming movement
                x_, y_ = random.choice(conforming_movement)

        x_, y_, agent_value = int(x_), int(y_), int(agent_value)

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

        self.shape_x = self.matrix.shape[0]
        self.shape_y = self.matrix.shape[1]

    def __call__(self, neighbourhood, coords, timestamp, r):
        """

        Inputs:
            n: the neighbourhood.

            c: the index of the current cell.

            t: the current timestep.

        Outputs:
            r: the new cell state.

        """

        x, y = coords
        # air_value = neighbourhood[r][r]
        air_value = neighbourhood[1][1]

        if not self.matrix[x][y] == 0:  # if not air, pass
            return (x, y), None  # skip this value

        # remove neighbours if illegal (wraparound) or are walls
        valid_neighbour_coords = np.empty((0, 2), dtype=int)
        for (i, j), neighbour in np.ndenumerate(neighbourhood):
            a, b = (x - 1 + i), (y - 1 + j)
            if (
                (a >= 0 and a < self.shape_x and b >= 0 and b < self.shape_y)
                and not (  # bounds checks
                    (x == 0 and i == 0)
                    or (x == 9 and i == 2)
                    or (y == 0 and j == 0)
                    or (y == 9 and j == 2)
                )
                # and (abs(a-r-1) + abs(b-r-1)) == 1
                and self.matrix[a][b] == 0
            ):
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
