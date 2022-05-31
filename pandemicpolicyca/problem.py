from platypus import *
import matplotlib.pyplot as plt
import numpy as np
import cellpylib as cpl
import random
from collections import Counter
import itertools

from pandemicpolicyca import AirflowRule, AgentRule, InfectionRule

class AutomataOptimisation(Problem):
    def __init__(
        self,
        environment,
        pathogen_info,
        n_objectives=3,
        init_infected_ratio=0.1,
        agentPopSize=10,
        mask_effectiveness=0.5,
        epochs=50,
        verbose=0,
        evaluating=1,
    ):
        """Optimise social and physical factors to minimise infection rate in enclosed environment.

        NOTE: '#' commented decision variables / objectices are not implemented at present. When implementing, adjust all indexing for accesses of solutions.

        Decision variables (indexed):
            [0]: face mask toggle (boolean).
            [1]: social distancing (agents are disuaded from moving into spaces that violate social distancing, and will move away fron others if made to violate) (integer range 0-5 metres).
            [2-N]: window open/closed (boolean).

        Objectives:
            [0]: Infection count (integer).
            [1]: Inverse social distance (integer).
            [2]: Inverse tempurature (float).

        """

        self.environment = environment

        # map solution features to map coordinates (for physical feature toggles)
        self.windowMap = np.empty((0, 2), dtype=int)
        for (x, y), value in np.ndenumerate(self.environment):
            if value == 2:  # if window, add coords to window list
                self.windowMap = np.append(self.windowMap, [[x, y]], axis=0)

        # decision variables = number of windows (physical) + social distance (social)
        self.physical_nvars = self.windowMap.shape[0]
        self.social_nvars = 2  # masking, social distancing
        decision_variables = self.physical_nvars + self.social_nvars
        objectives = n_objectives

        super(AutomataOptimisation, self).__init__(decision_variables, objectives)

        # type format: [<social vars>, <physical vars (windows)>]
        self.types = [
            Integer(0, 1),
            Integer(0, 3),
        ]  # masking, social distancing, ...
        for _ in range(self.windowMap.shape[0]):
            self.types.append(Integer(0, 1))  # open/closed window

        self.dimensions = self.environment.shape
        self.evaluating = evaluating
        self.verbose = verbose
        self.agentPopSize = agentPopSize  # number of agents in simulation
        self.init_infected_ratio = init_infected_ratio
        self.epochs = (
            epochs  # number of simulation epochs for testing effectiveness of solution
        )
        self.max_airflow_epochs = 1000  # maximum epochs to allow airflow to run for before compromising with current best approximation
        self.pathogen_info = pathogen_info
        self.mask_effectiveness = mask_effectiveness  # efficiency of masking

    def evaluate(self, solution):
        """Evaluate using lazy or expensive methods."""

        if self.verbose > 0:
            stats = self.full_evaluate(solution)

            return stats

        else:
            self.full_evaluate(solution)

        # if self.evaluating == 0:
        #    self.lazy_evaluate(solution)
        # elif self.evaluating == 1:
        #    self.full_evaluate(solution)

    def full_evaluate(self, solution):
        """Expensive evaluation. Run full simulation with CA."""

        x = solution.variables[:]

        params = {
            "masking": x[0],
            "social_distance": x[1],
            "physical_vars": x[self.social_nvars :],
        }

        matrix, airflow = self.get_environment(params["physical_vars"])
        agents = self.get_agents(matrix)

        if self.verbose > 0:
            x, y, z, stats = self.simulate_environment(
                params, matrix, agents, airflow
            )
            solution.objectives[:] = [x, y, z]

            return stats
        else:
            solution.objectives[:] = self.simulate_environment(
                params, matrix, agents, airflow
            )

    # def lazy_evaluate(self, solution):
    #    """Lazy evaluation. Estimate objective values."""
    #    x = solution.variables[:]
    #    infection_count, inverse_tempurature, = (
    #        int(),
    #        int(),
    #        int(),
    #    )
    #    # temp score is inverse to windows opened.
    #    inverse_tempurature = Counter(solution[: self.windowMap.shape[0]])[1]
    #    # infection count is projected infection count
    #    """ Infection rate for single epoch, multiplied by total epochs.
    #    * Contact infection is proportional to available space in room, number of agents, number of infected agents, and chance of infection from contact.
    #    * Airborne infection is proportional to number of open windows, number of agents, number of infected agents, airborne infection release rate, airborne decay of infection, and chance of infection from airborne infection.
    #    """
    #    free_space = Counter(i for i in list(itertools.chain.from_iterable(self.environment)))[0]
    #    init_infected = int(self.agentPopSize * self.init_infected_ratio)
    #    init_healthy = self.agentPopSize - init_infected
    #    chance_of_contact_infection = self.pathogen_info.infection_from_contact
    #    chance_of_airborne_infection = self.pathogen_info.infection_from_max_pollutants
    #    airborne_infection_release_rate = self.pathogen_info.pathogen_release_payload
    #    max_pollutants = self.pathogen_info.max_pollutants
    #    airborne_decay = self.pathogen_info.airborne_decay
    #    chance_of_contact = (self.agentPopSize / free_space)
    #    contact_infections = ((free_space / self.agentPopSize) * chance_of_contact_infection * init_infected) * self.epochs
    #    airborne_infections = ((free_space / self.agentPopSize) * chance_of_airborne_infection * init_infected / inverse_tempurature) * self.epochs
    #    infection_count = contact_infections + airborne_infections + init_infected
    #    # update solution objectives
    #    solution.objectives[:] = infection_count, inverse_tempurature

    def get_environment(self, physical_vars):
        """Modify environment matrix by options.

        Inputs:
            solution: uses physical factors influencing environment.

        Outputs:
            matrix: map of environment, adjusted for physical factors.
            airflow: superimposed map of airflow in room.

        """

        matrix = self.environment[:]  # copy environment map

        # toggle for windows
        for idx, toggle in enumerate(physical_vars):  # indexes 0-9
            x, y = self.windowMap[idx]

            if toggle == 0:
                matrix[x][y] = 1  # set window to wall if not open

        # recalculate airflow map
        airflow = self.get_airflow(matrix)

        return matrix, airflow

    def get_airflow(self, matrix):
        """Calculate airflow map for environment by appling CA ruleset.

        Inputs:
            matrix: map of environment, with windows toggled open/closed.
            #toggle_ac: indicator of wether air conditioning is enabled in room or not.

        Outputs:
            airflow: map of airflow cell objects, indicating airflow velocity at given cell.

        """

        cellular_automaton = cpl.init_simple2d(*self.dimensions)

        # set positive and negative air sources
        for x, y in self.windowMap:
            if matrix[x][y] == 2:
                if y < int(
                    self.environment.shape[1] / 2
                ):  # positive incoming pressure for windows in top half
                    pressure = 1000000
                else:  # negative incoming pressure for windows in bottom half
                    pressure = -1000000

                cellular_automaton[:, x, y] = pressure

        # add random low pressure airflow for natural cirulation
        valid_space = np.empty((0, 2), dtype=int)
        for (x, y), value in np.ndenumerate(matrix):
            if value == 0:  # if air, then free for spawning
                valid_space = np.append(valid_space, [[x, y]], axis=0)
        
        np.random.shuffle(valid_space)

        for x, y in [coord for coord in valid_space[:10]]:  # select 10 random points
            cellular_automaton[:, x, y] = -500000 if random.random() < 0.5 else 500000  # set to random positive or negative pressure

        # hard set for testing
        # cellular_automaton[:, [0, 0, 0, 4, 5], [1, 5, 8, 9, 0]] = 10000
        # cellular_automaton[:, [8, 8, 9, 9, 9], [0, 9, 2, 5, 7]] = -10000

        airflow_ca = cpl.evolve2d(
            cellular_automaton,
            timesteps=self.until_fixed_point_or_max(),
            apply_rule=AirflowRule(matrix),
            neighbourhood="Moore",
        )

        if self.verbose > 2:
            print("Number of timesteps to reach fixed point: %s" % len(airflow_ca))
            airflow_ani = cpl.plot2d_animate(
                airflow_ca,
                show_grid=True,
                title="Airflow",
                interval=50,
                colormap="Blues",
            )
            #display(HTML(airflow_ani.to_jshtml()))

        airflow = airflow_ca[-1]  # final state == fixed state and correct airflow

        return airflow

    def until_fixed_point_or_max(self):
        """Run evolution till stable or reached max epochs.

        Outputs:
            _timesteps: callable function to determine if evolution has halted. Exits if previous and current generations are same state, or a maximum number of generations has been reached.

        """

        def _timesteps(ca, t):
            if len(ca) > 1:
                return (
                    False
                    if (ca[-2] == ca[-1]).all() or len(ca) >= self.max_airflow_epochs
                    else True
                )

            return True

        return _timesteps

    def get_agents(self, matrix):
        """Create population of agents.

        Inputs:
            solution: uses social factors influencing environment.

        Outputs:
            agents: population.

        """

        cellular_automaton = cpl.init_simple2d(*self.dimensions)

        # get list of valid non-wall cells to spawn agents in
        spawnpoints = np.empty((0, 2), dtype=int)
        for (x, y), value in np.ndenumerate(matrix):
            if value == 0:  # if air, then free for spawning
                spawnpoints = np.append(spawnpoints, [[x, y]], axis=0)

        # random shuffle coords and extract popSize list of spawnpoints
        np.random.shuffle(spawnpoints)

        init_infected = int(self.agentPopSize * self.init_infected_ratio)
        init_healthy = self.agentPopSize - init_infected

        for x, y in [coord for coord in spawnpoints[:init_infected]]:
            cellular_automaton[:, x, y] = 2

        for x, y in [
            coord for coord in spawnpoints[init_infected + 1 : init_healthy + 1]
        ]:
            cellular_automaton[:, x, y] = 1

        return cellular_automaton

    def simulate_environment(self, params, matrix, agents, airflow):
        """Simulate environment for N epochs.

        Outputs:
            infection count: final number of infected agents.
            inverse social distance: social distance parameter fed from solution.
            inverse tempurature: number of windows open.

        """

        infection_count, inverse_social_distance, inverse_tempurature, = (
            int(),
            int(),
            int(),
        )

        inverse_tempurature = Counter(params["physical_vars"])[
            1
        ]  # tempurature punishment == number of open windows
        inverse_social_distance = social_distance = params[
            "social_distance"
        ]  # social distance punishment == social distance policy, to discourage large distancing
        masking = params["masking"]

        infection = cpl.init_simple2d(
            *self.dimensions
        )  # initialise empty space for airborne infection layer

        agent_ca, infection_ca = evolveParallel(
            agents,
            infection,
            timesteps=self.epochs,
            apply_rule1=AgentRule(
                matrix,
                social_distance,
                self.pathogen_info,
                masking,
                self.mask_effectiveness,
            ),
            apply_rule2=InfectionRule(matrix, airflow, pathogen_info),
            neighbourhood="Moore",
        )

        if self.verbose > 2:
            agent_ani = cpl.plot2d_animate(
                agent_ca,
                show_grid=True,
                title="Agents",
                interval=100,
                colormap="Accent",
            )
            infection_ani = cpl.plot2d_animate(
                infection_ca,
                show_grid=True,
                title="Infection airflow",
                interval=100,
                colormap="YlGn",
            )

            # render as js animations
            #display(HTML(agent_ani.to_jshtml()))
            #display(HTML(infection_ani.to_jshtml()))

        final_state = agent_ca[-1]
        
        infection_count = Counter(
            i for i in list(itertools.chain.from_iterable(final_state))
        )[2]

        stats = np.empty((0,2), dtype=int)
        for state in agent_ca:
            counts = Counter(i for i in list(itertools.chain.from_iterable(state)))
            stats = np.append(stats, [[counts[1], counts[2]]], axis=0)  # add healthy and infected

        if self.verbose > 1:
            # graph infection rate over time
            timestamps = list(range(agent_ca.shape[0]))
            plt.plot(timestamps, stats[:,0])
            plt.plot(timestamps, stats[:,1])
            plt.show()

        if self.verbose > 0:
            # return infection_count, inverse_tempurature
            return infection_count, inverse_social_distance, inverse_tempurature, stats
        
        else:
            return infection_count, inverse_social_distance, inverse_tempurature
