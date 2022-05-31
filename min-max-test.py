import matplotlib.pyplot as plt
import numpy as np

from pandemicpolicyca import PathogenInfo, AutomataOptimisation
from utility import parse_matrix_from_image

###############################################################################
# MAX TESTING
###############################################################################
filepath = "testing/smb.png"
environment = parse_matrix_from_image(filepath, verbose=0)

pathogen_info = PathogenInfo(
    infection_from_contact=0.1,
    infection_from_max_pollutants=0.01,
    max_pollutants=50,
    airborne_decay=-4,
    pathogen_release_payload=10,
)

class Dummy:
    def __init__(self, variables, n_objectives):
        self.variables = variables
        self.objectives = [0,]*n_objectives


# show solution simulation
epochs = 40
problem = AutomataOptimisation(
    environment, 
    pathogen_info,
    agentPopSize=80,
    init_infected_ratio=0.1,
    mask_effectiveness=0.5,
    epochs=epochs,
    n_objectives=3, 
    verbose=1,
    evaluating=1
)

print("Eval of English Gov Policy")

all_stats = []
avg_0, avg_1, avg_2 = 0,0,0
for i in range(10): # repeat 10 times for consistency
    dummy = Dummy(variables=[1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], n_objectives=3)
    stats = problem.evaluate(dummy)
    
    avg_0 += dummy.objectives[0]
    avg_1 += dummy.objectives[1]
    avg_2 += dummy.objectives[2]
    
    all_stats.append(stats)

avg_0 /= 10
avg_1 /= 10
avg_2 /= 10

print(f"Solution: [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]\nEvaluation: {[avg_0, avg_1, avg_2]}")

timestamps = list(range(epochs))

mean_gov = np.empty((0,2), dtype=float)

for i in timestamps:
    healthy = 0
    infected = 0
    for stats in all_stats:
        healthy += stats[:,0][i]
        infected += stats[:,1][i]
    
    healthy /= 10
    infected /= 10

    mean_gov = np.append(mean_gov, [[healthy, infected]], axis=0)

plt.title("Maximising Policy")
plt.plot(timestamps, mean_gov[:,0])
plt.plot(timestamps, mean_gov[:,1])
plt.show()



###############################################################################
# MIN TESTING
###############################################################################

filepath = "testing/smb.png"
environment = parse_matrix_from_image(filepath, verbose=0)

pathogen_info = PathogenInfo(
    infection_from_contact=0.1,
    infection_from_max_pollutants=0.01,
    max_pollutants=50,
    airborne_decay=-4,
    pathogen_release_payload=10,
)

class Dummy:
    def __init__(self, variables, n_objectives):
        self.variables = variables
        self.objectives = [0,]*n_objectives


# show solution simulation
epochs = 40
problem = AutomataOptimisation(
    environment, 
    pathogen_info,
    agentPopSize=80,
    init_infected_ratio=0.1,
    mask_effectiveness=0.5,
    epochs=epochs,
    n_objectives=3, 
    verbose=1,
    evaluating=1
)

print("Eval of Worst/No Policy")

all_stats = []
avg_0, avg_1, avg_2 = 0,0,0
for i in range(10): # repeat 10 times for consistency
    dummy = Dummy(variables=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_objectives=3)
    stats = problem.evaluate(dummy)
    
    avg_0 += dummy.objectives[0]
    avg_1 += dummy.objectives[1]
    avg_2 += dummy.objectives[2]
    
    all_stats.append(stats)

avg_0 /= 10
avg_1 /= 10
avg_2 /= 10

print(f"Solution: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\nEvaluation: {[avg_0, avg_1, avg_2]}")

timestamps = list(range(epochs))

mean_worst = np.empty((0,2), dtype=float)

for i in timestamps:
    healthy = 0
    infected = 0
    for stats in all_stats:
        healthy += stats[:,0][i]
        infected += stats[:,1][i]
    
    healthy /= 10
    infected /= 10

    mean_worst = np.append(mean_worst, [[healthy, infected]], axis=0)

plt.title("Worst Policy")
plt.plot(timestamps, mean_worst[:,0])
plt.plot(timestamps, mean_worst[:,1])
plt.show()


###############################################################################
# COMPARISON
###############################################################################

plt.title("Policy comparisons")
#plt.suptitle("Decreasing Healthy Population, Increasing Infected Population")
plt.ylim(1, 80)
#plt.plot(timestamps, mean_worst[:,0], color='red', label='Control group')
plt.plot(timestamps, mean_worst[:,1], color='red', label='Control group')
#plt.plot(timestamps, mean_nsgaII[:,0], color='green', label='Optimised policy')
plt.plot(timestamps, mean_nsgaII[:,1], color='green', label='Optimised policy')
#plt.plot(timestamps, mean_gov[:,0], color='orange', label='Maximising policy')
plt.plot(timestamps, mean_gov[:,1], color='orange', label='Max. policy')
plt.legend(loc='upper right')
plt.xlabel('Time steps')
plt.ylabel('Infected Population')
plt.show()