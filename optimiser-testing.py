from platypus import *
import matplotlib.pyplot as plt
import numpy as np

from pandemicpolicyca import PathogenInfo, AutomataOptimisation
from utility import parse_matrix_from_image

filepath = "testing/smb.png"
environment = parse_matrix_from_image(filepath, verbose=0)

pathogen_info = PathogenInfo(
    infection_from_contact=0.1,
    infection_from_max_pollutants=0.01,
    max_pollutants=50,
    airborne_decay=-4,
    pathogen_release_payload=10,
)

problem = AutomataOptimisation(
    environment, 
    pathogen_info,
    agentPopSize=80,
    init_infected_ratio=0.1,
    mask_effectiveness=0.5,
    epochs=50,
    n_objectives=3, 
    verbose=0
)
        
algorithms = [
    NSGAII,
    (NSGAIII, {"divisions_outer":12}),
    #GDE3,
    #IBEA,
    #SMPSO,
    SPEA2,
]

algorithm_names = ["NSGAII", "NSGAIII", "SPEA2"]    #"GDE3", "IBEA", "SMPSO", 

results = []
for unpack in algorithms:
    if type(unpack) == tuple:
        algorithm, options = unpack
    else:
        algorithm = unpack

    if algorithm == NSGAII or algorithm == GDE3 or algorithm == IBEA or algorithm == SMPSO or algorithm == SPEA2:
        model = algorithm(problem, population_size=100)
    
    elif algorithm == NSGAIII:
        model = NSGAIII(problem, divisions_outer=options["divisions_outer"], population_size=100)
    
    model.run(100)

    nondominated_solutions = nondominated(model.result)
    Y = np.array([s.objectives for s in nondominated_solutions])

    results.append(Y)

# display the results
fig = plt.figure()

for i, algorithm in enumerate(results):
    ax = fig.add_subplot(2, 5, i+1, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2])
    ax.set_title(algorithm_names[i])
    ax.set_xlabel("infection count")
    ax.set_ylabel("social distancing")
    ax.set_zlabel("temperature")

plt.savefig('test.png')