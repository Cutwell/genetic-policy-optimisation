from platypus import *
import matplotlib.pyplot as plt
import numpy as np

from pandemicpolicyca import PathogenInfo, AutomataOptimisation
from utility import parse_matrix_from_image

from progress.bar import Bar

epochs = 50
population_size = 100
progressbar = Bar("Processing", max=(epochs * population_size))

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
    agentPopSize=100,
    init_infected_ratio=0.1,
    mask_effectiveness=0.5,
    epochs=20,
    n_objectives=3,
    verbose=0,
    progressbar=progressbar,
)

algorithm = NSGAII(problem, population_size=population_size)
algorithm.run(epochs)

# finish bar
progressbar.finish()

Y = np.array([s.objectives for s in algorithm.result])

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2])
ax.set_xlabel("infection count")
ax.set_ylabel("social distancing")
ax.set_zlabel("tempurature")
plt.show()

nondominated_solutions = nondominated(algorithm.result)
Y = np.array([s.objectives for s in nondominated_solutions])

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
ax.scatter3D(Y[:, 0], Y[:, 1], Y[:, 2])
ax.set_xlabel("infection count")
ax.set_ylabel("social distancing")
ax.set_zlabel("tempurature")
plt.show()
