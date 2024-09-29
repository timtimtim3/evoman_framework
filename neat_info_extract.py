import neat
import pickle
from neat.six_util import itervalues
from neat.math_util import mean
from plotter import plot_evolution



runs = [1,2,3]
# runs = [4,5,6]
generations = 50
enemies = [1]
fit_trackers = []
for run in runs:
    fit_tracker = {"mean": [], "max": []}
    for i in range(generations):
        name = 'neatcheckpoints/enemy-' + str(1) + '/run-' + str(run) + '/' + str(i)
        population = neat.Checkpointer.restore_checkpoint(name)
        fitnesses = [c.fitness for c in itervalues(population.population)]

        fitnesses_clean = [c if c is not None else 0 for c in fitnesses]
        fit_tracker['mean'].append(mean(fitnesses_clean))
        fit_tracker['max'].append(max(fitnesses_clean))
    fit_trackers.append(fit_tracker)
    

# print(fit_tracker['max'])
# print(fit_tracker['mean'])

plot_evolution(fit_trackers)
# with open('testsssssssssssssssssssssssssssssssssssssssss', 'wb') as f:
#         pickle.dump(fit_tracker, f)
#         print("Finished dumping")
