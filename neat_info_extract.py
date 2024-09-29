import neat
import pickle
from neat.six_util import itervalues
from neat.math_util import mean
from plotter import plot_evolution, save_evolution_data


# Variables
runs = 10
generations = 50
enemies = [1, 2, 3]

# Loop through enemies
for enemy in enemies:
    fit_trackers = []

    # Loop through all 10 runs
    for run in range(1, runs+1):
        fit_tracker = {"mean": [], "max": []}

        # Loop through all generations
        for i in range(generations):
            # Restore checkpoint
            name = 'neatcheckpoints/enemy-' + str(enemy) + '/run-' + str(run) + '/' + str(i)
            population = neat.Checkpointer.restore_checkpoint(name)

            # Get fitnesses and calculate mean and max
            fitnesses = [c.fitness for c in itervalues(population.population)]
            fitnesses_clean = [c if c is not None else 0 for c in fitnesses]
            fit_tracker['mean'].append(mean(fitnesses_clean))
            fit_tracker['max'].append(max(fitnesses_clean))

        # Track for each run
        fit_trackers.append(fit_tracker)

    # Save for each enemy
    save_evolution_data(fit_trackers, "NEAT", enemy)
    
    


plot_evolution(fit_trackers)

