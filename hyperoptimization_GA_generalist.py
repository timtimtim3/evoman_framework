###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import random
import optuna
import pymysql
from statistics import mean
pymysql.install_as_MySQLdb()

from optimization_utils import Individual, evolve, update_population
from plotter import *
from tqdm import tqdm
import argparse

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

import argparse

def objective(trial):
    fit_tracker = {"max": [], "mean": []}

    tempenemies = []
    tempenemies.append(trial.suggest_categorical("enemy 1", [True, False]))
    tempenemies.append(trial.suggest_categorical("enemy 2", [True, False]))
    tempenemies.append(trial.suggest_categorical("enemy 3", [True, False]))
    tempenemies.append(trial.suggest_categorical("enemy 4", [True, False]))
    tempenemies.append(trial.suggest_categorical("enemy 5", [True, False]))
    tempenemies.append(trial.suggest_categorical("enemy 6", [True, False]))
    tempenemies.append(trial.suggest_categorical("enemy 7", [True, False]))
    tempenemies.append(trial.suggest_categorical("enemy 8", [True, False]))
    
    enemies = []
    for i in range(8):
        if tempenemies[i]:
            enemies.append(i+1) 
    
    if len(enemies) < 2:
        print('otherwise crashes sometimes ')
        enemies = [1,2]
        trial.set_user_attr('failure', True)
    trial.set_user_attr('enemies', enemies)

    n_hidden=10
    npop = trial.suggest_int('Population size', 10, 100)
    n_generations = trial.suggest_int('Generations', 100, 500)
    p = trial.suggest_float('Proportion of individuals who get changed', 0, 1)
    mutation_std = trial.suggest_float('Standard deviation for mutation',0,5)
    mutation_std_end = trial.suggest_float('Ending standard deviation for mutation', 0, 1)
    mutation_std_decreasing = trial.suggest_categorical('Whether the mutation rate is decreasing',[True, False])
    mutation_rate = trial.suggest_float('Mutation rate', 0, 1)
    mutation_prop= trial.suggest_float('Proportion of genes to mutate', 0, 1)
    learnable_mutation = trial.suggest_categorical('Whether to learn the mutation rate and std', [True, False])
    n_parents = trial.suggest_int('Number of parents', 2, 10)
    n_children = trial.suggest_int('Number of children', 2, 10)
    elitism = trial.suggest_int('Number of elitism', 0, 5)
    crossover_function = trial.suggest_categorical('Crossover', ["crossover_mixed", "crossover_recombination", "crossover_avg"])
    
    stagnation_threshold = trial.suggest_int('Stagnation threshold', 10, 50)
    doomsday_survival_rate = trial.suggest_float('Doomsday survival rate', 0.05, 0.40)
    
    k_round_robin = trial.suggest_int('K for round robin', 2, 30)
    n_islands = trial.suggest_int('Number of islands', 1, 10)
    migration_interval = trial.suggest_int('Migration interval', 5, 30)
    migration_rate = trial.suggest_float('Migration rate', 0.05, 0.40)
    if n_islands <= 1 or migration_rate == 0:
        migration_interval == 0
    
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=enemies,
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=player_controller(n_hidden), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    
    env2 = Environment(experiment_name=experiment_name,
                      enemies=[1,2,3,4,5,6,7,8],
                      playermode="ai",
                      multiplemode="yes",
                      player_controller=player_controller(n_hidden),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)
    
    # Initialize Islands
    pop_per_island = npop // n_islands
    remainder = npop % n_islands

    # Initialize population
    print(npop)
    n_inputs = env.get_num_sensors()
    network_size = (n_inputs+1) * n_hidden + (n_hidden+1)*5
    if learnable_mutation:
        list_learnable_params = []
        for i in range(npop):
            list_learnable_params.append({"mutation_std": max(0.01, mutation_std+np.random.normal(0, mutation_std/10)), 
                                          "mutation_rate": min(0.99, max(0.01, mutation_rate+np.random.normal(0, mutation_std/10)))})
                                        
    else:
        list_learnable_params = [False] * npop

    islands = []
    for i in range(n_islands):
        # Distribute extra individuals (remainder) to the first islands
        current_island_size = pop_per_island + (1 if i < remainder else 0)
        island_pop = [Individual(player_controller(n_hidden), list_learnable_params[i], n_inputs=n_inputs) for i in range(current_island_size)]
        pop_init = np.random.uniform(-1, 1, (current_island_size, network_size))
        n_inputs = env.get_num_sensors()
        for i, individual in enumerate(island_pop):
            individual.controller.set(pop_init[i], n_inputs)
            individual.evaluate(env)
        islands.append(island_pop)
    best_individual = islands[0][0]

    # Initialise std scheme
    if mutation_std_decreasing:
        stds = np.linspace(mutation_std, mutation_std_end, n_generations)
    else:
        stds = [mutation_std]*n_generations

    #Evolve
    generations_since_improvement = 0
    for i in range(n_generations):
        fitness = [individual.fitness for island in islands for individual in island]
        fit_tracker["max"].append(max(fitness))
        fit_tracker["mean"].append(np.mean(fitness))
        for island_pop in islands:
            island_pop = update_population(island_pop, 
                                p = p, 
                                mutation_std = stds[i], 
                                mutation_rate = mutation_rate, 
                                mutation_prop = mutation_prop, 
                                n_parents = n_parents, 
                                n_children = n_children, 
                                elitism = elitism, 
                                crossover_function=crossover_function,
                                k_round_robin = k_round_robin)
            for individual in island_pop:
                individual.evaluate(env)
            # print('one island evaluated')

        # Migration
        if migration_interval and i % migration_interval == 0:
            for i in range(n_islands):
                # Select a portion of individuals to migrate from island j to the next island
                n_migrants = int(len(islands[i]) * migration_rate)
                migrants = np.random.choice(islands[i], n_migrants, replace=False).tolist()
                target_island = islands[(i + 1) % n_islands]  # Wrap around to the first island

                # Add migrants to the next island and remove them from the current island
                target_island.extend(migrants)
                for migrant in migrants:
                    islands[i].remove(migrant)

        all_individuals = [individual for island in islands for individual in island] 
        best_of_generation = max(all_individuals, key=lambda x: x.fitness)
        if best_of_generation.fitness > best_individual.fitness:
            best_individual = best_of_generation
            generations_since_improvement = 0
        else:
            generations_since_improvement += 1

        # Doomsday
        if stagnation_threshold and generations_since_improvement >= stagnation_threshold: 
            all_individuals.sort(key=lambda x: x.fitness, reverse=True)  # Sort population by fitness
            survivors = int(npop * doomsday_survival_rate)
            new_population = all_individuals[:survivors]  # Keep only the top-performing individuals

            # Fill the rest with new random individuals
            new_random_individuals = [Individual(player_controller(n_hidden), list_learnable_params[ind]) for ind in range(npop - survivors)]
            pop_init = np.random.uniform(-1, 1, (npop - survivors, network_size))
            n_inputs = env.get_num_sensors()
            for j, individual in enumerate(new_random_individuals):
                individual.controller.set(pop_init[j], n_inputs)
                individual.evaluate(env)
            new_population.extend(new_random_individuals)

            # Redistribute the new population across islands
            np.random.shuffle(new_population)
            islands = []
            for j in range(n_islands):
                current_island_size = pop_per_island + (1 if j < remainder else 0)
                island_pop = new_population[:current_island_size]
                new_population = new_population[current_island_size:]  # Remove assigned individuals from list
                islands.append(island_pop)

            generations_since_improvement = 0  # Reset stagnation counter after Doomsday

        f, player, e, t = env2.play(pcont=best_of_generation.controller_to_genotype())
        print(f'fitness against all enemies: {f} in generation {i} trained fitness {best_of_generation.fitness}')
        trial.report(f, i)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    print(f'enemies {enemies}')
    f, player, e, t = env2.play(pcont=best_individual.controller_to_genotype())
    print(f'fitness against all enemies: {f}')
    trial.set_user_attr('best_model_solution', best_individual.controller_to_genotype().tolist())
    trial.set_user_attr('fit_tracker', fit_tracker)
    return f

    


    
if __name__ == '__main__':
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    study = optuna.create_study(study_name='run1', direction="maximize", storage='mysql+pymysql://root:UbuntuMau@localhost/GA', load_if_exists=True)

    study.optimize(objective, n_trials=10000, timeout=8*60*60)
    experiment_name = 'test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    
    # args = parser.parse_args()

    # main(args)