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
    num_enemies = trial.suggest_int('Number of enemies', 2, 8)
    n_parents = trial.suggest_int('Number of parents', 2, 10)
    n_children = trial.suggest_int('Number of children', 2, 10)
    print(f'pop_size {npop}, generations {n_generations}')
    
    value_range = list(range(1, 9))
    
    # Randomly sample unique integers without overlap
    enemies = random.sample(value_range, num_enemies)   
    
    trial.set_user_attr('enemies', enemies)

    
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=enemies,
                    multiplemode='yes',
                    playermode="ai",
                    player_controller=player_controller(n_hidden), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    
    # Initialize population
    network_size = (env.get_num_sensors()+1) * n_hidden + (n_hidden+1)*5
    if learnable_mutation:
        learnable_params = {"mutation_std": mutation_std, "mutation_rate": mutation_rate}
    else:
        learnable_params = False
    pop = [Individual(player_controller(n_hidden), learnable_params) for _ in range(npop)]
    best_individual = pop[0]
    pop_init = np.random.uniform(-1, 1, (npop, network_size))
    n_inputs = env.get_num_sensors()
    for i, individual in enumerate(pop):
        individual.controller.set(pop_init[i], n_inputs)

    # Initialise std scheme
    if mutation_std_decreasing:
        stds = np.linspace(mutation_std, mutation_std_end, n_generations)
    else:
        stds = [mutation_std]*n_generations
    #Evolve
    for individual in pop:
        # print('kiki')
        individual.evaluate(env)

    env2 = Environment(experiment_name=experiment_name,
                enemies=[1,2,3,4,5,6,7,8],
                multiplemode='yes',
                playermode="ai",
                player_controller=player_controller(n_hidden), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)
    print("Starting evolution")
    for i in range(n_generations):
        fitness = [individual.fitness for individual in pop]
        fit_tracker["max"].append(max(fitness))
        fit_tracker["mean"].append(np.mean(fitness))
        pop = update_population(pop, 
                                p = p, 
                                mutation_std = stds[i], 
                                mutation_rate = mutation_rate, 
                                mutation_prop = mutation_prop, 
                                n_parents = n_parents, 
                                n_children = n_children)
        for individual in pop:
                individual.evaluate(env)  
        print(len(pop)) 

        best_of_generation = max(pop, key=lambda x: x.fitness)
        if best_of_generation.fitness > best_individual.fitness:
            best_individual = best_of_generation

        f, player, e, t = env2.play(pcont=best_of_generation.controller_to_genotype())
        print(f'fitness against all enemies: {f} in generation {i} trained fitness {best_of_generation.fitness}')
        trial.report(f, i)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    print(f'enemies {enemies}')
    f, player, e, t = env2.play(pcont=best_individual)
    print(f'fitness against all enemies: {f}')
    trial.set_user_attr('best_model_solution', best_individual.controller)
    trial.set_user_attr('fit_tracker', fit_tracker)
    return f

    


    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Neuroevolution - Genetic Algorithm neural network.')
    # parser.add_argument('--npop', type=int, default=100, help='Population size')
    # parser.add_argument('--n_generations', type=int, default=50, help='Number of generations')
    # parser.add_argument('--p', type=float, default=0.5, help='Proportion of individuals who get changed')
    # parser.add_argument('--std', type=float, default=1, help='Standard deviation for mutation')
    # parser.add_argument('--std_end', type=float, default=0.1, help='Ending standard deviation for mutation')
    # parser.add_argument('--std_decreasing', type=bool, default=True, help='Whether the mutation rate is decreasing')
    # parser.add_argument('--mutation_rate', type=float, default=0.2, help='Mutation rate')
    # parser.add_argument('--mutation_prop', type=float, default=0.1, help='Proportion of genes to mutate')
    # parser.add_argument('--n_hidden', type=int, default=10, help='Number of hidden neurons')
    # parser.add_argument('--n_experiments', type=int, default=10, help='Number of experiments to run')
    # parser.add_argument('--enemies', type=int, nargs='+', default=[1,2,3], help='List of enemies to use')
    # parser.add_argument('--algo_name', type=str, default='GA', help='Name of the algorithm')   
    # parser.add_argument('--experiment_name', type=str, default='optimization_GA', help='Name of the experiment')
    # parser.add_argument('--save', type=int, default=1, help='Whether to save the plots')
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    study = optuna.create_study(study_name='test', direction="maximize", storage='mysql+pymysql://root:UbuntuMau@localhost/GA', load_if_exists=True)

    study.optimize(objective, n_trials=1, timeout=28800)
    experiment_name = 'test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    
    # args = parser.parse_args()

    # main(args)