###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################
import argparse
# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import cma
from plotter import plot_evolution, plot_box, save_individual_gains, save_evolution_data
import pymysql
pymysql.install_as_MySQLdb()

import optuna
import random
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))

# def define_model(trial, env, n_hidden_neurons):
#     # number of weights for multilayer with 10 hidden neurons
#     n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
#
#     # params
#     initial_guess = np.random.uniform(-1, 1, n_vars)
#     sigma = trial.suggest_float("sigma", 0, 5) #idk what the size should be for sigma
#     pop_size = trial.suggest_int("Population size", 10,100)
#     es = cma.CMAEvolutionStrategy(initial_guess, sigma, {'popsize': pop_size})
#
#     return es


def define_model(trial, env, n_hidden_neurons):
    # Number of weights for multilayer with hidden neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # Define ranges for params
    sigma = trial.suggest_float("sigma", 0.1, 5.0)  # Suggested range for sigma
    pop_size = trial.suggest_int("Population size", 10, 100)  # Suggested range for population size
    cma_mu = trial.suggest_int("CMA_mu", 2, pop_size // 2)  # Selection pressure
    cma_rankmu = trial.suggest_float("CMA_rankmu", 0.5, 2.0)  # Multiplier for rank-mu update
    cma_rankone = trial.suggest_float("CMA_rankone", 0.5, 2.0)  # Multiplier for rank-one update
    cma_dampsvec_fac = trial.suggest_float("CMA_dampsvec_fac", 0.1, 1.0)  # Damping factor

    # No tolerance needed because we train for fixed num iterations, CMA_stds only needed if we want
    # different sigma for each input, but just tuning sigma is simpler
    # tolstagnation = trial.suggest_int("tolstagnation", 50, 500)  # Stagnation tolerance
    # cma_stds = trial.suggest_float("CMA_stds", 0.1, 1.0)  # Range for standard deviations
    # tolx = trial.suggest_float("tolx", 1e-12, 1e-9)  # Tolerance in x-changes
    # tolfun = trial.suggest_float("tolfun", 1e-12, 1e-9)  # Tolerance in function value

    # Additional params
    cma_active = trial.suggest_categorical("CMA_active", [True, False])  # Active covariance update
    cma_mirrors = trial.suggest_categorical("CMA_mirrors", [True, False])  # Mirror sampling
    cma_mirrormethod = trial.suggest_categorical("CMA_mirrormethod", [0, 1, 2]) if cma_mirrors else None  # Only used if mirrors are on
    cma_dampsvec_fade = trial.suggest_float("CMA_dampsvec_fade", 0.01, 1.0)  # Fading damping factor
    minstd = trial.suggest_float("minstd", 1e-12, 1e-6)  # Minimal std in any direction
    maxstd = trial.suggest_float("maxstd", 0.1, 1.0)  # Maximal std

    params = {
        'popsize': pop_size,
        'CMA_mu': cma_mu,
        'CMA_rankmu': cma_rankmu,
        'CMA_rankone': cma_rankone,
        # 'CMA_stds': cma_stds,
        'CMA_dampsvec_fac': cma_dampsvec_fac,
        # 'tolx': tolx,
        # 'tolfun': tolfun,
        # 'tolstagnation': tolstagnation,
        'CMA_active': cma_active,
        'CMA_mirrors': cma_mirrors,
        'CMA_mirrormethod': cma_mirrormethod,
        'CMA_dampsvec_fade': cma_dampsvec_fade,
        'minstd': minstd,
        'maxstd': maxstd
    }

    initial_guess = np.random.uniform(-1, 1, n_vars)
    es = cma.CMAEvolutionStrategy(initial_guess, sigma, params)

    return es

# Global

def objective(trial):
    num_enemies = trial.suggest_int('Number of enemies', 2, 8)
    
    value_range = list(range(1, 9))
    
    # Randomly sample unique integers without overlap
    enemies = random.sample(value_range, num_enemies)   
    
    trial.set_user_attr('enemies', enemies)

    n_hidden_neurons = 10
    env = Environment(experiment_name=experiment_name,
                      enemies=enemies,
                      playermode="ai",
                      multiplemode="yes",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)
    env2 = Environment(experiment_name=experiment_name,
                      enemies=[1,2,3,4,5,6,7,8],
                      playermode="ai",
                      multiplemode="yes",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    model = define_model(trial, env, n_hidden_neurons)

    fit_tracker = {"mean": [], "max": []}
    # generations = trial.suggest_int("generations", 10, 50)
    generations = 1000
    best_fitness = float('-inf')
    best_solution = None
    best_generation = 0


    for generation in range(generations):
        solutions = model.ask()
        fitness_values = evaluate(env, solutions)
        model.tell(solutions, -fitness_values)  # Use negative because cma minimizes passed values but we want to
        # maximize fitness

        current_best_fitness = max(fitness_values)
        current_best_solution = solutions[np.argmax(fitness_values)]
        mean_fitness = np.mean(fitness_values)
        fit_tracker["mean"].append(mean_fitness)
        fit_tracker["max"].append(current_best_fitness)

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            best_generation = generation + 1

        print(f'Generation {generation + 1}, best fitness: {current_best_fitness}')
        f, p, e, t = env2.play(pcont=current_best_solution)
        print(f'fitness against all enemies: {f}')
        trial.report(f, generation)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    print(f'enemies {enemies}')
    f, p, e, t = env2.play(pcont=best_solution)
    print(f'fitness against all enemies: {f}')
    trial.set_user_attr('best_model_solution', best_solution)
    trial.set_user_attr('fit_tracker', fit_tracker)
    return f
    
if __name__ == '__main__':    
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    study = optuna.create_study(study_name='test', direction="maximize", storage='mysql+pymysql://root:UbuntuMau@localhost/example', load_if_exists=True)

    study.optimize(objective, n_trials=100, timeout=28800)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # print(optuna.importance.get_param_importances(study))
