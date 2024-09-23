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

from optimization_utils import Individual, evolve
from plotter import plot_evolution
from tqdm import tqdm
import argparse

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def main(args):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = args.n_hidden

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=args.enemies,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


    # # number of weights for multilayer with 10 hidden neurons
    fit_trackers = []
    best_individuals = []
    for _ in tqdm(range(args.n_experiments)):
        pop, fit_tracker, best_individual = evolve(env, 
                                npop=args.npop,
                                n_generations=args.n_generations,
                                p=args.p,
                                std=args.std,
                                std_end=args.std_end,
                                std_decreasing=args.std_decreasing,
                                mutation_rate=args.mutation_rate,
                                mutation_prop=args.mutation_prop)
        fit_trackers.append(fit_tracker)
        best_individuals.append(best_individual)
    plot_evolution(fit_trackers)
    


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neuroevolution - Genetic Algorithm neural network.')
    parser.add_argument('--npop', type=int, default=100, help='Population size')
    parser.add_argument('--n_generations', type=int, default=2, help='Number of generations')
    parser.add_argument('--p', type=float, default=0.2, help='Proportion of individuals who get changed')
    parser.add_argument('--std', type=float, default=1, help='Standard deviation for mutation')
    parser.add_argument('--std_end', type=float, default=0.1, help='Ending standard deviation for mutation')
    parser.add_argument('--std_decreasing', type=bool, default=True, help='Whether the mutation rate is decreasing')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='Mutation rate')
    parser.add_argument('--mutation_prop', type=float, default=0.1, help='Proportion of genes to mutate')
    parser.add_argument('--n_hidden', type=int, default=10, help='Number of hidden neurons')
    parser.add_argument('--n_experiments', type=int, default=2, help='Number of experiments to run')
    parser.add_argument('--enemies', type=int, nargs='+', default=[2], help='List of enemies to use')

    args = parser.parse_args()

    main(args)