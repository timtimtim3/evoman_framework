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


def main(args):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    
    if not os.path.exists(args.experiment_name):
        os.makedirs(args.experiment_name)

    n_hidden_neurons = args.n_hidden



    # # number of weights for multilayer with 10 hidden neurons
    fit_trackers = []
    best_models = []

    individual_gains_by_enemy = {}
    for enemy in args.enemies:
        for _ in tqdm(range(args.n_experiments)):
            pop, fit_tracker, best_individual = evolve(args, enemy, args.experiment_name)
            fit_trackers.append(fit_tracker)
            best_models.append(best_individual)
        save_evolution_data(fit_trackers, args.algo_name, enemy)

        test_env = Environment(experiment_name=args.experiment_name,
                               enemies=[enemy],
                               playermode="ai",
                               player_controller=player_controller(args.n_hidden),
                               # you  can insert your own controller here
                               enemymode="static",
                               level=2,
                               speed="fastest",
                               visuals=False)
        
        mean_gains = []
        for i in range(args.n_experiments):
            individual_gains = []
            for _ in range(5):
                f, p, e, t = test_env.play(pcont=best_models[i])
                individual_gain = p - e
                individual_gains.append(individual_gain)
            mean_gain = np.mean(individual_gains)
            mean_gains.append(mean_gain)

        individual_gains_by_enemy[f"{args.algo_name} {enemy}"] = mean_gains
    plot_box(individual_gains_by_enemy, save=bool(args.save), save_path=f"{args.algo_name}_enemy{enemy}_box.png")
    save_individual_gains(individual_gains_by_enemy, algo_name=args.algo_name)

    


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neuroevolution - Genetic Algorithm neural network.')
    parser.add_argument('--npop', type=int, default=100, help='Population size')
    parser.add_argument('--n_generations', type=int, default=50, help='Number of generations')
    parser.add_argument('--p', type=float, default=0.5, help='Proportion of individuals who get changed')
    parser.add_argument('--std', type=float, default=1, help='Standard deviation for mutation')
    parser.add_argument('--std_end', type=float, default=0.1, help='Ending standard deviation for mutation')
    parser.add_argument('--std_decreasing', type=bool, default=True, help='Whether the mutation rate is decreasing')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='Mutation rate')
    parser.add_argument('--mutation_prop', type=float, default=0.1, help='Proportion of genes to mutate')
    parser.add_argument('--n_hidden', type=int, default=10, help='Number of hidden neurons')
    parser.add_argument('--n_experiments', type=int, default=10, help='Number of experiments to run')
    parser.add_argument('--enemies', type=int, nargs='+', default=[1,2,3], help='List of enemies to use')
    parser.add_argument('--algo_name', type=str, default='GA', help='Name of the algorithm')   
    parser.add_argument('--experiment_name', type=str, default='optimization_GA', help='Name of the experiment')
    parser.add_argument('--save', type=int, default=1, help='Whether to save the plots')


    args = parser.parse_args()

    main(args)