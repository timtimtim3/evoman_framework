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
from shared import load_hypertune_params



# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


class Params:
    def __init__(self, params_dict):
        # Dynamically add attributes to the object
        for key, value in params_dict.items():
            setattr(self, key, value)

def main(args):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(args.experiment_name):
        os.makedirs(args.experiment_name)

    n_hidden_neurons = args.n_hidden

    individual_gains_by_enemy_group = {}

    # Load hypertune params json files, split into two enemy groups
    GA_params, _, enemy_groups = load_hypertune_params(GA_path='hypertune_params/GA-params-short.json')

    GA_params = Params(GA_params)

    print(vars(GA_params))

    for group_algo_name, enemy_group in enemy_groups.items():
        print(enemy_group)

        fit_trackers, best_models = [], []
        for i in range(args.n_experiments):
            pop, fit_tracker, best_individual = evolve(GA_params, enemy_group, args.experiment_name, n_hidden=n_hidden_neurons)

            best_models.append(best_individual)
            fit_trackers.append(fit_tracker)

        save_evolution_data(fit_trackers, args.algo_name, enemy_group, target_directory=args.save_dir_evolution)
        print(fit_trackers)

        plot_evolution(fit_trackers, save=bool(args.save), save_path=f"{args.algo_name}_enemy{enemy_group}_evolution.png",
                       save_dir=args.save_dir_evolution)

        test_env = Environment(experiment_name=args.experiment_name,
                               enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                               playermode="ai",
                               multiplemode="yes",
                               player_controller=player_controller(n_hidden_neurons),
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

        individual_gains_by_enemy_group[f"{args.algo_name} {enemy_group}"] = mean_gains

    print(individual_gains_by_enemy_group)
    save_individual_gains(individual_gains_by_enemy_group, algo_name=args.algo_name, save_dir=args.save_dir_gains)
    plot_box(individual_gains_by_enemy_group, save=bool(args.save), save_path=f"{args.algo_name}_boxplot.png",
             save_dir=args.save_dir_gains)


import argparse

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Neuroevolution - Genetic Algorithm neural network.')
    # Current assignment is light version
    parser.add_argument('--n_hidden', type=int, default=10, help='Number of hidden neurons')
    parser.add_argument('--n_experiments', type=int, default=10, help='Number of experiments to run')
    parser.add_argument('--algo_name', type=str, default='GA', help='Name of the algorithm')
    parser.add_argument('--experiment_name', type=str, default='optimization_GA_generalist',
                        help='Name of the experiment')
    parser.add_argument('--save', type=int, default=1, help='Whether to save the plots')
    parser.add_argument("--save_dir_gains", type=str, default="mean_gains_generalist",
                        help="save dir for mean gains")
    parser.add_argument("--save_dir_evolution", type=str, default="evolution_data_generalist",
                        help="save dir for evolution data")

    args = parser.parse_args()

    main(args)
