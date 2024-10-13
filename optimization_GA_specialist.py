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
    

    individual_gains_by_enemy = {}
    for enemy in args.enemies:
        best_models = []
        print(f"Running training for enemy {enemy}")
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
        assert len(best_models) == args.n_experiments
        for i in range(args.n_experiments):
            individual_gains = []
            for _ in range(5):
                f, p, e, t = test_env.play(pcont=best_models[i])
                individual_gain = p - e
                individual_gains.append(individual_gain)
            mean_gain = np.mean(individual_gains)
            mean_gains.append(mean_gain)

        individual_gains_by_enemy[f"{args.algo_name} {enemy}"] = mean_gains
    plot_evolution(fit_trackers, save=bool(args.save), save_path=f"{args.algo_name}_evolution.png", show = True)
    plot_box(individual_gains_by_enemy, save=bool(args.save), save_path=f"{args.algo_name}_box.png")
    save_individual_gains(individual_gains_by_enemy, algo_name=args.algo_name)

import argparse
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Neuroevolution - Genetic Algorithm neural network.')
    #Current assignment is light version
    parser.add_argument('--npop', type=int, default=500, help='Population size')
    parser.add_argument('--n_generations', type=int, default=40, help='Number of generations')
    parser.add_argument('--p', type=float, default=0.5, help='Proportion of individuals who get changed')
    parser.add_argument('--mutation_std', type=float, default=1, help='Standard deviation for mutation')
    parser.add_argument('--mutation_std_end', type=float, default=0.1, help='Ending standard deviation for mutation')
    parser.add_argument('--mutation_std_decreasing', type=bool, default=True, help='Whether the mutation rate is decreasing')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='Mutation rate')
    parser.add_argument('--mutation_prop', type=float, default=0.1, help='Proportion of genes to mutate')
    parser.add_argument('--n_parents', type=int, default=4, help='Number of parents to select')
    parser.add_argument('--n_children', type=int, default=3, help='Number of children to generate')
    parser.add_argument('--learnable_mutation', type=bool, default=True, help='Whether to learn the mutation rate and std')
    parser.add_argument('--n_hidden', type=int, default=10, help='Number of hidden neurons')
    parser.add_argument('--n_experiments', type=int, default=6, help='Number of experiments to run')
    parser.add_argument('--enemies', type=int, nargs='+', default=[1], help='List of enemies to use')
    parser.add_argument('--algo_name', type=str, default='GA', help='Name of the algorithm')   
    parser.add_argument('--experiment_name', type=str, default='optimization_GA', help='Name of the experiment')
    parser.add_argument('--save', type=int, default=1, help='Whether to save the plots')
    parser.add_argument('--elitism', type=int, default=1, help='If 0, no elitism is used, otherwise the amount given')
    parser.add_argument('--crossover_function', type=str, default='crossover_avg', help='Type of crossover to use')
    parser.add_argument('--k_round_robin', type=int, default=10, help='Number of individuals to play against each other')
    parser.add_argument('--n_islands', type=int, default=4, help='Number of islands for migration')
    parser.add_argument('--migration_interval', type=int, default=5, help='Number of generations between migrations')
    parser.add_argument('--migration_rate', type=float, default=0.2, help='Proportion of individuals to migrate during migration')

    parser.add_argument('--stagnation_threshold', type=int, default=15, help='Number of generations without improvement before doomsday')
    parser.add_argument('--doomsday_survival_rate', type=float, default=0.2, help='Proportion of individuals to survive doomsday events')
    args = parser.parse_args()

    main(args)