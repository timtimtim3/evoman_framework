###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import time  # Import time module to track durations

from evoman.environment import Environment
from demo_controller import player_controller
# imports other libs
import numpy as np
import os

from optimization_utils import Individual, evolve
from plotter import *
from tqdm import tqdm
import argparse
from shared import load_hypertune_params, evaluate_and_save_best_model, calculate_time_metrics
import argparse



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
    # Choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(args.experiment_name):
        os.makedirs(args.experiment_name)

    n_hidden_neurons = args.n_hidden_neurons

    individual_gains_by_enemy_group = {}

    if bool(args.short_run):
        # Load hypertune params json files, split into two enemy groups
        GA_params, _, enemy_groups = load_hypertune_params(
            GA_path='hypertune_params/GA-params-short.json',
            CMA_path='hypertune_params/CMA-ES-params-short.json'
        )
    else:
        GA_params, _, enemy_groups = load_hypertune_params()

    GA_params = Params(GA_params)

    print(vars(GA_params))

    # Initialize variables to keep track of the best model overall
    best_model_overall = None
    best_individual_gain_overall = float('-inf')
    best_model_enemy_group = None  # To keep track of the group of the best model

    # Initialize total runs counters
    total_runs_completed = 0
    total_runs = len(enemy_groups) * args.n_experiments

    # Record the start time
    start_time = time.time()

    for idx, (group_algo_name, enemy_group) in enumerate(enemy_groups.items(), start=1):
        fit_trackers, best_models = [], []

        for i in range(args.n_experiments):
            percent_completed = (total_runs_completed / total_runs) * 100

            print()
            print("-----------------------------")
            print(f"Training on enemy group {idx} of {len(enemy_groups)}")
            print(f"Running experiment {i + 1} of {args.n_experiments}")

            # Record the experiment start time
            experiment_start_time = time.time()

            pop, fit_tracker, best_individual = evolve(
                GA_params,
                enemy_group,
                args.experiment_name,
                n_hidden=n_hidden_neurons
            )

            best_models.append(best_individual)
            fit_trackers.append(fit_tracker)

            # Record the experiment end time
            experiment_end_time = time.time()

            total_runs_completed += 1  # Increment the total runs completed

            # Call the function to calculate time metrics
            total_time_so_far_td, eta_td, experiment_duration_td, percent_completed = calculate_time_metrics(
                experiment_start_time,
                experiment_end_time,
                start_time,
                total_runs_completed,
                total_runs
            )

            # Print progress and timing information
            print(f"Completed {total_runs_completed} out of {total_runs} runs ({percent_completed:.2f}%)")
            print(f"Experiment duration: {experiment_duration_td}")
            print(f"Total time elapsed: {total_time_so_far_td}")
            print(f"Estimated time remaining (ETA): {eta_td}")
            print("-----------------------------")

        # Save evolution data and plot
        save_evolution_data(
            fit_trackers,
            args.algo_name,
            enemy_group,
            target_directory=args.save_dir_evolution
        )
        plot_evolution(
            fit_trackers,
            save=bool(args.save),
            save_path=f"{args.algo_name}_enemy{enemy_group}_evolution.png",
            save_dir=args.save_dir_evolution
        )

        # Create the test environment for multiple enemies
        test_env = Environment(
            experiment_name=args.experiment_name,
            enemies=[1, 2, 3, 4, 5, 6, 7, 8],
            playermode="ai",
            multiplemode="yes",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False
        )

        individual_gains = []
        for i in range(args.n_experiments):
            f, p, e, t = test_env.play(pcont=best_models[i])
            individual_gain = p - e
            individual_gains.append(individual_gain)

            # Update the best model overall if this individual gain is higher
            if individual_gain > best_individual_gain_overall:
                best_individual_gain_overall = individual_gain
                best_model_overall = best_models[i]
                best_model_enemy_group = enemy_group  # Keep track of which group this model came from

        # Store gains for the current enemy group
        individual_gains_by_enemy_group[f"{args.algo_name} {enemy_group}"] = individual_gains

    print("Individual gains by enemy group:", individual_gains_by_enemy_group)
    save_individual_gains(
        individual_gains_by_enemy_group,
        algo_name=args.algo_name,
        save_dir=args.save_dir_gains
    )
    plot_box(
        individual_gains_by_enemy_group,
        save=bool(args.save),
        save_path=f"{args.algo_name}_boxplot.png",
        save_dir=args.save_dir_gains
    )

    # Call the modular function to evaluate and save the best model results
    evaluate_and_save_best_model(
        args,
        best_model_overall,
        best_model_enemy_group,
        best_individual_gain_overall
    )


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser(description='Neuroevolution - Genetic Algorithm neural network.')
    # Current assignment is light version
    parser.add_argument('--n_hidden_neurons', type=int, default=10, help='Number of hidden neurons')
    parser.add_argument('--n_experiments', type=int, default=10, help='Number of experiments to run')
    parser.add_argument('--algo_name', type=str, default='GA', help='Name of the algorithm')
    parser.add_argument('--experiment_name', type=str, default='optimization_GA_generalist',
                        help='Name of the experiment')
    parser.add_argument('--save', type=int, default=1, help='Whether to save the plots')
    parser.add_argument("--save_dir_gains", type=str, default="mean_gains_generalist",
                        help="save dir for mean gains")
    parser.add_argument("--save_dir_evolution", type=str, default="evolution_data_generalist",
                        help="save dir for evolution data")

    parser.add_argument('--short_run', type=int, default=0, help='Whether to do a short test run (less gens)')

    args = parser.parse_args()

    main(args)
