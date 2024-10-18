###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################
import argparse
import json
import pickle
from datetime import timedelta  # For formatting time durations
import time  # Import time module to track durations

from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
import cma
from plotter import plot_evolution, plot_box, save_individual_gains, save_evolution_data
from shared import load_hypertune_params, evaluate_and_save_best_model, calculate_time_metrics
import pandas as pd

# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


def train(args, enemy_group, generations, sigma, model_params):
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=args.experiment_name,
                      enemies=enemy_group,
                      playermode="ai",
                      multiplemode="yes",
                      player_controller=player_controller(args.n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * args.n_hidden_neurons + (args.n_hidden_neurons + 1) * 5

    # params
    initial_guess = np.random.uniform(-1, 1, n_vars)
    best_fitness = float('-inf')
    best_solution = None
    best_generation = 0

    es = cma.CMAEvolutionStrategy(initial_guess, sigma, model_params)

    fit_tracker = {"mean": [], "max": []}

    for generation in range(generations):
        solutions = es.ask()
        fitness_values = evaluate(env, solutions)
        es.tell(solutions, -fitness_values)  # Use negative because cma minimizes passed values but we want to
        # maximize fitness

        current_best_fitness = max(fitness_values)
        current_best_solution = solutions[np.argmax(fitness_values)]
        fit_tracker["mean"].append(np.mean(fitness_values))
        fit_tracker["max"].append(current_best_fitness)

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            best_generation = generation + 1

        print(f'Generation {generation + 1}, best fitness: {current_best_fitness}')

    print(f'Best final solution found in generation {best_generation} with fitness: {best_fitness}')
    return fit_tracker, best_solution, best_fitness

def main(args):
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(args.experiment_name):
        os.makedirs(args.experiment_name)

    individual_gains_by_enemy_group = {}

    if bool(args.short_run):
        # Load hypertune params json files, split into two enemy groups
        _, CMA_params, enemy_groups = load_hypertune_params(
            GA_path='hypertune_params/GA-params-short.json',
            CMA_path='hypertune_params/CMA-ES-params-short.json'
        )
    else:
        _, CMA_params, enemy_groups = load_hypertune_params()

    # Remove generations and sigma
    generations = CMA_params.pop("generations")
    sigma = CMA_params.pop("sigma")

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

            fit_tracker, best_model, best_fitness = train(
                args,
                enemy_group,
                generations,
                sigma,
                model_params=CMA_params
            )
            best_models.append(best_model)
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
            player_controller=player_controller(args.n_hidden_neurons),
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

    # Save gains and plot boxplot
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

    # Evaluate the best model overall on each enemy individually
    player_energies = []
    enemy_energies = []
    for enemy in range(1, 9):
        test_env = Environment(
            experiment_name=args.experiment_name,
            enemies=[enemy],
            playermode="ai",
            multiplemode="no",
            player_controller=player_controller(args.n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False
        )
        f, p, e, t = test_env.play(pcont=best_model_overall)
        player_energies.append(p)
        enemy_energies.append(e)

    # Save the results as a CSV file
    results_df = pd.DataFrame({
        'Enemy': range(1, 9),
        'Player Energy': player_energies,
        'Enemy Energy': enemy_energies
    })

    # Construct the filename with the group and algo_name
    group_str = str(best_model_enemy_group)  # Enemy group as a list
    filename = f"{args.algo_name}_best_model_results_enemy{group_str}.csv"
    save_path = os.path.join(args.save_dir_gains, filename)

    results_df.to_csv(save_path, index=False)
    print(f"Saved per-enemy results to {save_path}")

    evaluate_and_save_best_model(
        args,
        best_model_overall,
        best_model_enemy_group,
        best_individual_gain_overall
    )


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="CMA-ES Evolutionary Algorithm for Evoman")

    parser.add_argument('--n_hidden_neurons', type=int, default=10,
                        help="Number of hidden neurons in the neural network.")
    parser.add_argument('--n_experiments', type=int, default=10, help="Number of experiments to run.")
    parser.add_argument('--experiment_name', type=str, default='cma_es_optimization_specialist')
    parser.add_argument('--algo_name', type=str, default='CMA-ES')
    parser.add_argument('--save', type=int, default=1, help="0 for not saving plots 1 for yes")
    parser.add_argument("--save_dir_gains", type=str, default="mean_gains_generalist",
                        help="save dir for mean gains")
    parser.add_argument("--save_dir_evolution", type=str, default="evolution_data_generalist",
                        help="save dir for evolution data")

    parser.add_argument('--short_run', type=int, default=0, help='Whether to do a short test run (less gens)')

    args = parser.parse_args()

    main(args)
