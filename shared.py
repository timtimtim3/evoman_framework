import json
import os
from evoman.environment import Environment
from demo_controller import player_controller
import pandas as pd
from datetime import timedelta  # For formatting time durations
import time  # Import time module to track durations

def load_json(file_path):
    """
    Loads a JSON file into a dictionary.

    :param file_path: Path to the JSON file.
    :return: A dictionary containing the data from the JSON file.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def load_hypertune_params(GA_path='hypertune_params/GA-params.json',
                          CMA_path='hypertune_params/CMA-ES-params.json'):
    GA_params = load_json(GA_path)
    CMA_params = load_json(CMA_path)

    enemy_group_GA = []
    enemy_group_CMA = []

    enemy_groups = {"CMA-ES": enemy_group_CMA, "GA": enemy_group_GA}

    for i in range(1, 9):
        # Remove enemies from params and collect them in a single list
        if GA_params.pop(f"enemy {i}"):
            enemy_group_GA.append(i)
        if CMA_params.pop(f"enemy {i}"):
            enemy_group_CMA.append(i)
    return GA_params, CMA_params, enemy_groups


def evaluate(model, test_env):
    f, p, e, t = test_env.play(pcont=model)
    gain = p - e
    return gain


def evaluate_and_save_best_model(args, best_model_overall, best_model_enemy_group, best_individual_gain_overall):
    """
    Evaluates the best model on each enemy individually, saves the results as a CSV file,
    and updates the 'best_models_gains.json' file with the best individual gain.

    Args:
        args: Argument parser containing necessary configurations.
        best_model_overall: The best model obtained across all runs and enemy groups.
        best_model_enemy_group: The enemy group on which the best model was trained.
        best_individual_gain_overall: The best individual gain achieved by the best model.
    """
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

    # Now, handle the 'best_models_gains.json' file
    # The file will be saved in 'args.save_dir_gains'
    best_models_gains_file = os.path.join(args.save_dir_gains, 'best_models_gains.json')

    # Initialize the dictionary
    if os.path.exists(best_models_gains_file):
        # Load the existing dictionary
        with open(best_models_gains_file, 'r') as f:
            best_models_gains = json.load(f)
    else:
        # Create a new dictionary
        best_models_gains = {}

    # Add the new entry
    best_models_gains[filename] = best_individual_gain_overall

    # Save the dictionary back to the JSON file
    with open(best_models_gains_file, 'w') as f:
        json.dump(best_models_gains, f, indent=4)

    print(f"Updated best_models_gains.json with {filename}: {best_individual_gain_overall}")


def calculate_time_metrics(experiment_start_time, experiment_end_time, start_time, total_runs_completed, total_runs):
    """
    Calculates time metrics for the experiments, including total time elapsed, ETA,
    experiment duration, and percentage completed. Formats the times into timedelta objects.

    Args:
        experiment_start_time (float): Start time of the current experiment.
        experiment_end_time (float): End time of the current experiment.
        start_time (float): Start time of the entire training process.
        total_runs_completed (int): Total number of runs completed so far.
        total_runs (int): Total number of runs to be executed.

    Returns:
        tuple: A tuple containing:
            - total_time_so_far_td (timedelta): Total time elapsed so far.
            - eta_td (timedelta): Estimated time remaining.
            - experiment_duration_td (timedelta): Duration of the current experiment.
            - percent_completed (float): Percentage of runs completed.
    """
    # Calculate total time so far
    total_time_so_far = experiment_end_time - start_time

    # Calculate average time per experiment
    average_time_per_experiment = total_time_so_far / total_runs_completed

    # Estimate total time and ETA
    estimated_total_time = average_time_per_experiment * total_runs
    eta = estimated_total_time - total_time_so_far

    # Calculate percent completed
    percent_completed = (total_runs_completed / total_runs) * 100

    # Format times into timedelta objects
    total_time_so_far_td = timedelta(seconds=int(total_time_so_far))
    eta_td = timedelta(seconds=int(eta))
    experiment_duration = experiment_end_time - experiment_start_time
    experiment_duration_td = timedelta(seconds=int(experiment_duration))

    return total_time_so_far_td, eta_td, experiment_duration_td, percent_completed