###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################
import argparse
import json
import pickle

from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
import cma
from plotter import plot_evolution, plot_box, save_individual_gains, save_evolution_data


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

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
            print('kiki')
        if CMA_params.pop(f"enemy {i}"):
            enemy_group_CMA.append(i)
            print("kaka")
    return GA_params, CMA_params, enemy_groups

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
    return fit_tracker, best_solution

def main(args):
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(args.experiment_name):
        os.makedirs(args.experiment_name)

    individual_gains_by_enemy_group = {}

    # Load hypertune params json files, split into two enemy groups
    _, CMA_params, enemy_groups = load_hypertune_params()

    # Remove generations and sigma
    generations = CMA_params.pop("generations")
    sigma = CMA_params.pop("sigma")

    for group_algo_name, enemy_group in enemy_groups.items():
        print(enemy_group)

        fit_trackers, best_models = [], []
        for i in range(args.n_experiments):
            fit_tracker, best_model = train(args, enemy_group, generations, sigma, model_params=CMA_params)
            best_models.append(best_model)
            fit_trackers.append(fit_tracker)
            os.makedirs('best_models/tim', exist_ok = True)
            with open(f'best_models/tim/{i}.json', 'w') as json_file:
                json.dump(best_model.tolist(), json_file, indent=3)
        save_evolution_data(fit_trackers, args.algo_name, enemy_group, target_directory=args.save_dir_evolution)

        print(fit_trackers)
        plot_evolution(fit_trackers, save=bool(args.save), save_path=f"{args.algo_name}_enemy{enemy_group}_evolution.png",
                       save_dir=args.save_dir_evolution)

        test_env = Environment(experiment_name=args.experiment_name,
                               enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                               playermode="ai",
                               multiplemode="yes",
                               player_controller=player_controller(args.n_hidden_neurons),
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

        name = 'tournament_models_tim/'
        os.makedirs(name, exist_ok = True)

        path = name + str(enemy_group) + '.pkl'

        with open(path, 'wb') as f:
            pickle.dump(best_models, f)
            print("Finished dumping")

    print(individual_gains_by_enemy_group)
    save_individual_gains(individual_gains_by_enemy_group, algo_name=args.algo_name, save_dir=args.save_dir_gains)
    plot_box(individual_gains_by_enemy_group, save=bool(args.save), save_path=f"{args.algo_name}_boxplot.png",
             save_dir=args.save_dir_gains)


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

    args = parser.parse_args()

    main(args)
