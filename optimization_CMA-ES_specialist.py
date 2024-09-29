###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################
import argparse
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


# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


def train(args, enemy):
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=args.experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(args.n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * args.n_hidden_neurons + (args.n_hidden_neurons + 1) * 5

    # params
    initial_guess = np.random.uniform(-1, 1, n_vars)
    sigma = args.sigma
    generations = args.generations
    best_fitness = float('-inf')
    best_solution = None
    best_generation = 0

    es = cma.CMAEvolutionStrategy(initial_guess, sigma, {'popsize': args.pop_size})

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

    individual_gains_by_enemy = {}
    for enemy in args.enemies:
        print(f"Training on enemy {enemy}")

        fit_trackers, best_models = [], []
        for _ in range(args.n_experiments):
            fit_tracker, best_model = train(args, enemy)
            best_models.append(best_model)
            fit_trackers.append(fit_tracker)
        save_evolution_data(fit_trackers, args.algo_name, enemy)

        print(fit_trackers)
        plot_evolution(fit_trackers, save=bool(args.save), save_path=f"{args.algo_name}_enemy{enemy}_evolution.png")

        test_env = Environment(experiment_name=args.experiment_name,
                               enemies=[enemy],
                               playermode="ai",
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

        individual_gains_by_enemy[f"{args.algo_name} {enemy}"] = mean_gains

    print(individual_gains_by_enemy)
    save_individual_gains(individual_gains_by_enemy, algo_name=args.algo_name)
    plot_box(individual_gains_by_enemy, save=bool(args.save), save_path=f"{args.algo_name}_boxplot.png")


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="CMA-ES Evolutionary Algorithm for Evoman")

    parser.add_argument('--n_hidden_neurons', type=int, default=10,
                        help="Number of hidden neurons in the neural network.")
    parser.add_argument('--n_experiments', type=int, default=10, help="Number of experiments to run.")
    parser.add_argument('--generations', type=int, default=50, help="Number of generations for the evolution process.")
    parser.add_argument('--sigma', type=float, default=0.5, help="Initial step size (sigma) for CMA-ES.")
    parser.add_argument('--enemies', type=int, nargs='+', default=[1, 2, 3],
                        help="List of enemies to train against (can be a single integer or multiple integers).")
    parser.add_argument('--pop_size', type=int, default=100, help="Population size for CMA-ES.")
    parser.add_argument('--experiment_name', type=str, default='cma_es_optimization_specialist')
    parser.add_argument('--algo_name', type=str, default='CMA-ES')
    parser.add_argument('--save', type=int, default=1, help="0 for not saving plots 1 for yes")

    args = parser.parse_args()

    main(args)
