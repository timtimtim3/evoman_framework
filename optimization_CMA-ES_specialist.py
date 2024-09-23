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
from plotter import plot_evolution

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def main(args, enemy):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'cma_es_optimization_specialist'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    def train(args, enemy):
        # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=experiment_name,
                        enemies=[enemy],
                        playermode="ai",
                        player_controller=player_controller(args.n_hidden_neurons), # you  can insert your own controller here
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)

        # number of weights for multilayer with 10 hidden neurons
        n_vars = (env.get_num_sensors()+1) * args.n_hidden_neurons + (args.n_hidden_neurons+1)*5

        # start writing your own code from here

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
            es.tell(solutions, -fitness_values) # Use negative because cma minimizes passed values but we want to
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
            print(f'Population size in this generation: {es.popsize}')  # <-- Added this line to print population size

            # if generation % 10 == 0:
            #     np.save(f"{experiment_name}/best_solution_gen_{generation}.npy", es.result.xbest)

        print(f'Best final solution found in generation {best_generation} with fitness: {best_fitness}')
        # print(f'Final solution: {es.result.xbest}, best fitness: {-es.result.fbest}')
        # np.save(f"{experiment_name}/best_final_solution.npy", best_solution)  # Save the best solution
        return fit_tracker

    fit_trackers = []
    for _ in range(args.n_experiments):
        fit_tracker = train(args, enemy)
        print(fit_tracker)
        fit_trackers.append(fit_tracker)
    plot_evolution(fit_trackers)


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="CMA-ES Evolutionary Algorithm for Evoman")

    parser.add_argument('--n_hidden_neurons', type=int, default=10,
                        help="Number of hidden neurons in the neural network.")
    parser.add_argument('--n_experiments', type=int, default=2, help="Number of experiments to run.")
    parser.add_argument('--generations', type=int, default=10, help="Number of generations for the evolution process.")
    parser.add_argument('--sigma', type=float, default=0.5, help="Initial step size (sigma) for CMA-ES.")
    parser.add_argument('--enemies', type=int, nargs='+', default=[2],
                        help="List of enemies to train against (can be a single integer or multiple integers).")
    parser.add_argument('--pop_size', type=int, default=20, help="Population size for CMA-ES.")

    args = parser.parse_args()

    for enemy in args.enemies:
        print(f"Training enemy on {enemy}")
        main(args, enemy)
