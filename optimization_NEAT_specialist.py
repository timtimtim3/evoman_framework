###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import pickle
import multiprocessing

from evoman.environment import Environment
from evoman.controller import Controller
from optimization_utils import NEAT_Controller


# imports other libs
import numpy as np
import os
import neat

def eval_genomes(genomes, config): # used to evaluate genome, creates neural network using genome and then uses it to play a game to find fitness
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness, p,e,t = env.play(pcont=net)

def eval_genome(genome, config): # used to evaluate genome, creates neural network using genome and then uses it to play a game to find fitness
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    genome.fitness, p,e,t = env.play(pcont=net)
    return genome.fitness


def main(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,  neat.DefaultStagnation,
                         filename=config_file)
    
    for i in range(1,runs+1):
        # Initializes population with correct settings like mutation rate and pop size etc.
        try:
            print('exists')
            population = neat.Checkpointer.restore_checkpoint('neatcheckpoints/enemy-' + str(10) + '/run-' + str(2) + '/734')
            population.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            population.add_reporter(stats)
            name = 'neatcheckpoints/enemy-' + str(enemy) + '/run-' + str(2) + '/'
            population.add_reporter(neat.Checkpointer(1,filename_prefix=name))

        except:
            print("jipe")
            population = neat.Population(config)

            # Adding reporters to store data for later and to see progress
            population.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            population.add_reporter(stats)
            name = 'neatcheckpoints/enemy-' + str(enemy) + '/run-' + str(2) + '/'
            population.add_reporter(neat.Checkpointer(1,filename_prefix=name))


        # Run the algorithm for number of generations

        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        # pe = neat.ThreadedEvaluator(4, eval_genome)

        winner = population.run(pe.evaluate, num_generations)

        # winner = population.run(eval_genomes, num_generations)

        # Save best genome
        name = 'neat-enemy' + str(enemy) + '-' + str(i) + '.pkl'
        with open(name, 'wb') as f:
            pickle.dump(winner, f)
            print("Finished dumping")

        os.system('beep')



# choose this for not using visuals and thus making experiments faster
headless = True

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'neat_optimization_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Variables
n_hidden_neurons = 10
enemy = 10
runs = 1
num_generations = 100

# Setup environment for a single static enemy
env = Environment(experiment_name=experiment_name,
                        enemies=[1,2,3,4,5,6,7,8],
                        playermode="ai",
                        multiplemode="yes",
                        player_controller=NEAT_Controller(), # you  can insert your own controller here
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)

if __name__ == '__main__':
    # finds file for where neat-config is stored
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config')
    main(config_path)