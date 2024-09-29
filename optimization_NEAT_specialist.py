###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import pickle

from evoman.environment import Environment
from evoman.controller import Controller
from optimization_utils import NEAT_Controller


# imports other libs
import numpy as np
import os
import neat

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness, p,e,t = env.play(pcont=net)


def main(config_file):

    
    # start writing your own code from here

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,  neat.DefaultStagnation,
                         filename=config_file)
    
    for i in range(1,runs+1):
        # initializes simulation in individual evolution mode, for single static enemy.
        
        population = neat.Population(config)

        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        name = 'neatcheckpoints/enemy-' + str(enemy) + '/run-' + str(i) + '/'
        population.add_reporter(neat.Checkpointer(1,filename_prefix=name))

        winner = population.run(eval_genomes, 50)
        name = 'neat-enemy' + str(enemy) + '-' + str(i) + '.pkl'

        with open(name, 'wb') as f:
            pickle.dump(winner, f)
            print("Finished dumping")
        print('')
        print('-------------------------------------')
        print('finished run' 
              , i)
        print('-------------------------------------')
        print('')



# choose this for not using visuals and thus making experiments faster
headless = True

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'neat_optimization_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
enemy = 3
runs = 10

env = Environment(experiment_name=experiment_name,
                        enemies=[enemy],
                        playermode="ai",
                        player_controller=NEAT_Controller(), # you  can insert your own controller here
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)


# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config')
    main(config_path)