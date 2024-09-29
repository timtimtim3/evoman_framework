#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os
import pickle
import neat
import neat.statistics
from evoman.environment import Environment
from optimization_utils import NEAT_Controller

# imports other libs
import numpy as np

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness, p,e,t = env.play(pcont=net)

experiment_name = 'controller_specialist_NEAT_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
# tests saved demo solutions for each enemy

#Update the enemy
enemy = 1
runnumber = 6

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat-config')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
name = 'neat-enemy' + str(enemy) + '-' + str(runnumber) + '.pkl'

with open(name, 'rb') as f:
    winner = pickle.load(f)
# winner = population.run(eval_genomes, 1)


# neat.statistics.StatisticsReporter.post_evaluate(config, population )
# winner = population.best_genome
# print(population)
print(winner)
net = neat.nn.FeedForwardNetwork.create(winner, config)

env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=NEAT_Controller(),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)
env.update_parameter('enemies',[enemy])
for i in range(10):
    print(env.play(pcont=net))

