###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from evoman.controller import Controller

# imports other libs
import numpy as np
import os
import neat

class NEAT_Controller(Controller):
	# def __init__(self, _n_hidden):
	# 	self.n_hidden = [_n_hidden]

	# def set(self,controller, n_inputs):
	# 	# Number of hidden neurons

	# 	if self.n_hidden[0] > 0:
	# 		# Preparing the weights and biases from the controller of layer 1

	# 		# Biases for the n hidden neurons
	# 		self.bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
	# 		# Weights for the connections from the inputs to the hidden nodes
	# 		weights1_slice = n_inputs * self.n_hidden[0] + self.n_hidden[0]
	# 		self.weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((n_inputs, self.n_hidden[0]))

	# 		# Outputs activation first layer.


	# 		# Preparing the weights and biases from the controller of layer 2
	# 		self.bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
	# 		self.weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))

	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
		
		output = controller.activate(inputs)
	
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness, p,e,t = env.play(pcont=net)


def main(config_file):
    # start writing your own code from here

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))

    winner = population.run(eval_genomes, 300)

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                enemies=[2],
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