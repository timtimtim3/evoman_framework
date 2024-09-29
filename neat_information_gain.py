#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os
import matplotlib.pyplot as plt
import pickle
import neat
import neat.statistics
from neat.math_util import mean
from evoman.environment import Environment
from optimization_utils import NEAT_Controller
from plotter import *

# imports other libs
import numpy as np

# tests saved demo solutions for each enemy
os.environ["SDL_VIDEODRIVER"] = "dummy"

#Update the enemy
enemy = 8
runnumber = 1

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat-config')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
env = Environment(experiment_name='test',
				  playermode="ai",
				  player_controller=NEAT_Controller(),
			  	  speed="fastest",
				  enemymode="static",
				  level=2,
				  visuals=False)


# Variables
enemies = [1,2,3]
runs = 10

individual_gains_by_enemy = {}

# Loop through all enemies
for enemy in enemies:
    # Update enemy
    env.update_parameter('enemies',[enemy])

    # Loop through all runs
    individual_gain_means = []
    for i in range(1,runs+1):
        # Load best genome
        name = 'neatbest/neat-enemy' + str(enemy) + '-' + str(i) + '.pkl'
        with open(name, 'rb') as f:
            winner = pickle.load(f)

        # Create Neural network
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        individual_gain = []

        # Make it play 5 times and save individual gain 
        for i in range(5):
            f, p, e, t = env.play(pcont=net)
            individual_gain.append(p - e)

        # Calculate mean of 5 runs and then save the result
        individual_gain_mean = mean(individual_gain)
        individual_gain_means.append(individual_gain_mean)

    # Save for each enemy in dict
    individual_gains_by_enemy[f"NEAT {enemy}"] = individual_gain_means

print(individual_gains_by_enemy)
plot_box(individual_gains_by_enemy)

# Save the data for later
save_individual_gains(individual_gains_by_enemy, algo_name='NEAT')

