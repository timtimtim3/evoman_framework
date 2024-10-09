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
from neat.math_util import mean

headless = False

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Variables to get correct run
enemy = 10
runnumber = 1


# Get config file path
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat-config')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

# Load best genome from specific run
name = 'neat-enemy' + str(enemy) + '-' + str(runnumber) + '.pkl'
with open(name, 'rb') as f:
    winner = pickle.load(f)

# Create neural network with genome
net = neat.nn.FeedForwardNetwork.create(winner, config)

# Setup environment for a single static enemy
env = Environment(experiment_name='test',
				  playermode="ai",
                #   multiplemode='yes',
                #   enemies=[1,2,3,4,5,6,7,8],
				  player_controller=NEAT_Controller(),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)

# Update enemy to fight
# print(env.play(pcont=net))
# Show result
f = []
for i in range(1,9):
    env.update_parameter('enemies',[i])
    fitness, p, e, t = env.play(pcont=net)
    # print(fitness)
    if p > 0:
        print("win")
    f.append(fitness)
print(mean(f))
