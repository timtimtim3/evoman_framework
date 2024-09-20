from demo_controller import player_controller
from evoman.controller import Controller
import numpy
from optimization_dummy import simulation

class Individual:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = None

    def evaluate(self, env):
        self.fitness = simulation(env, self.genotype)
        return self.fitness

def clone_controller(controller):
    new_controller = player_controller(controller.n_hidden)
    new_controller.bias1    = controller.bias1.copy()
    new_controller.bias2    = controller.bias2.copy()
    new_controller.weights1 = controller.weights1.copy()
    new_controller.weights2 = controller.weights2.copy()
    return new_controller

def mutate(controller, std = 0.1, mutation_rate = 0.2):
    # TODO: gradient as mutation
    # Neural network crossover in genetic algorithms using genetic programming - page 7
    """
    Mutates the controller by adding a random value from a normal distribution with mean 0 and standard deviation sigma
    to each element of the controller.
    :param controller: numpy array, the controller to be mutated
    :param sigma: float, the standard deviation of the normal distribution
    :return: numpy array, the mutated controller
    """
    p = [1-mutation_rate, mutation_rate]
    # Get the shape of the controller
    bias1_s = controller.bias1.shape
    bias2_s = controller.bias2.shape
    weights1_s = controller.weights1.shape
    weights2_s = controller.weights2.shape
    # Generate the random values
    r_bias1 = numpy.random.choice([0, 1], size=bias1_s, p=p) * numpy.random.normal(0, std, bias1_s)
    r_bias2 = numpy.random.choice([0, 1], size=bias2_s, p=p) * numpy.random.normal(0, std, bias2_s)
    r_weights1 = numpy.random.choice([0, 1], size=weights1_s, p=p) * numpy.random.normal(0, std, weights1_s)
    r_weights2 = numpy.random.choice([0, 1], size=weights2_s, p=p) * numpy.random.normal(0, std, weights2_s)
    # Change the weights of the controller
    new_controller = clone_controller(controller)
    new_controller.bias1    += r_bias1
    new_controller.bias2    += r_bias2
    new_controller.weights1 += r_weights1
    new_controller.weights2 += r_weights2
    return new_controller

def crossover_avg(controllers, equal = True):
    # Neural network crossover in genetic algorithms using genetic programming - page 7
    """
    Crosses over the controllers by averaging the weights and biases of the controllers.
    :param controllers: list of numpy arrays, the controllers to be crossed over
    :return: numpy array, the crossed over controller
    """
    # Get the number of controllers
    n = len(controllers)
    # Sum the weights and biases
    avg_bias1 = numpy.zeros(controllers[0].bias1.shape)
    avg_bias2 = numpy.zeros(controllers[0].bias2.shape)
    avg_weights1 = numpy.zeros(controllers[0].weights1.shape)
    avg_weights2 = numpy.zeros(controllers[0].weights2.shape)
    if equal:
        probs = [1/n]*n
    else:
        probs = np.random.uniform(0, 1, n)
        probs = probs/sum(probs)
    for controller, p in zip(controllers, probs):
        avg_bias1    += controller.bias1 *p
        avg_bias2    += controller.bias2 *p
        avg_weights1 += controller.weights1 *p
        avg_weights2 += controller.weights2 *p
    # Create the new controller
    new_controller = player_controller(controllers[0].n_hidden)
    new_controller.bias1    = avg_bias1
    new_controller.bias2    = avg_bias2
    new_controller.weights1 = avg_weights1
    new_controller.weights2 = avg_weights2
    return new_controller

def get_recombination(x):
    x = list(x)
    matrix_shape = x[0].shape
    recombined_matrix = numpy.zeros(matrix_shape)
    if len(matrix_shape)==2:
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                I = numpy.random.choice(range(len(x)))
                random_matrix = x[I]
                element = random_matrix[i, j]
                recombined_matrix[i, j] = element
    elif len(matrix_shape)==1:
        for i in range(matrix_shape[0]):
            I = numpy.random.choice(range(len(x)))
            random_matrix = x[I]
            element = random_matrix[i]
            recombined_matrix[i] = element
    return recombined_matrix


def crossover_recombination(controllers):
    """
    Crosses over the controllers using recombination.
    :param controllers: list of numpy arrays, the controllers to be crossed over
    :return: numpy array, the crossed over controller
    """
    # Get the shape of the controller
    biases1 = [controller.bias1 for controller in controllers]
    biases2 = [controller.bias2 for controller in controllers]
    weights1 = [controller.weights1 for controller in controllers]
    weights2 = [controller.weights2 for controller in controllers]
    # Create the new controller
    new_controller = player_controller(controllers[0].n_hidden)
    # Perform recombination for each parameter
    new_controller.bias1    = get_recombination(biases1)
    new_controller.bias2    = get_recombination(biases2)
    new_controller.weights1 = get_recombination(weights1)
    new_controller.weights2 = get_recombination(weights2)
    return new_controller



class NEAT_Controller(Controller):
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
def round_robin(pop, k = 10, choose_best = True):
    """
    Chooses k random controllers from the population and returns the best one.
    :param pop: list of individuals, the population of controllers
    :param k: int, the number of controllers to choose
    :param choose_best: bool, whether to choose the best controller from the k controllers
    :return: Individual, the chosen controller
    """
    # Choose k random controllers
    chosen = numpy.random.choice(pop, k)
    # Choose the best controller
    if choose_best:
        return max(chosen, key=lambda x: x.fitness)
    else:
        return min(chosen, key=lambda x: x.fitness)

def get_children(pop, p_children, crossover_function = crossover_avg):
    """
    Gets the children of the population using crossover.
    :param pop: list of individuals, the population of controllers
    :param p_children: float, the proportion of children to generate
    :param crossover_function: function, the function to use for crossover
    :return: list of individuals, the children of the population
    """
    N = len(pop)* p_children
    children = []
    for _ in range(int(N)):
        parent1 = round_robin(pop)
        parent2 = round_robin(pop)
        child = crossover_function([parent1.genotype, parent2.genotype])
        children.append(Individual(child))
    return children

def remove_worst(pop,  p = 0.25):
    """
    Removes the n worst controllers from the population using round robin.
    :param pop: list of individuals, the population of controllers
    :param p: float, the proportion of controllers to remove
    :return: list of individuals, the population of controllers with the n worst controllers removed
    """
    N = len(pop)* p
    for _ in range(int(N)):
        worst = round_robin(pop, choose_best = False)
        pop.remove(worst)
    return pop

def mutate_population(pop, std = 0.1, mutation_rate = 0.2, mutation_prop = 0.1):
    """
    Mutates the population of controllers.
    :param pop: list of individuals, the population of controllers
    :param sigma: float, the standard deviation of the normal distribution
    :return: list of individuals, the mutated population of controllers
    """
    for individual in pop:
        if numpy.random.uniform(0, 1) < mutation_prop:
            individual.genotype = mutate(individual.genotype, std, mutation_rate)
    return pop

def update_population(pop, p= 0.25, std = 0.1, mutation_rate = 0.2, mutation_prop = 0.1):
    """
    Updates the population by removing the worst controllers and generating new children.
    :param pop: list of individuals, the population of controllers
    :param p: float, the proportion of controllers to remove
    :return: list of individuals, the updated population of controllers
    """
    children = get_children(pop, p)
    pop = remove_worst(pop, p)
    pop = mutate_population(pop, std = std, mutation_rate=mutation_rate, mutation_prop=mutation_prop)
    pop += children
    return pop