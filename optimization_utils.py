from demo_controller import player_controller
from evoman.controller import Controller
import numpy as np
# from optimization_dummy import simulation
from tqdm import tqdm 
from evoman.environment import Environment
from demo_controller import player_controller

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



# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

class Individual:
    def __init__(self, controller, learnable_mutation= False):
        self.controller = controller
        self.fitness = None
        self.network_size = controller.n_vars()
        self.learnable_mutation = bool(learnable_mutation)
        if self.learnable_mutation:
            self.mutation_std = learnable_mutation["mutation_std"]
            self.mutation_rate = learnable_mutation["mutation_rate"]

    def controller_to_genotype(self):
        genotype = []
        for params in [self.controller.bias1, self.controller.bias2, self.controller.weights1, self.controller.weights2]:
            genotype += list(params.flatten())
        genotype = np.array(genotype)
        return genotype

    def evaluate(self, env):
        genotype = self.controller_to_genotype()
        self.fitness = simulation(env, genotype)
        return self.fitness

def clone_controller(controller):
    new_controller = player_controller(controller.n_hidden)
    new_controller.bias1    = controller.bias1.copy()
    new_controller.bias2    = controller.bias2.copy()
    new_controller.weights1 = controller.weights1.copy()
    new_controller.weights2 = controller.weights2.copy()
    return new_controller

def mutate(individual, mutation_std = 0.1, mutation_rate = 0.2):
    # Neural network crossover in genetic algorithms using genetic programming - page 7
    """
    Mutates the controller by adding a random value from a normal distribution with mean 0 and standard deviation sigma
    to each element of the controller.
    :param controller: numpy array, the controller to be mutated
    :param sigma: float, the standard deviation of the normal distribution
    :return: numpy array, the mutated controller
    """
    controller = individual.controller
    if individual.learnable_mutation:
        mutation_std = individual.mutation_std
        mutation_rate = individual.mutation_rate
    p = [1-mutation_rate, mutation_rate]
    # Get the shape of the controller
    bias1_s = controller.bias1.shape
    bias2_s = controller.bias2.shape
    weights1_s = controller.weights1.shape
    weights2_s = controller.weights2.shape
    # Generate the random values
    r_bias1 = np.random.choice([0, 1], size=bias1_s, p=p) * np.random.normal(0, mutation_std, bias1_s)
    r_bias2 = np.random.choice([0, 1], size=bias2_s, p=p) * np.random.normal(0, mutation_std, bias2_s)
    r_weights1 = np.random.choice([0, 1], size=weights1_s, p=p) * np.random.normal(0, mutation_std, weights1_s)
    r_weights2 = np.random.choice([0, 1], size=weights2_s, p=p) * np.random.normal(0, mutation_std, weights2_s)
    # Change the weights of the controller
    new_controller = clone_controller(controller)
    new_controller.bias1    += r_bias1
    new_controller.bias2    += r_bias2
    new_controller.weights1 += r_weights1
    new_controller.weights2 += r_weights2
    # Update controller
    individual.controller = new_controller
    # Mutate the mutation rate and std
    if individual.learnable_mutation and np.random.uniform(0, 1) < individual.mutation_rate:
        individual.mutation_std = max(0.01, mutation_std + np.random.normal(0, mutation_std/10))
        individual.mutation_rate = max(0.01, mutation_rate + np.random.normal(0, mutation_std/10))
    return individual

def crossover_avg(individuals, equal = True):
    # Neural network crossover in genetic algorithms using genetic programming - page 7
    """
    Crosses over the controllers by averaging the weights and biases of the controllers.
    :param controllers: list of numpy arrays, the controllers to be crossed over
    :return: numpy array, the crossed over controller
    """
    controllers = [individual.controller for individual in individuals]
    # Get the number of controllers
    n = len(individuals)
    # Sum the weights and biases
    c = individuals[0].controller
    avg_bias1 = np.zeros(c.bias1.shape)
    avg_bias2 = np.zeros(c.bias2.shape)
    avg_weights1 = np.zeros(c.weights1.shape)
    avg_weights2 = np.zeros(c.weights2.shape)
    avg_mutation_rate = 0
    avg_mutation_std = 0
    if equal:
        probs = [1/n]*n
    else:
        probs = np.random.uniform(0, 1, n)
        probs = probs/sum(probs)
    for individual, p in zip(individuals, probs):
        avg_bias1    += individual.controller.bias1 * p
        avg_bias2    += individual.controller.bias2 * p
        avg_weights1 += individual.controller.weights1 * p
        avg_weights2 += individual.controller.weights2 * p
        if individual.learnable_mutation:
            avg_mutation_rate += individual.mutation_rate * p
            avg_mutation_std += individual.mutation_std * p

    # Create the new controller
    new_controller = player_controller(controllers[0].n_hidden)
    new_controller.bias1    = avg_bias1
    new_controller.bias2    = avg_bias2
    new_controller.weights1 = avg_weights1
    new_controller.weights2 = avg_weights2
    new_individual = Individual(new_controller, learnable_mutation={"mutation_std": avg_mutation_std, "mutation_rate": avg_mutation_rate})
    return new_individual

def get_recombination(x):
    x = list(x)
    matrix_shape = x[0].shape
    recombined_matrix = np.zeros(matrix_shape)
    if len(matrix_shape)==2:
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                I = np.random.choice(range(len(x)))
                random_matrix = x[I]
                element = random_matrix[i, j]
                recombined_matrix[i, j] = element
    elif len(matrix_shape)==1:
        for i in range(matrix_shape[0]):
            I = np.random.choice(range(len(x)))
            random_matrix = x[I]
            element = random_matrix[i]
            recombined_matrix[i] = element
    return recombined_matrix


def crossover_recombination(individuals):
    """
    Crosses over the controllers using recombination.
    :param controllers: list of numpy arrays, the controllers to be crossed over
    :return: numpy array, the crossed over controller
    """
    controllers = [individual.controller for individual in individuals]
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
    
    std = np.random.choice([individual.mutation_std for individual in individuals])
    mutation_rate = np.random.choice([individual.mutation_rate for individual in individuals])
    new_individual = Individual(new_controller, learnable_mutation={"mutation_std": std, "mutation_rate": mutation_rate})
    return new_individual

     
def round_robin(pop, k = 10, choose_best = True):
    """
    Chooses k random controllers from the population and returns the best one.
    :param pop: list of individuals, the population of controllers
    :param k: int, the number of controllers to choose
    :param choose_best: bool, whether to choose the best controller from the k controllers
    :return: Individual, the chosen controller
    """
    # Choose k random controllers
    chosen = np.random.choice(pop, k)
    # Choose the best controller
    if choose_best:
        return max(chosen, key=lambda x: x.fitness)
    else:
        return min(chosen, key=lambda x: x.fitness)

def get_best(pop, p=0.5):
    """
    Gets the best controller from the population.
    :param pop: list of individuals, the population of controllers
    :param p: float, the proportion of controllers to consider
    :return: Individual, the best controller
    """
    N = len(pop)* p
    best = sorted(pop, key=lambda x: x.fitness, reverse=True)
    return best[:int(N)]

def get_children(pop, p, n_parents, n_children, crossover_function):
    """
    Gets the children of the population using crossover.
    :param pop: list of individuals, the population of controllers
    :param p_children: float, the proportion of children to generate
    :param crossover_function: function, the function to use for crossover
    :return: list of individuals, the children of the population
    """
    candidates = get_best(pop, p)
    children = []
    while len(children) < len(pop)*p:
        parents = []
        for _ in range(n_parents):
            parent = round_robin(candidates)
            parents.append(parent)
        for _ in range(n_children):
            child = crossover_function(parents)
            children.append(child)
            if len(children) == len(pop)*p:
                break
    return children

def remove_worst(pop,  p):
    """
    Removes the n worst controllers from the population using round robin.
    :param pop: list of individuals, the population of controllers
    :param p: float, the proportion of controllers to remove
    :return: list of individuals, the population of controllers with the n worst controllers removed
    """
    if p==1:
        return []
    N = len(pop)* p
    for _ in range(int(N)):
        worst = round_robin(pop, choose_best = False)
        pop.remove(worst)
    return pop

def mutate_population(pop, mutation_std, mutation_rate, mutation_prop):
    """
    Mutates the population of controllers.
    :param pop: list of individuals, the population of controllers
    :param sigma: float, the standard deviation of the normal distribution
    :return: list of individuals, the mutated population of controllers
    """
    for individual in pop:
        if np.random.uniform(0, 1) < mutation_prop:
            individual = mutate(individual, mutation_std, mutation_rate)
    return pop

def update_population(pop, p, mutation_std, mutation_rate, mutation_prop, n_parents, n_children):
    """
    Updates the population by removing the worst controllers and generating new children.
    :param pop: list of individuals, the population of controllers
    :param p: float, the proportion of controllers to remove
    :return: list of individuals, the updated population of controllers
    """
    children = get_children(pop, p, n_parents, n_children, crossover_function = crossover_avg)
    pop = remove_worst(pop, p)
    pop += children
    pop = mutate_population(pop, mutation_std = mutation_std, mutation_rate=mutation_rate, mutation_prop=mutation_prop)
    return pop

def evolve(args, enemy, experiment_name):
    fit_tracker = {"max": [], "mean": []}

    npop=args.npop
    n_generations=args.n_generations
    p=args.p
    mutation_std=args.mutation_std
    mutation_std_end=args.mutation_std_end
    mutation_std_decreasing=args.mutation_std_decreasing
    mutation_rate=args.mutation_rate
    mutation_prop=args.mutation_prop
    n_hidden=args.n_hidden
    learnable_mutation=args.learnable_mutation
    n_parents = args.n_parents
    n_children = args.n_children

    
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=player_controller(n_hidden), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    
    # Initialize population
    network_size = (env.get_num_sensors()+1) * n_hidden + (n_hidden+1)*5
    if learnable_mutation:
        list_learnable_params = []
        for i in range(npop):
            list_learnable_params.append({"mutation_std": max(0.01, mutation_std+np.random.normal(0, mutation_std/10)), 
                                          "mutation_rate": max(0.01, mutation_rate+np.random.normal(0, mutation_std/10))})
                                        
    else:
        list_learnable_params = [False] * npop
    pop = [Individual(player_controller(n_hidden), list_learnable_params[i]) for i in range(npop)]
    best_individual = pop[0]
    pop_init = np.random.uniform(-1, 1, (npop, network_size))
    n_inputs = env.get_num_sensors()
    for i, individual in enumerate(pop):
        individual.controller.set(pop_init[i], n_inputs)

    # Initialise std scheme
    if mutation_std_decreasing:
        stds = np.linspace(mutation_std, mutation_std_end, n_generations)
    else:
        stds = [mutation_std]*n_generations
    #Evolve
    for individual in pop:
            individual.evaluate(env)
    for i in range(n_generations):
        fitness = [individual.fitness for individual in pop]
        fit_tracker["max"].append(max(fitness))
        fit_tracker["mean"].append(np.mean(fitness))
        pop = update_population(pop, 
                                p = p, 
                                mutation_std = stds[i], 
                                mutation_rate = mutation_rate, 
                                mutation_prop = mutation_prop, 
                                n_parents = n_parents, 
                                n_children = n_children)
        for individual in pop:
                individual.evaluate(env)   
        best_of_generation = max(pop, key=lambda x: x.fitness)
        if best_of_generation.fitness > best_individual.fitness:
            best_individual = best_of_generation
    
    return pop, fit_tracker, best_individual.controller_to_genotype()