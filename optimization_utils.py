from demo_controller import player_controller
from evoman.controller import Controller
import numpy as np
# from optimization_dummy import simulation
from tqdm import tqdm 
from evoman.environment import Environment
from demo_controller import player_controller
from copy import deepcopy

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
    def __init__(self, controller, learnable_mutation= False, n_inputs = 20):
        self.controller = controller
        self.fitness = None
        self.network_size = controller.n_vars()
        self.n_inputs = n_inputs
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

def crossover_mixed(individuals):
    """
    Crosses over the controllers by mixing the genotypes.
    :param individuals: list of Individual, the individuals to be crossed over
    :return: Individual, the crossed over individual
    """
    # Get the genotypes
    genotypes = [individual.controller_to_genotype() for individual in individuals]
    # Choose a random individual as baseline
    baseline = genotypes[np.random.randint(len(genotypes))]
    new_genotype = deepcopy(baseline)
    counter = np.zeros(len(genotypes[0]))
    for genotype in genotypes:
        if np.array_equal(genotype, baseline):
            continue
        i = np.random.randint(len(genotype))
        counter[i:] += 1
        new_genotype[i:] = (new_genotype[i:] * counter[i:] + genotype[i:]) / (counter[i:] + 1)
    
    # Create a new controller with the new genotype
    new_controller = player_controller(individuals[0].controller.n_hidden)
    n_inputs = individuals[0].n_inputs
    new_controller.set(new_genotype, n_inputs)
    
    # Create a new individual with the new controller
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
    if isinstance(crossover_function, str):
        crossover_function = globals()[crossover_function]
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

def update_population(pop, p, mutation_std, mutation_rate, mutation_prop, n_parents, n_children, elitism, crossover_function = crossover_mixed):
    """
    Updates the population by removing the worst controllers and generating new children.
    :param pop: list of individuals, the population of controllers
    :param p: float, the proportion of controllers to remove
    :return: list of individuals, the updated population of controllers
    """
    if elitism>0:
        best = get_best(pop, elitism/len(pop))
    pop = remove_worst(pop, p)
    children = get_children(pop, p, n_parents, n_children, crossover_function = crossover_function)
    pop += children
    pop = mutate_population(pop, mutation_std = mutation_std, mutation_rate=mutation_rate, mutation_prop=mutation_prop)
    
    if elitism>0:
        #Allow for the best to be mutated, but if they are, we save a copy of the original
        best_possibly_mutated = mutate_population(best, mutation_std = mutation_std, mutation_rate=mutation_rate, mutation_prop=mutation_prop)
        possibly_mutated_geno = [individual.controller_to_genotype() for individual in best_possibly_mutated]
        best_geno = [individual.controller_to_genotype() for individual in best]
        counter =0
        for i in range(len(best)):
            if not np.array_equal(possibly_mutated_geno[i], best_geno[i]):
                pop.append(best[i])
                counter += 1
        if counter > 0:
            pop = remove_worst(pop, counter/len(pop))
             
        
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
    elitism = args.elitism #TODO: Implement elitisim

    n_islands = args.n_islands
    migration_interval = args.migration_interval
    migration_rate = args.migration_rate
    if n_islands <= 1 or migration_rate == 0:
        migration_interval == 0

    stagnation_threshold = args.stagnation_threshold
    doomsday_survival_rate = args.doomsday_survival_rate
    
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=player_controller(n_hidden), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    
    # Initialize Islands
    pop_per_island = npop // n_islands
    remainder = npop % n_islands

    # Initialize population
    n_inputs = env.get_num_sensors()
    network_size = (n_inputs+1) * n_hidden + (n_hidden+1)*5
    if learnable_mutation:
        list_learnable_params = []
        for i in range(npop):
            list_learnable_params.append({"mutation_std": max(0.01, mutation_std+np.random.normal(0, mutation_std/10)), 
                                          "mutation_rate": max(0.01, mutation_rate+np.random.normal(0, mutation_std/10))})
                                        
    else:
        list_learnable_params = [False] * npop

    islands = []
    for i in range(n_islands):
        # Distribute extra individuals (remainder) to the first islands
        current_island_size = pop_per_island + (1 if i < remainder else 0)
        island_pop = [Individual(player_controller(n_hidden), list_learnable_params[i], n_inputs=n_inputs) for i in range(current_island_size)]
        pop_init = np.random.uniform(-1, 1, (current_island_size, network_size))
        n_inputs = env.get_num_sensors()
        for i, individual in enumerate(island_pop):
            individual.controller.set(pop_init[i], n_inputs)
            individual.evaluate(env)
        islands.append(island_pop)
    best_individual = islands[0][0]

    # Initialise std scheme
    if mutation_std_decreasing:
        stds = np.linspace(mutation_std, mutation_std_end, n_generations)
    else:
        stds = [mutation_std]*n_generations
    #Evolve
    for i in range(n_generations):
        fitness = [individual.fitness for island in islands for individual in island]
        fit_tracker["max"].append(max(fitness))
        fit_tracker["mean"].append(np.mean(fitness))
        for island_pop in islands:
            island_pop = update_population(island_pop, 
                                p = p, 
                                mutation_std = stds[i], 
                                mutation_rate = mutation_rate, 
                                mutation_prop = mutation_prop, 
                                n_parents = n_parents, 
                                n_children = n_children, 
                                elitism = elitism)
            for individual in island_pop:
                individual.evaluate(env)   

        # Migration
        if migration_interval and i % migration_interval == 0:
            for i in range(n_islands):
                # Select a portion of individuals to migrate from island j to the next island
                n_migrants = int(len(islands[i]) * migration_rate)
                migrants = np.random.choice(islands[i], n_migrants, replace=False).tolist()
                target_island = islands[(i + 1) % n_islands]  # Wrap around to the first island

                # Add migrants to the next island and remove them from the current island
                target_island.extend(migrants)
                for migrant in migrants:
                    islands[i].remove(migrant)

        all_individuals = [individual for island in islands for individual in island] 
        best_of_generation = max(all_individuals, key=lambda x: x.fitness)
        if best_of_generation.fitness > best_individual.fitness:
            best_individual = best_of_generation
            generations_since_improvement = 0
        else:
             generations_since_improvement += 1

        # Doomsday
        if stagnation_threshold and generations_since_improvement >= stagnation_threshold: 
            all_individuals.sort(key=lambda x: x.fitness, reverse=True)  # Sort population by fitness
            survivors = int(npop * doomsday_survival_rate)
            new_population = all_individuals[:survivors]  # Keep only the top-performing individuals

            # Fill the rest with new random individuals
            new_random_individuals = [Individual(player_controller(n_hidden), list_learnable_params[ind]) for ind in range(npop - survivors)]
            pop_init = np.random.uniform(-1, 1, (npop - survivors, network_size))
            n_inputs = env.get_num_sensors()
            for j, individual in enumerate(new_random_individuals):
                individual.controller.set(pop_init[j], n_inputs)
                individual.evaluate(env)
            new_population.extend(new_random_individuals)

            # Redistribute the new population across islands
            np.random.shuffle(new_population)
            islands = []
            for j in range(n_islands):
                current_island_size = pop_per_island + (1 if j < remainder else 0)
                island_pop = new_population[:current_island_size]
                new_population = new_population[current_island_size:]  # Remove assigned individuals from list
                islands.append(island_pop)

            generations_since_improvement = 0  # Reset stagnation counter after Doomsday
    
    all_individuals = [individual for island in islands for individual in island] 
    return all_individuals, fit_tracker, best_individual.controller_to_genotype()