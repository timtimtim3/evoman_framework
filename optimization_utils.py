from demo_controller import player_controller
import numpy

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

def crossover_avg(controllers):
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
    for controller in controllers:
        avg_bias1    += controller.bias1 /n
        avg_bias2    += controller.bias2 /n
        avg_weights1 += controller.weights1 /n
        avg_weights2 += controller.weights2 /n
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
        
        # Iterate over each element in the matrices
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                # Choose a random matrix from x
                I = numpy.random.choice(range(len(x)))
                random_matrix = x[I]
                # Get the element at the current position
                element = random_matrix[i, j]
                # Assign the element to the recombined matrix
                recombined_matrix[i, j] = element
    elif len(matrix_shape)==1:
        # Iterate over each element in the matrices
        for i in range(matrix_shape[0]):
            # Choose a random matrix from x
            I = numpy.random.choice(range(len(x)))
            random_matrix = x[I]
            # Get the element at the current position
            element = random_matrix[i]
            # Assign the element to the recombined matrix
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


