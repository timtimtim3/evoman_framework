import numpy as np
from optimization_utils import mutate
from demo_controller import player_controller
from optimization_utils import mutate, crossover_avg
from optimization_utils import crossover_recombination

def test_mutate():
    # Create a sample controller
    controller = player_controller(10)
    controller.bias1 = np.zeros((10,))
    controller.bias2 = np.zeros((10,))
    controller.weights1 = np.zeros((10, 10))
    controller.weights2 = np.zeros((10, 10))

    # Mutate the controller
    mutated_controller = mutate(controller, std=0.1, mutation_rate=0.2)

    # Check that the mutated controller is not the same as the original
    assert not np.array_equal(controller.bias1, mutated_controller.bias1), "bias1 should be mutated"
    assert not np.array_equal(controller.bias2, mutated_controller.bias2), "bias2 should be mutated"
    assert not np.array_equal(controller.weights1, mutated_controller.weights1), "weights1 should be mutated"
    assert not np.array_equal(controller.weights2, mutated_controller.weights2), "weights2 should be mutated"

    # Check that the mutated values are within the expected range
    assert np.all(np.abs(mutated_controller.bias1) <= 0.3), "bias1 values should be within the expected range"
    assert np.all(np.abs(mutated_controller.bias2) <= 0.3), "bias2 values should be within the expected range"
    assert np.all(np.abs(mutated_controller.weights1) <= 0.3), "weights1 values should be within the expected range"
    assert np.all(np.abs(mutated_controller.weights2) <= 0.3), "weights2 values should be within the expected range"

def test_avg_crossover():
    # Create two sample controllers
    controller1 = player_controller(10)
    controller2 = player_controller(10)
    controller1.bias1 = np.random.rand(10)
    controller1.bias2 = np.random.rand(10)
    controller1.weights1 = np.random.rand(10, 10)
    controller1.weights2 = np.random.rand(10, 10)
    controller2.bias1 = np.random.rand(10)
    controller2.bias2 = np.random.rand(10)
    controller2.weights1 = np.random.rand(10, 10)
    controller2.weights2 = np.random.rand(10, 10)

    # Perform average crossover
    offspring = crossover_avg((controller1, controller2))

    # Check that the offspring's parameters are the average of the parents' parameters
    assert np.array_equal(offspring.bias1, (controller1.bias1 + controller2.bias1) / 2), "bias1 should be the average of the parents"
    assert np.array_equal(offspring.bias2, (controller1.bias2 + controller2.bias2) / 2), "bias2 should be the average of the parents"
    assert np.array_equal(offspring.weights1, (controller1.weights1 + controller2.weights1) / 2), "weights1 should be the average of the parents"
    assert np.array_equal(offspring.weights2, (controller1.weights2 + controller2.weights2) / 2), "weights2 should be the average of the parents"

def test_crossover_recombination():
    # Create two sample controllers
    controller1 = player_controller(10)
    controller2 = player_controller(10)
    controller1.bias1 = np.random.rand(10)
    controller1.bias2 = np.random.rand(10)
    controller1.weights1 = np.random.rand(10, 10)
    controller1.weights2 = np.random.rand(10, 10)
    controller2.bias1 = np.random.rand(10)
    controller2.bias2 = np.random.rand(10)
    controller2.weights1 = np.random.rand(10, 10)
    controller2.weights2 = np.random.rand(10, 10)

    # Perform recombination crossover
    offspring = crossover_recombination((controller1, controller2))

    # Check that the offspring's parameters are a recombination of the parents' parameters
    assert np.all((offspring.bias1 == controller1.bias1) | (offspring.bias1 == controller2.bias1)), "bias1 should be a recombination of the parents"
    assert np.all((offspring.bias2 == controller1.bias2) | (offspring.bias2 == controller2.bias2)), "bias2 should be a recombination of the parents"
    assert np.all((offspring.weights1 == controller1.weights1) | (offspring.weights1 == controller2.weights1)), "weights1 should be a recombination of the parents"
    assert np.all((offspring.weights2 == controller1.weights2) | (offspring.weights2 == controller2.weights2)), "weights2 should be a recombination of the parents"

    # Check that the offspring is not the same as one of the parents
    assert not np.array_equal(offspring.bias1, controller1.bias1), "offspring should not be the same as parent 1"
    assert not np.array_equal(offspring.bias1, controller2.bias1), "offspring should not be the same as parent 2"
    assert not np.array_equal(offspring.bias2, controller1.bias2), "offspring should not be the same as parent 1"
    assert not np.array_equal(offspring.bias2, controller2.bias2), "offspring should not be the same as parent 2"
    assert not np.array_equal(offspring.weights1, controller1.weights1), "offspring should not be the same as parent 1"
    assert not np.array_equal(offspring.weights1, controller2.weights1), "offspring should not be the same as parent 2"
    assert not np.array_equal(offspring.weights2, controller1.weights2), "offspring should not be the same as parent 1"
    assert not np.array_equal(offspring.weights2, controller2.weights2), "offspring should not be the same as parent 2"
if __name__ == "__main__":
    test_mutate()
    test_avg_crossover()
    test_crossover_recombination()
    print("All tests passed.")




