import sys
import numpy as np
import random

# Neural Network Class
# Neural network with one layer
class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # weights and bias are randomly selected
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)

    # Given an example X, run it through the network to determinate it's classification
    def forward(self, X):
        # use activation function tanh
        weighted_sum = np.tanh(np.dot(self.weights, X) + self.bias)
        output = 1 if weighted_sum >= 0 else 0
        return output


# Genetic Algorithm Class
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    # Creates two children from two parents.
    # Select an index x randomly and then creates 2 children.
    def crossover(self, parent1, parent2):
        child1 = NeuralNetwork(input_size=parent1.input_size, output_size=parent1.output_size)
        child2 = NeuralNetwork(input_size=parent1.input_size, output_size=parent1.output_size)

        num = random.randint(0, 15) # the index that randomly selected

        # Copy weights from parent1 up to the selected index (num)
        child1.weights[:num] = parent1.weights[:num]

        # Copy weights from parent2 after crossover point
        child1.weights[num:] = parent2.weights[num:]

        # Copy weights from parent2 up to crossover point
        child2.weights[:num] = parent2.weights[:num]

        # Copy weights from parent1 after crossover point
        child2.weights[num:] = parent1.weights[num:]

        # Select a random number 0/1.
        # if the number is 1, child1 has the same bias value as parent1, and child2 has same bias value as parent2,
        # else, child1 has the same bias value as parent2, and child2 has same bias value as parent1
        if random.randint(0, 1):
            child1.bias = parent1.bias
            child2.bias = parent2.bias
        else:
            child1.bias = parent2.bias
            child2.bias = parent1.bias
        return child1, child2

    # The function mutates the weights and biases, with a probability of mutation_rate (0.2) for each.
    # Generate a random index x (n1) between 0 to 15.
    # Add the weight which is in the index x, a random number between -1 to 1.
    # If the result is larger than 1 or smaller than -1, round it.
    # The mutation do the same for the bias.
    def mutate(self, network):
        # with a probability of mutation_rate
        if random.random() < self.mutation_rate:

            n1 = random.randint(0, network.input_size - 1)
            mutated_weight = network.weights[n1] + random.uniform(-1, 1)
            network.weights[n1] = np.clip(mutated_weight, -1, 1)

        # with a probability of mutation_rate
        if random.random() < self.mutation_rate:
            mutated_bias = network.bias + random.uniform(-1, 1)
            network.bias = np.clip(mutated_bias, -1, 1)

    # Sort the solutions (neural network) according to their fitness value.
    # Chooses two solutions randomly where the higher the fitness score of a solution,
    # the higher its chance of being selected
    def select_parents(self, population, fitness_scores):
        dic = {}
        for i in range(len(population)):
            dic[i] = (population[i], fitness_scores[i])
        # sort the dictionary according to the fitness score
        sorted_dict = dict(sorted(dic.items(), key=lambda x: x[1][1]))
        probabilities = []
        l_population = []
        for i, key in enumerate(sorted_dict):
            l_population.append(sorted_dict[key][0])
            probabilities.append(i)
        # Chooses two solutions randomly in accordance with probabilities
        parents = random.choices(l_population, weights=probabilities, k=2)
        return parents[0], parents[1]

    # Create the new population for the next generation
    def evolve_population(self, population, fitness_scores, reflection_size):
        new_population = []
        dic = {}
        # Reflection - sort the population and pick the best ones for the next generation
        for i in range(len(population)):
            f = fitness_function(train_data, population[i])
            dic[i] = (population[i], f)
        sorted_dict = dict(sorted(dic.items(), key=lambda x: x[1][1]))
        sorted_dict = dict(list(sorted_dict.items())[-reflection_size:])
        # Passes the best solutions to the next generation (reflection)
        for key in sorted_dict:
            new_population.append(sorted_dict[key][0])
        # generate a new solutions
        for _ in range(new_solutions):
            new_population.append(NeuralNetwork(input_size=16, output_size=1))
        # The number of times to perform cross over
        r = int((self.population_size - reflection_size - new_solutions) / 2)
        for _ in range(r):
            # select 2 parents for the cross over
            parent1, parent2 = self.select_parents(population, fitness_scores)
            child1, child2 = self.crossover(parent1, parent2)
            # Mutation operation with a high probability of 0.2
            self.mutate(child1)
            self.mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        return new_population


# Read the dataset from the text file
def read_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        for line in file:
            binary_string, label = line.split()
            dataset.append((binary_string, int(label)))
    return dataset


# Split the dataset into train and test sets
def split_dataset(dataset, train_ratio):
    random.shuffle(dataset)
    num_train = int(len(dataset) * train_ratio)
    train_data = dataset[:num_train]
    test_data = dataset[num_train:]
    return train_data, test_data


# Fitness function for evaluating individuals
# Count how many examples the network classified correctly, and divide it by the number of examples
def fitness_function(train_data, neural_network):
    correct_predictions = 0
    for binary_string, label in train_data:
        output = neural_network.forward(np.array(list(binary_string), dtype=int))
        if output == label:
            correct_predictions += 1
    fitness = correct_predictions / len(train_data)
    return fitness

# Genetic Algorithm parameters
population_size = 100
reflection_size = 15
mutation_rate = 0.2
generations = 40
new_solutions = 5
population = []


# # Read the dataset from the text file
# dataset = read_dataset('nn0.txt')
#
#
# # Split the dataset into train and test sets
# train_data, test_data = split_dataset(dataset, train_ratio=0.75)
#
# with open('train_data0.txt', 'w') as f:
#     for item in train_data:
#         line = " ".join(str(i) for i in item)  # Convert tuple elements to strings and join with space
#         f.write(line + "\n")
#
# with open('test_data0.txt', 'w') as f:
#     for item in test_data:
#         line = " ".join(str(i) for i in item)  # Convert tuple elements to strings and join with space
#         f.write(line + "\n")


train = sys.argv[1]
test = sys.argv[2]
# Read the dataset from the text file
train_data = read_dataset(train)

# Read the dataset from the text file
test_data = read_dataset(test)

# Create the neural network
for _ in range(population_size):
    neural_network = NeuralNetwork(input_size=16, output_size=1)
    population.append(neural_network)

genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate)
# Run the genetic algorithm
for generation in range(generations):
    # Each epoch works with small batch. after each batch, the network is being updated
    random.shuffle(train_data)
    for i in range(0, len(train_data), 1000):
        train_data_batch = train_data[i:i + 1000]
        # Evaluate the fitness of each individual in the population
        fitness_scores = [fitness_function(train_data_batch, nnetwork) for nnetwork in population]
        population = genetic_algorithm.evolve_population(population, fitness_scores, reflection_size)

    # Select the best network
    best_individual_index = np.argmax(fitness_scores)
    best_individual = population[best_individual_index]
    print(best_individual)
    print(f"Generation {generation + 1}: Best Individual: {best_individual}, Fitness: {fitness_scores[best_individual_index]}")
    # Evolve the population
    population = genetic_algorithm.evolve_population(population, fitness_scores, reflection_size)

# Evaluate the best solution (neural network) on the test set
best_individual_fitness = fitness_function(test_data, best_individual)
print(f"\nBest Individual Fitness on Test Set: {best_individual_fitness}")
weights_array = best_individual.weights
bias = best_individual.bias
# Write the learned weights and bias to a file
with open('wnet0.npy', 'wb') as f:
    np.save(f, np.array(weights_array))
    np.save(f, np.array(bias))