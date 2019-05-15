#!/usr/bin/python3

# ----------------------------
# Author: Ravexina (Milad As)
# ----------------------------
# https://github.com/ravexina/Binary-Genetic-Algorithm 

import numpy as np
from matplotlib import pyplot as plt


class BGA:

    def __init__(self, dimension=2, pop_size=50, pc=0.9, pm=0.005, bit_length=(10, 10),
            low=(-4, -1.5), hi=(2, 1), max_iter=100, log=False):

        # Parameters
        self.pm = pm
        self.pc = pc
        self.bit_length = bit_length
        self.chromosome_length = sum(bit_length)  # Li
        self.pop_size = pop_size  # It's actually N
        self.dimension = dimension  # And this one is m

        # Values range
        self.hi = hi
        self.low = low

        self.log = log
        self.max_iter = max_iter

        # Fitness function
        # Should be set by user
        self.fitness_function = None

        # Will be set during lunch
        self.population_fitness = None
        self.decoded_population = None

        # To keep track of algorithm
        self.best_so_far = []
        self.average_fitness = []

        # Generate random population
        self.population = np.random.randint(0, 2, (self.pop_size, self.chromosome_length))

    def set_fitness(self, fitness_func):
        self.fitness_function = fitness_func

    def selection(self, N, population, probability):
        pool = []
        for i in range(N):
            # len(population) can also be self.pop_size
            index = np.random.choice(len(population), p=probability)
            pool.append(population[index])

        return np.array(pool)

    def crossover(self, pool):
        new_population = []
        for i in range(0, len(pool), 2):
            child_1 = pool[i]
            child_2 = pool[i+1]
        
            # Do the cross over with the probability of Pc
            if np.random.uniform() < self.pc:
                cut_point = np.random.randint(1, self.chromosome_length)
                temp = child_2[:cut_point].copy()
                child_2[:cut_point], child_1[:cut_point] = child_1[:cut_point], temp

            new_population.extend([child_1, child_2])

        return new_population


    # Works directly on self.population
    def mutation(self):
        mask = np.random.choice([0, 1], (self.pop_size, self.chromosome_length), p=[1-self.pm, self.pm])
        self.population = np.bitwise_xor(self.population, mask)

        # WORKS FINE !! -- HOWEVER I EXCHANGED IT FOR MORE PYTHONIC WAY
        #
        # mask = np.random.randint(0, 2, (self.pop_size, self.chromosome_length))
        # for i in range(self.pop_size):
        #    for j in range(self.chromosome_length):
        #           if mask[i][j] == 1 and np.random.uniform() < self.pm:
        #            self.population[i][j] = 1 if self.population[i][j] == 0 else 0
            
    def decode(self, chromosome):
        def normal(gene):
            pow2_list = np.array([2 ** i for i in range(0, len(gene))])[::-1]
            decimal = np.sum(pow2_list * gene)
            normalized = decimal / (2 ** len(gene) - 1)
            return normalized

        # Split chromosome to its genes
        # [0 1 0 1] : Bit length (2 2) > Would become [[0 1] [0 1]]
        genes = np.split(chromosome, np.cumsum(self.bit_length)[:-1])

        # Normalize genes between 0-1
        normalized_genes = list(map(normal, genes))

        # Bring it to common range (low - high)
        decoded_chromosome = np.full((self.dimension), 0, dtype=float)
        for i in range(self.dimension):
            xi = normalized_genes[i]
            xi = self.low[i] + (self.hi[i] - self.low[i]) * xi
            decoded_chromosome[i] = xi

        return decoded_chromosome

    def decode_population(self):
        self.decoded_population = np.array(
            [self.decode(chromosome) for chromosome in self.population])

    def calculate_fitness(self):
        self.population_fitness = np.array(
            [self.fitness_function(chromosome) for chromosome in self.decoded_population])

    def update_best_so_far(self, max_fitness, solution):
        if len(self.best_so_far) == 0:
            self.best_so_far.append(max_fitness)
            self.best_so_far_solution = solution
            return True

        if self.best_so_far[-1] < max_fitness:
            self.best_so_far.append(max_fitness)
            self.best_so_far_solution = solution
        else:
            self.best_so_far.append(self.best_so_far[-1])

    def start(self, plot):
        for iteration in range(self.max_iter):
            # Decode chromosomes
            self.decode_population()

            # Calculate fitness of population and related probabilities
            self.calculate_fitness()

            # Find best solution and fitness yet far
            max_fitness = np.max(self.population_fitness)
            index = np.where(self.population_fitness == max_fitness)[0][0]
            optimal_solution = self.decoded_population[index]

            # Update statics
            self.update_best_so_far(max_fitness, optimal_solution)
            self.average_fitness.append(np.mean(self.population_fitness))
            
            if plot and iteration in [1, 20, 50]:
                x, y = self.decoded_population.T
                plt.scatter(x, y)

            if self.log:
                print('Iteration:', iteration)
                print('MaxFitness:', max_fitness, 'Solution:', optimal_solution)
                print('Best so far:', self.best_so_far[-1], 'At:', list(self.best_so_far_solution))
                print('-' * 70)

            # Calculate selection probability
            selection_probability = self.population_fitness / np.sum(self.population_fitness)

            pool = self.selection(self.pop_size, self.population, selection_probability)

            self.population = self.crossover(pool)

            self.mutation()

        if plot:
            print('ploting...')
            plt.legend(('Gen: 1', 'Gen: 2', 'Gen: 3'))
            plt.show()

        return self.best_so_far, self.average_fitness

def func(chromosome):
    # F Optimal is 2 for X* = (0,0)
    x1, x2 = chromosome
    return (1 + np.cos(2 * np.pi * x1 * x2)) * np.exp(- 0.5 * (abs(x1) + abs(x2)))

best_so_far = []
average = []
for i in range(10):
    plot = False
    if i == 1:
        plot = True
    
    my_ga = BGA()
    my_ga.set_fitness(func)
    bsf, avrg = my_ga.start(plot)
    best_so_far.append(bsf)
    average.append(avrg)

best_so_far = np.array(best_so_far)
average = np.array(average)
best_so_far = np.mean(best_so_far, axis=0)
average = np.mean(average, axis=0)

plt.plot(best_so_far)
plt.plot(average)
plt.legend(('Best so far', 'Average'))
plt.show()
