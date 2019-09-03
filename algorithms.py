#!/usr/bin/python3
"""
Evolutionary algorithms are implemented here.
"""
import sys
import numpy as np


class Parameter:
    def __init__(self, value, datatype):
        self.datatype = datatype

        self.value = value if value is None else datatype(value)

    def set(self, value):
        self.value = self.datatype(value)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class Algorithm:
    """
    In every epoch, an algorithm modifies given population.
    Evolution runs until any condition is met:
        1. Epoch count exceedes the limit (epoch_count).
        2. Good-enough solution has been found (desired_fitness).

    Parameters common to all algorithms:
        elitism - whether best known solution should be preserved
    """

    def __init__(self):
        self.params = {
            'epoch_count': Parameter(1000, int),
            'desired_fitness': Parameter(None, float),
            'elitism': Parameter(True, bool),
        }

    def update_params(self, params):
        for k, v in params.items():
            if isinstance(v, Parameter):
                self.params[k] = v
            else:
                try:
                    self.params[k].set(v)
                except:
                    self.params[k] = Parameter(v, str)

    def __str__(self):
        return "%s (%s)" % (self.__class__.__name__, self.params)


class HillClimbing(Algorithm):
    """
    In Hill Climbing algorithm every individual generate a number of
    individuals close to it. The best offspring is used.

    Parameters:
        spawn_range (float) - maximum distance between old and new individual
        spawn_count (int) - number of offsprings
    """

    def __init__(self, **user_params):
        super().__init__()
        default_params = {
            # visible distance
            'spawn_range': Parameter(1.0, float),
            # how many new spots to check per individual
            'spawn_count': Parameter(10, int)
        }
        self.update_params(default_params)
        self.update_params(user_params)

    def epoch(self, population, fitness):
        spawn_range = self.params['spawn_range'].value
        spawn_count = self.params['spawn_count'].value
        elitism = self.params['elitism'].value

        result = np.empty((0, population.shape[1]))
        # for each individual:
        for individual in population:
            # generate spawn_count close individuals
            changes = np.random.random(
                (spawn_count, population.shape[1])) * 2 * spawn_range - spawn_range + individual
            if elitism:
                changes = np.vstack((changes, individual))
            # use the best one
            fitness_values = fitness(changes.T)
            best_index = np.where(fitness_values == np.amin(fitness_values))[0]
            result = np.vstack((result, changes[best_index]))
        return result


class Annealing(Algorithm):
    """
    Simulated Annealing algorithm slightly alter every individual
    in a manner similar to HillClimbing. Better result is always accepted,
    worse result is randomly accepted (the chance drops every epoch).
    That means at the beginning local extremes can be overcame.

    Parameters:
        distance (float) - maximum distance between old and new individual
        temperature (float) - system temperature, drops over time
        cooling (float) - determines speed of temperature dropping
    """

    def __init__(self, **user_params):
        super().__init__()
        default_params = {
            # visible distance
            'distance': Parameter(1.0, float),
            # initial acceptance threshold
            'temperature': Parameter(1, float),
            # cooling multiplier
            'cooling': Parameter(0.95, float),
        }
        self.update_params(default_params)
        self.update_params(user_params)

    def epoch(self, population, fitness):
        distance = self.params['distance'].value
        temperature = self.params['temperature'].value
        cooling = self.params['cooling'].value
        elitism = self.params['elitism'].value

        fitness_population = fitness(population.T)
        differences = np.random.random(
            population.shape) * (2 * distance) - distance

        if elitism:
            # replace worst old with best old
            # set no difference for it
            best_old = np.where(fitness_population ==
                                np.amin(fitness_population))[0][0]
            worst_old = np.where(fitness_population ==
                                 np.amax(fitness_population))[0][0]
            population[worst_old] = population[best_old]
            differences[worst_old] = np.zeros(differences.shape[1])

        new = population + differences
        fitness_new = fitness(new.T)

        # use better
        new_better = (fitness_new < fitness_population).reshape(-1, 1)
        population = np.where(new_better, new, population)
        # use worse if lucky enough
        badluck = np.random.rand()

        def acceptance(old, new):
            return np.exp((fitness(old.T) - fitness(new.T)) / temperature) > badluck
        accepted = acceptance(population, new).reshape(-1, 1)
        lucky = np.logical_and(np.logical_not(new_better), accepted)
        population = np.where(lucky, new, population)
        # cooldown
        self.params['temperature'].set(temperature * cooling)
        return population


class TabuSearch(Algorithm):
    """
    Tabu Search works like HillClimbing, but last steps are remembered.
    This helps avoiding running in cycles.
    """
    pass


class SOMA(Algorithm):
    """
    Self-Organizing Migrating Algorithm
    """
    pass


class ES(Algorithm):
    """
    Evolutionary Strategy
    """
    pass


class AntColony(Algorithm):
    """
    Ant Colony Optimization
    """
    pass


class Genetic(Algorithm):
    """
    Genetic Algorithm
    """

    def __init__(self, **user_params):
        super().__init__()
        default_params = {
            # crossover mask
            'mask': Parameter(0x00000000ffffffff, int),
            # chance of mutation
            'mutation_chance': Parameter(0.01, float),
            # number of bits to mutate
            # or range of change
            'mutation_strength': Parameter(2, int),
        }
        self.update_params(default_params)
        self.update_params(user_params)

    def epoch(self, population, fitness):
        parent_count = 2
        mask = np.uint64(self.params['mask'].value)
        mutation_chance = self.params['mutation_chance'].value
        mutation_strength = self.params['mutation_strength'].value
        elitism = self.params['elitism'].value

        fitness_values = fitness(population.T)
        min_fitness = np.amin(fitness_values)
        max_fitness = np.amax(fitness_values)
        normalized = (fitness_values - min_fitness) / \
            (max_fitness - min_fitness)

        new = np.empty((0, population.shape[1]), dtype=np.float64)
        if elitism:
            new = np.vstack((new, population))

        for _ in range(len(population) // parent_count):
            # pick 2 parents
            parent_indices = set()
            while len(parent_indices) < parent_count:
                r = np.random.random()
                available = normalized[normalized > r]
                parent_index = np.where(normalized == np.amin(
                    available))[0][0]
                parent_indices.add(parent_index)
            # get parents as binary
            binary = population[list(parent_indices)].view(np.uint64)
            # get offsprings
            negmask = np.bitwise_xor(mask, np.uint64(0xffffffffffffffff))
            offsprings = np.array(
                [np.bitwise_and(binary[0], mask) + np.bitwise_and(binary[1], negmask),
                 np.bitwise_and(binary[1], mask) + np.bitwise_and(binary[0], negmask)]).view(np.float64)
            # mutate
            r = np.random.random(offsprings.shape) > mutation_chance
            """
            # mutation is done by adding a small number
            # bit flipping is not used, cause it gives crazy numbers
            
            mutations = np.fromfunction(
                np.vectorize(
                    lambda i, j: sum(1 << x for x in np.random.randint(1, 63, mutation_strength))), offsprings.shape).astype(np.uint64)
            offsprings ^= mutations
            """
            mutations = np.random.random(
                offsprings.shape) * mutation_strength - (mutation_strength/2)
            offsprings += np.where(r, mutations, 0)
            new = np.vstack((new, offsprings.astype(np.float64)))

        fitness_values = fitness(new.T)
        result = new[fitness_values.argsort()][: population.shape[0]]
        return result


class Immunology(Algorithm):
    """
    Immunology System Method
    """
    pass


class Memetic(Algorithm):
    """
    Memetic Algorithms
    """
    pass


class ScatterSearch(Algorithm):
    """
    Scatter Search
    """
    pass


class ParticleSwarm(Algorithm):
    """
    Particle Swarm
    """
    pass


class Differential(Algorithm):
    """
    Differential Evolution
    """
    pass
