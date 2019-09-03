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
    """

    def __init__(self):
        self.params = {
            'epoch_count': Parameter(1000, int),
            'desired_fitness': Parameter(None, float),
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

        result = np.empty((0, population.shape[1]))
        # for each individual:
        for individual in population:
            # generate spawn_count close individuals
            changes = np.random.random(
                (spawn_count, population.shape[1])) * 2 * spawn_range - spawn_range + individual
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

        differences = np.random.random(
            population.shape) * (2*distance) - distance
        new = population + differences
        fitness_population = fitness(population.T)
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
    pass


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
