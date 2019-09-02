#!/usr/bin/python3
import sys
import numpy as np
"""
Evolutionary algorithms are implemented here.
"""


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
    def __init__(self):
        self.params = {
            'epoch_count': Parameter(1000, int),
            'desired_fitness': Parameter(None, float),
        }

    def epoch(self, population, fitness):
        print("Abstract algorithm cannot run!")

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
