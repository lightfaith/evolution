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
        limits - string in form min1,max1,min2,max2,... describing allowed values
    """

    def __init__(self):
        self.params = {
            'epoch_count': Parameter(1000, int),
            'desired_fitness': Parameter(None, float),
            'desired_diversity': Parameter(None, float),
            'elitism': Parameter(True, bool),
            'limits': Parameter('', str)
        }
        self.limits_min = None
        self.limits_max = None

    def update_params(self, params):
        for k, v in params.items():
            if isinstance(v, Parameter):
                self.params[k] = v
            else:
                try:
                    self.params[k].set(v)
                except:
                    self.params[k] = Parameter(v, str)

    def set_limits(self, population):
        if self.params['limits'].value:
            # use given limits
            parsed = np.array(
                [int(x) for x in self.params['limits'].value.split(',')[:population.shape[1] * 2]]).reshape(-1, 2)
            # if parsed.size != population.shape[1] * 2:
            #    print('Limits and population have different shape.', file=sys.stderr)
            self.limits_min = np.tile(parsed.T[0], (population.shape[0], 1))
            self.limits_max = np.tile(parsed.T[1], (population.shape[0], 1))
        else:
            # use min and max values
            self.limits_min = np.tile(
                np.amin(population, axis=0), (population.shape[0], 1))
            self.limits_max = np.tile(
                np.amax(population, axis=0), (population.shape[0], 1))
            # self.limits = np.vstack(
            #    (np.amin(population, axis=1), np.amax(population, axis=1))).T

    def enforce_limits(self, population, generate_random=True):
        if self.limits_min is None:
            self.set_limits(population)
        if generate_random:
            # generate random new if underflow/overflow
            random = np.random.random(population.shape) * \
                (self.limits_max - self.limits_min) + self.limits_min
            population = np.where(
                population < self.limits_min, random, population)
            population = np.where(
                population > self.limits_max, random, population)
        else:
            # use min/max if underflow/overflow
            # population = np.where(
            #    population < self.limits_min, self.limits_min, population)
            # population = np.where(
            #    population > self.limits_max, self.limits_max, population)
            population = np.clip(population, self.limits_min, self.limits_max)
        return population

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

        # set limits if not defined
        if self.limits_min is None:
            self.set_limits(population)

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

        # set limits if not defined
        if self.limits_min is None:
            self.set_limits(population)

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
        return self.enforce_limits(population)


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

    def __init__(self, **user_params):
        super().__init__()
        default_params = {
            'path_length': Parameter(3, float),
            'step': Parameter(0.11, float),
            'prt': Parameter(0.1, float),
        }
        self.update_params(default_params)
        self.update_params(user_params)

    def epoch(self, population, fitness):
        path_length = self.params['path_length'].value
        step = self.params['step'].value
        prt = self.params['prt'].value
        step_count = int(path_length // step)
        population_size, dimension = population.shape

        # set limits if not defined
        if self.limits_min is None:
            self.set_limits(population)

        # find best
        fitness_values = fitness(population.T)
        best = population[fitness_values.argsort()][0]
        step_coefs = np.tile(np.linspace(
            0, path_length, step_count).reshape(-1, 1), dimension)

        # get step values for each step for each individual
        big_shape = (population_size, step_count, dimension)
        step_coefs_tiled = np.tile(
            step_coefs, (population_size, 1)).reshape(big_shape)

        step_inc = best - population
        step_inc_tiled = np.repeat(
            step_inc, step_count, axis=0).reshape(big_shape)
        step_values = step_coefs_tiled * step_inc_tiled

        # generate perturbation vector for steps
        r = np.random.random(big_shape)
        prt_vector = np.where(r < prt, 1, 0)

        # compute final steps
        population_tiled = np.repeat(
            population, step_count, axis=0).reshape(big_shape)
        steps = population_tiled + step_values * prt_vector
        # find best step
        fitnesses = np.apply_along_axis(fitness, 2, steps)
        #bests = np.amin(fitnesses, axis=1).reshape(-1, 1)
        bests_indices = np.argmin(fitnesses, axis=1)
        #new = steps[:, bests_indices]
        #new = np.take(steps, bests_indices, axis=2)
        new = steps[[np.arange(population_size, dtype=int), bests_indices]]
        return self.enforce_limits(new, generate_random=False)


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
            # 'mask': Parameter(0x00000000ffffffff, int),
            # chance of mutation
            'mutation_chance': Parameter(0.01, float),
            # number of bits to mutate
            # or range of change
            'mutation_strength': Parameter(2, int),
            # whether parameters should be swapped instead of
            # slightly altered on mutation
            # good for TSP
            'mutation_swap': Parameter(False, bool),
        }
        self.update_params(default_params)
        self.update_params(user_params)

    def epoch(self, population, fitness):
        parent_count = 2
        # mask = np.uint64(self.params['mask'].value)
        mutation_chance = self.params['mutation_chance'].value
        mutation_strength = self.params['mutation_strength'].value
        mutation_swap = self.params['mutation_swap'].value
        elitism = self.params['elitism'].value

        # set limits if not defined
        if self.limits_min is None:
            self.set_limits(population)

        fitness_values = fitness(population.T)
        min_fitness = np.amin(fitness_values)
        max_fitness = np.amax(fitness_values)
        # normalized = (fitness_values - min_fitness) / \
        #    (max_fitness - min_fitness)

        new = np.empty((0, population.shape[1]), dtype=np.float64)
        if elitism:
            new = np.vstack((new, population))

        for _ in range(len(population) // parent_count):
            """
            # pick 2 parents
            parent_indices = set()
            while len(parent_indices) < parent_count:
                r = np.random.random()
                available = normalized[normalized > r]
                parent_index = np.where(normalized == np.amin(
                    available))[0][0]
                if parent_index in parent_indices:
                    try:
                        parent_index = np.where(normalized == np.amin(
                            available))[0][len(parent_indices)]
                    except:
                        pass
                parent_indices.add(parent_index)
            # get parents as binary
            """
            parent_indices = (np.random.beta(
                1, 3, size=parent_count) * population.shape[0]).astype(int)
            binary = population[list(parent_indices)].view(np.uint64)
            # generate crossover mask
            mask = np.zeros(64, dtype=int)
            mask[np.random.randint(64):] = 1
            mask = mask.dot(1 << np.arange(mask.size)[::-1]).astype(np.uint64)
            negmask = np.bitwise_xor(mask, np.uint64(0xffffffffffffffff))
            # get offsprings

            offsprings = np.array(
                [np.bitwise_and(binary[0], mask) + np.bitwise_and(binary[1], negmask),
                 np.bitwise_and(binary[1], mask) + np.bitwise_and(binary[0], negmask)]).view(np.float64)
            # mutate
            if mutation_swap:
                # mutate by swapping x parameters
                to_mutate = np.random.random(
                    offsprings.shape[0]) < mutation_chance
                for _, offspring_index in np.ndenumerate(np.where(to_mutate)):
                    a, b = np.random.randint(0, offsprings.shape[1], 2)
                    tmp = offsprings[offspring_index][a]
                    offsprings[offspring_index][a] = offsprings[offspring_index][b]
                    offsprings[offspring_index][b] = tmp

            else:
                # mutate by adding a small number
                r = np.random.random(offsprings.shape) < mutation_chance
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
        return self.enforce_limits(result)


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

    def __init__(self, **user_params):
        super().__init__()
        default_params = {
            'c1': Parameter(2.0, float),
            'c2': Parameter(2.0, float),
            'neighborhood': Parameter(0, int),
        }
        self.update_params(default_params)
        self.update_params(user_params)

        self.current_speed = None
        self.gbests = None  # global best of all time (for group)
        self.gbests_fitness = None
        self.pbests = None  # personal bests
        self.pbests_fitness = None

    def epoch(self, population, fitness):
        # TODO some elitism?
        c1 = self.params['c1'].value
        c2 = self.params['c2'].value
        neighborhood = self.params['neighborhood'].value % population.shape[0]

        # set limits if not defined
        if self.limits_min is None:
            self.set_limits(population)

        if self.current_speed is None:
            # first run; generate speed, use population as pbest
            self.current_speed = np.random.random(
                population.shape) * 2 - 1  # TODO param for generation range?
            self.pbests = np.copy(population)
        else:
            # nth run; get pbest
            fitness_values = fitness(population.T)
            self.pbests = np.where(fitness_values.reshape(-1, 1) <
                                   self.pbests_fitness.reshape(-1, 1), self.pbests, population)
        self.pbests_fitness = fitness(self.pbests.T)

        # gest gbest (from whole population or closest individuals)
        if neighborhood == 0:  # no neighborhood
            # find total global best, use tile to get
            # same shape as neighborhood variant
            self.gbests_fitness = np.tile(
                np.amin(self.pbests_fitness), self.pbests.shape[0])
            self.gbests = np.tile(
                self.pbests[np.where(self.pbests_fitness == self.gbests_fitness[0])[0]], self.pbests.shape[0]).reshape(self.pbests.shape)
        else:
            # with neighborhood
            # get individual index offset
            offsets = np.arange(neighborhood, dtype=int) - neighborhood // 2
            # get neighbors together for each individual
            neighborships = np.hstack([np.roll(self.pbests, o) for o in offsets]).reshape(
                population.shape[0], -1, population.shape[1])
            # get fitness for independent neighborships
            fitnesses = np.apply_along_axis(fitness, 2, neighborships)
            # get order of best fitness in neighborship
            orders = np.where(np.argsort(fitnesses) == 0)[1]
            # use the order to get real index in pbest
            orders = (
                (orders + np.arange(self.pbests.shape[0], dtype=int)) % self.pbests.shape[0]).reshape(-1, 1)
            # get correct individuals
            self.gbests = np.take(self.pbests, orders)
            # compute their fitness
            self.gbests_fitness = fitness(self.gbests.T)

        # update current speed
        self.current_speed += c1 * np.random.random() * (self.pbests - population) + \
            c2 * np.random.random() * (self.gbests - population)
        # update population
        population += self.current_speed
        return self.enforce_limits(population)


class Differential(Algorithm):
    """
    Differential Evolution
    """
    pass
