import array

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import numpy as np
import random


class ES:
    def __init__(self, evaluate_function, ind_size, min_val, max_val, min_strat, max_strat):
        self.evaluate = evaluate_function
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
        creator.create("Strategy", array.array, typecode="d")
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.generateES, creator.Individual, creator.Strategy,
                              ind_size, min_val, max_val, min_strat, max_strat)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=10)
        self.toolbox.register("evaluate", self.evaluate)

    @staticmethod
    def generateES(icls, scls, size, imin, imax, smin, smax):
        ind = icls(random.uniform(imin, imax) for _ in range(size))
        ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
        return ind

    def run_algorithm(self, cxpb=0.5, mutpb=0.2, ngen=50, mu=40):
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop = self.toolbox.population(n=mu)
        bestGen = algorithms.eaSimple(pop, self.toolbox, cxpb, mutpb, ngen, stats)
        return bestGen
