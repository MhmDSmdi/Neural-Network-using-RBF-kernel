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
        creator.create("FitnessMin", base.Fitness, weights=[-1])
        creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
        creator.create("Strategy", array.array, typecode="d")
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.generateES, creator.Individual, creator.Strategy,
                              ind_size, min_val, max_val, min_strat, max_strat)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
        self.toolbox.register("select", tools.selTournament, tournsize=10)
        self.toolbox.register("evaluate", self.evaluate)

        self.toolbox.decorate("mate", self.checkStrategy(min_strat))
        self.toolbox.decorate("mutate", self.checkStrategy(max_strat))

    @staticmethod
    def generateES(icls, scls, size, imin, imax, smin, smax):
        ind = icls(random.uniform(imin, imax) for _ in range(size))
        ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
        return ind

    @staticmethod
    def crossover(ind1, ind2):
        res1, res2 = tools.cxESBlend(ind1, ind2, alpha=0.1)
        return res1, res2

    @staticmethod
    def checkStrategy(minstrategy):
        def decorator(func):
            def wrappper(*args, **kargs):
                children = func(*args, **kargs)
                for child in children:
                    for i, s in enumerate(child.strategy):
                        if s < minstrategy:
                            child.strategy[i] = minstrategy
                return children

            return wrappper

        return decorator

    def run_algorithm(self, cxpb=0.5, mutpb=0.2, ngen=20, mu=10):
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop = self.toolbox.population(n=mu)
        pop, logbook = algorithms.eaMuCommaLambda(pop, self.toolbox, mu=mu, lambda_=100,
                                                  cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof)
        print(self.evaluate(hof[0])[0])
        return pop, hof
