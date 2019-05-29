import array
import random

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

NUM_CLUSTER = 3
MIN_VALUE = 4
MAX_VALUE = 5
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3
IND_SIZE = 2 * NUM_CLUSTER
GAMA = 2
X = np.random.uniform(4, 5, (200, 2))
y = np.random.uniform(4, 5, (200, 1))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")


# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind


def evaluate(individual):
    y_star = calculate_G(individual)
    loss = 0.5 * np.linalg.norm(y - y_star)
    print(loss)
    return loss,


def calculate_G(individual):
    V_ind = np.reshape(individual, (-1, 2))
    G_matrix = np.zeros((X.shape[0], V_ind.shape[1]))
    for i in range(X.shape[0]):
        for j in range(V_ind.shape[1]):
            mantis = np.dot(X[i], V_ind[j])
            G_matrix[i, j] = np.exp(-GAMA * mantis)
    W_matrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(G_matrix.transpose(), G_matrix)), G_matrix.transpose()), y)
    y_star = np.matmul(G_matrix, W_matrix)
    return y_star


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


toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                 IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)
toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))


def main():
    random.seed()
    MU, LAMBDA = 10, 100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    # stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=0.6, mutpb=0.3, ngen=500, stats=stats, halloffame=hof)

    return pop, logbook, hof


if __name__ == "__main__":
    print((main()[2]))
