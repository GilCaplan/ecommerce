import numpy

from utils import *

import numpy as np


def mutate(s):
    new_s = s.copy()
    i = np.random.randint(10)

    new_s[i] = nodes_100[np.random.randint(len(nodes_100))]

    return new_s


def crossover(s1, s2):
    return np.random.permutation(np.union1d(s1, s2))[:10]


def partition():
    pass


if __name__ == '__main__':
    graph = read_graph()
    nodes_100 = list(filter(lambda node: get_cost(node) == 100, graph.nodes))

    N = 100
    n_iters = 500

    alpha = 0.5
    beta = 0.25

    print(f"N = {N}")
    print(f"n_iter = {n_iters}")
    print(f"alpha = {alpha}")
    print(f"beta = {beta}")

    population = np.array([np.random.choice(nodes_100, size=10, replace=False) for _ in range(N)])

    best_ind = 0
    best_score = 0
    best = np.array([])

    for k in range(n_iters):
        pop_scores = np.array([calc_score(graph, set(s), num_trials=10) for s in population])

        n_survived = int(alpha * N)
        n_died = N - n_survived

        partition_ind = numpy.argpartition(pop_scores, n_died)
        survived_indexes = partition_ind[n_died:]
        died_indexes = partition_ind[:n_died]

        if k != 0:
            print(f"Round {k} result")
            print(f"  min: {np.min(pop_scores)}")
            print(f"  max: {np.max(pop_scores)}")
            print(f"  median: {np.percentile(pop_scores, 0.5)}")
            best_ind = np.argmax(pop_scores)
            best = population[best_ind]
            best_score = pop_scores[best_ind]
            print(f"  {list(best)} -> {best_score}")

        for ind in died_indexes:
            parent_indexes = np.random.choice(survived_indexes, size=2, replace=False)

            parent1 = population[parent_indexes[0]]
            parent2 = population[parent_indexes[1]]

            child = crossover(parent1, parent2)

            population[ind] = child

        n_mutate = int(beta * N)
        indexes_to_mutate = np.random.choice(range(N), n_mutate, replace=False)

        for i in indexes_to_mutate:
            population[i] = mutate(population[i])

    print("Entire Population:")
    print(population)