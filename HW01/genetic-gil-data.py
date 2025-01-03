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
    # first list
    # nodes = [1051, 3654, 809, 2278, 1263, 1019, 3820, 384, 640, 2154, 777, 2298, 3448, 2295, 2868, 3851, 284, 2368, 2670, 3040, 2679, 3384, 1969, 24, 848, 367, 697, 2298, 2207, 2480, 1625, 619, 1995, 4026, 2035, 987, 2255, 3961, 975, 1478, 1608, 3299, 3117, 1769, 3439, 2389, 925, 2400, 566, 207, 3936, 2924, 1524, 600, 454,  1104, 924, 2591, 3173, 1809, 2215]

    # long list
    nodes = [1025, 3073, 3076, 1028, 1030, 1034, 525, 1037, 2576, 532, 3094, 2583, 24, 2584, 1051, 1055, 546, 2082, 3620, 38, 3112, 3115, 2095, 560, 3124, 3125, 1589, 1080, 3642, 1596, 2623, 2113, 1605, 3654, 1094, 71, 1846, 3144, 3659, 1100, 76, 581, 1616, 81, 1110, 3671, 2650, 1116, 606, 3681, 610, 617, 2154, 3177, 2670, 1651, 628, 2679, 632, 3708, 125, 640, 132, 644, 135, 1160, 2697, 136, 1675, 3722, 3894, 2191, 3731, 1177, 153, 1180, 2718, 3230, 2208, 3745, 1705, 2217, 169, 3758, 2224, 1719, 699, 1215, 707, 1220, 2245, 197, 1224, 200, 719, 3792, 2771, 3798, 729, 3802, 732, 3293, 736, 229, 3303, 2282, 746, 1259, 3820, 2799, 1264, 1263, 3316, 3829, 3318, 2295, 3320, 245, 1274, 2298, 1268, 259, 3844, 2821, 2824, 777, 3851, 1804, 3856, 2833, 274, 3862, 1817, 795, 284, 798, 2850, 291, 806, 809, 3884, 3372, 814, 2350, 815, 1842, 3380, 2357, 2868, 3895, 3384, 3892, 1338, 2359, 1340, 317, 318, 3391, 319, 2881, 322, 2368, 1855, 3926, 1878, 2394, 2395, 350, 1887, 3424, 3942, 1894, 872, 1897, 2409, 1386, 2412, 367, 2421, 3448, 1401, 378, 1407, 384, 3461, 2950, 1421, 397, 1936, 1937, 915, 3989, 3995, 2972, 1948, 2974, 3485, 932, 2473, 1453, 2478, 943, 1455, 2480, 3002, 954, 449, 2498, 1477, 2502, 2503, 3015, 1483, 3027, 2516, 3032, 2530, 3042, 3557, 3045, 3047, 498, 3062, 1019, 511]

    nodes_100 = list(filter(lambda node: get_cost(node) == 100, nodes))

    N = 40
    n_iters = 50

    alpha = 0.5
    beta = 0.4

    print(f"N = {N}")
    print(f"n_iter = {n_iters}")
    print(f"alpha = {alpha}")
    print(f"beta = {beta}")

    population = np.array([np.random.choice(nodes_100, size=10, replace=False) for _ in range(N-7)])
    arr1 = np.array([3654, 2298, 2670, 640, 2295, 3448, 1969, 1051, 3851, 3040])
    arr2 = np.array([3654, 2298, 1263, 640, 2295, 3448, 1969, 1051, 3851, 3040])
    arr3 = np.array([1051, 3851, 3654, 1019, 697, 848, 3448, 2480, 640, 2298])
    arr4 = np.array([848, 1051, 3851, 3654, 1019, 697, 3448, 2480, 640, 24])
    arr5 = np.array([384, 640, 3851, 1051, 284, 3654, 848, 1263, 3448, 1019])
    arr6 = np.array([384, 640, 3851, 1051, 2868, 3654, 848, 3448, 2298, 1019])
    arr7 = np.array([384, 640, 3851, 1051, 3654, 848, 1263, 3448, 2298, 1019])
    population = np.append(population, [arr1, arr2, arr3, arr4, arr5, arr6, arr7], axis=0)
    best_ind = 0
    best_score = 0
    best = np.array([])

    for k in range(n_iters):
        pop_scores = np.array([calc_score(graph, set(s), num_trials=20) for s in population])

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
        indexes_to_mutate = np.random.choice(range(int(alpha * N/2), N), n_mutate, replace=False)

        for i in indexes_to_mutate:
            population[i] = mutate(population[i])

    print("Entire Population:")
    print(population)