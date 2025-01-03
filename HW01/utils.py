import csv

import numpy as np
import pandas as pd

from Praducci_simulation import create_graph, NoseBook_path, cost_path, buy_products, product_exposure_score


def read_graph():
    graph = create_graph(NoseBook_path)

    return graph


def read_scores():
    with open("output.csv", "r") as f:
        csv_reader = csv.reader(f)

        scores = {int(row[0]): float(row[1]) for row in csv_reader}

        return scores


def read_distances():
    distances = dict()

    with open("distances.csv", "r") as f:
        csv_reader = csv.reader(f)

        for row in csv_reader:
            if len(row) != 0:
                distances[frozenset([int(row[0]), int(row[1])])] = int(row[2])

    return distances


def get_cost(node):
    costs = pd.read_csv(cost_path)
    return costs[costs['user'] == node]['cost'].item()


def calc_score(graph, s, num_trials=5, rounds=6):
    scores_l = []

    for _ in range(num_trials):
        purchased = s
        for i in range(rounds):
            purchased = buy_products(graph, purchased)

        scores_l.append(product_exposure_score(graph, purchased))

    return np.mean(scores_l)


def save_nodes_dict(data_dict, file_name):
    with open(file_name, "w", newline='') as f:
        csv_writer = csv.writer(f)

        for node, value in data_dict.items():
            csv_writer.writerow([node, value])


def loads_nodes_dict(file_name):
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        return {int(row[0]) : float(row[1]) for row in reader}

