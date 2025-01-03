import numpy as np
import networkx as nx
import random
import pandas as pd
import csv

NoseBook_path = 'NoseBook_friendships.csv'
cost_path = 'costs.csv'
costs = pd.read_csv(cost_path)
cost_dict = dict(zip(costs['user'], costs['cost']))


def normalize_dict_values(data_dict):
    values = np.array(list(data_dict.values()))
    # Normalize the values to the range [0, 100]
    normalized_values = 100 * (values - values.min()) / (values.max() - values.min())
    # Create a new dictionary with the normalized values
    normalized_dict = {k: v for k, v in zip(data_dict.keys(), normalized_values)}
    return normalized_dict


def influencers_submission(ID1, ID2, lst):
    with open(f'{ID1}_{ID2}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(lst)


def create_graph(edges_path: str) -> nx.Graph:
    """
    Creates the NoseBook social network
    :param edges_path: A csv file that contains information obout "friendships" in the network
    :return: The NoseBook social network
    """
    edges = pd.read_csv(edges_path).to_numpy()
    net = nx.Graph()
    net.add_edges_from(edges)
    return net


def buy_products(net: nx.Graph, purchased: set) -> set:
    """
    Gets the network at the beginning of stage t and simulates a purchase round
    :param net: The network at stage t
    :param purchased: All the users who recieved or bought the product up to and including stage t-1
    :return: All the users who recieved or bought the product up to and including stage t
    """
    new_purchases = set()
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))
        b = len(neighborhood.intersection(purchased))
        n = len(neighborhood)
        prob = b / n
        if prob >= random.uniform(0, 1):
            new_purchases.add(user)

    return new_purchases.union(purchased)


def product_exposure_score(net: nx.Graph, purchased_set: set) -> int:
    """
    Returns the number of users who have seen the product
    :param purchased_set: A set of users who bought the product
    :param net: The NoseBook social network
    :return:  The sum for all users who have seen the product.
    """
    exposure = 0
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))

        if user in purchased_set:
            exposure += 1
        elif len(neighborhood.intersection(purchased_set)) != 0:
            b = len(neighborhood.intersection(purchased_set))
            rand = random.uniform(0, 1)
            if rand < 1 / (1 + 10 * np.exp(-b/2)):
                exposure += 1
    return exposure


def get_influencers_cost(cost_path: str, influencers: list) -> int:
    """
    Returns the cost of the influencers you chose
    :param cost_path: A csv file containing the information about the costs
    :param influencers: A list of your influencers
    :return: Sum of costs
    """
    costs = pd.read_csv(cost_path)
    return sum([costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in influencers])


def run_simulation(influencers, G, alpha=10, beta=1.5):
    '''
    Runs a simulation of the expirement where we run it alpha times (Heuristic)
    :param influencers: list of nodes (seeds) that we're checking the simulation on
    :param G: Graph that we're checking
    :param alpha: magic number/hyper-parameter for iterations of experiment
    :param beta: magic number/hyper-parameter for iterations of experiment as seed set gets bigger
    :return score given a list of influencers (seeds):
    '''
    score = []
    for j in range(alpha + int(len(influencers) * beta)):
        purchased = set(influencers)

        for i in range(6):
            purchased = buy_products(G, purchased)

        score.append(product_exposure_score(NoseBook_network, purchased))
    return np.mean(score) / sum(cost_dict[node] for node in influencers)


def greedy_find_influencers(G, sim, cost=1000):
    '''
    Greedy Search algorithm to find influencers = argmax{\sigma(influencers) ^ cost(influencers) <= 1000}
    Step 2
    :param G: network of all Nodes
    :param sim: list of influencers to choose from
    :return: list of influencers that maximizes score value where cost is up to 1000
    '''
    print("-----------------------Searching for a greedy argmax solution-------------------------")
    c, rnd, seeds = 0, 0, []
    while c <= cost and rnd < 10:
        if c > 1000:
            seeds.pop()
        s_scores = {None: 0}
        cnt = 1
        for node in sim:
            mi = get_influencers_cost(cost_path, [seeds[len(seeds) - 1]]) if(len(seeds) > 0) else 0
            if node not in seeds and get_influencers_cost(cost_path, seeds + [node]) - mi <= 1000:
                prev = seeds.pop() if len(seeds) > rnd else None
                seeds.append(node)
                s_scores[node] = run_simulation(seeds, G)
                if node in s_scores.keys() and prev in s_scores.keys():
                    if prev is not None and s_scores[node] >= s_scores[prev]:
                        if prev in seeds:
                            seeds.remove(prev)
                    elif s_scores[prev] > s_scores[node]:
                        if node in seeds:
                            seeds.remove(node)
                        seeds.append(prev)

                c = get_influencers_cost(cost_path, seeds)
                if rnd < len(seeds) and cnt % 10 == 0:
                    print(cnt, get_influencers_cost(cost_path, seeds), s_scores[seeds[rnd]], seeds)
                cnt += 1
        rnd += 1
    return seeds


if __name__ == '__main__':

    print("STARTING")

    NoseBook_network = create_graph(NoseBook_path)
    deg_closeness = nx.degree_centrality(NoseBook_network)
    bet_closeness = nx.betweenness_centrality(NoseBook_network)
    # Step 1.1: Sort and rank nodes based on centrality measures
    sorted_bet_closeness = sorted(bet_closeness, key=bet_closeness.get, reverse=True)
    sorted_deg_closeness = sorted(deg_closeness, key=deg_closeness.get, reverse=True)

    # Step 1.2: Select top 600 nodes from each centrality measure
    top_600_bet = set(sorted_bet_closeness[:600])
    top_600_deg = set(sorted_deg_closeness[:600])

    # Step 1.3: Find union of top 600 nodes in both centrality measures
    common_nodes = top_600_bet.union(top_600_deg)
    nodes = [node for node in common_nodes if cost_dict[node] == 100]
    # Step 1 takes approximately ~3min to run
    # Step 2: Run greedy algorithm 5 times (can play around with the number of iterations)
    greedy_solutions = []
    greedy_iterations = 5
    for i in range(greedy_iterations):
        found = greedy_find_influencers(NoseBook_network, common_nodes)
        greedy_solutions.append(found)
        # print("found solution ", i+1, ":", found)

    # print(greedy_solutions)

    # pass over greedy_solutions to step 3
    # Step 3 - Genetic Algorithm
    influencers_submission('337604821','123',[3654, 2298, 1263, 640, 2295, 3448, 1969, 1051, 3851, 3040])
