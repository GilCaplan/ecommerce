import numpy as np
import networkx as nx
import random
import pandas as pd
import csv

NoseBook_path = 'NoseBook_friendships.csv'
cost_path = 'costs.csv'
costs = pd.read_csv(cost_path)
cost_dict = dict(zip(costs['user'], costs['cost']))

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
            if rand < 1 / (1 + 10 * np.exp(-b / 2)):
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
    return sum(
        [costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in
         influencers])


if __name__ == '__main__':
    best_nums = [1051, 3654, 1263, 1019, 3820, 384, 640, 777, 2298, 3448, 2295, 2868, 1019, 1051, 640, 3851, 284, 3448, 2368, 1051, 3654, 777, 3851, 2670, 640, 2679, 284, 3448, 24, 848, 1051, 3851, 3654, 1019, 697, 3448, 2480, 640, 2298,848, 1051, 3851, 3654, 1019, 697, 3448, 2480, 640, 24]

    NoseBook_network = create_graph(NoseBook_path)

    #influencers = [848, 1051, 3851, 3654, 617, 1019, 697, 3448, 2480, 640]#score 2180
    #influencers = [1625, 1969, 367, 2207, 2278, 2154, 3040, 3384,3851,809]
    #influencers = [848, 1051, 3851, 3654, 1019, 697, 3448, 2480, 640, 24] # 2233
    #influencers = [1051, 3654, 777, 3851, 2670, 640, 2679, 284, 3448, 24] #2190
    #influencers = [2295, 3654, 2868, 1019, 1051, 640, 3851, 284, 3448, 2368]# 2160
    #influencers = [1051, 3654, 1263, 1019, 3820, 384, 640, 777, 2298, 3448]# 2164
    #influencers = [384, 640, 777, 3851, 24, 1051, 1263, 3448, 2298, 1019]
    # influencers = [1051, 3851, 3654, 1019, 697, 848, 3448, 2480, 640, 2298] # 2330  20 + 2*|seeds|
    # influencers = [3654, 2298, 1263, 640, 2295, 3448, 1969, 1051, 3851, 3040]#2350
    #influencers = [3654, 2298, 2670, 640, 2295, 3448, 1969, 1051, 3851, 3040]#2350
    influencers =  [3654, 2298, 1263, 640, 1453, 3448, 1969, 1051, 3851, 3040]#2339
    influencers = [1538, 640, 1263, 919, 2295, 3851, 1839, 1969, 367, 777]

    # 3851, 3654, 3448, 640, 2298
    #influencers = [3654, 2298, 1263, 640, 1453, 3448, 1969, 1051, 3851, 3040]
    # for influencers in [[3654, 2298, 2670, 640, 2295, 3448, 1969, 1051, 3851, 3040],[3654, 2298, 1263, 640, 2295, 3448, 1969, 1051, 3851, 3040]]:
    influencers_cost = get_influencers_cost(cost_path, influencers)
    print("Influencers cost: ", influencers_cost)
    if influencers_cost > 1000:
        print("*************** Influencers are too expensive! ***************")
        exit()
    score = []
    for j in range(10000):
        purchased = set(influencers)
        for i in range(6):
            purchased = buy_products(NoseBook_network, purchased)

        s = product_exposure_score(NoseBook_network, purchased)
        score.append(s)
        print("finished round", j, ' with score: ', s, ' average:', np.mean(score), ' dist/std:', np.std(score))

    print("*************** Your final score is " + str(influencers)+", "+ str(np.mean(score)), ' with max score: '+str(np.max(score)) + " ***************")
#found solution  1 : [3215, 1438, 79, 197, 3557, 132, 962, 3448, 815, 848] taking:  18371.84859776497 seconds
#found solution  1 : [3215), 1438), 79), 197), 3557), 132), 962), 3448), 815), 848)] taking:  18371.84859776497 seconds
