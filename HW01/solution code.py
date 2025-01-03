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


def normalize_dict_values(data_dict):
    values = np.array(list(data_dict.values()))
    # Normalize the values to the range [0, 100]
    normalized_values = 100 * (values - values.min()) / (values.max() - values.min())
    # Create a new dictionary with the normalized values
    normalized_dict = {k: v for k, v in zip(data_dict.keys(), normalized_values)}
    return normalized_dict

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


def run_simulation(influencers, G, score_ls = None, normalize=True, num_iter=None, alpha=20, beta=1.3):
    '''
    Runs a simulation of the expirement where we run it alpha times (Heuristic)
    :param influencers: list of nodes (seeds) that we're checking the simulation on
    :param G: Graph that we're checking
    :param alpha: magic number/hyper-parameter for iterations of experiment
    :param beta: magic number/hyper-parameter for iterations of experiment as seed set gets bigger
    :return score given a list of influencers (seeds):
    '''

    n = num_iter if num_iter != None else alpha + int(len(influencers) * beta)
        
    score = []
    if normalize == None and sum(score_ls(s) for s in influencers)/sum(cost_dict[node] for node in influencers) < 0.65:
        return 0
    for j in range(n):
        purchased = set(influencers)

        for i in range(6):
            purchased = buy_products(G, purchased)

        score.append(product_exposure_score(NoseBook_network, purchased))

    if normalize: 
        return np.mean(score) / sum(cost_dict[node] for node in influencers)
    else:
        return np.mean(score)


def greedy_find_influencers(G, sim, score_ls, cost=1000):
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
        #score dictionary to keep track of scores that we calculate
        s_scores = {None: 0}
        cnt = 1
        for node in sim:
            mi = get_influencers_cost(cost_path, [seeds[len(seeds) - 1]]) if(len(seeds) > 0) else 0
            if node not in seeds and get_influencers_cost(cost_path, seeds + [node]) - mi <= 1000:
                # checking cost of current seed set to make sure we"re in the cost range
                prev = seeds.pop() if len(seeds) > rnd else None
                # remove the last seed in the seed node to test the new node
                seeds.append(node)
                # calculating score given run_simulation heuristic
                s_scores[node] = run_simulation(seeds, G, score_ls)
                if node in s_scores.keys() and prev in s_scores.keys():
                    # if seed score with new node is higher than previous seed set then update current seed set
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


def genetic_mutate(current_list, nodes):
    '''
    The mutate step for the genetic algorithm. 
    current list: a solution of nodes that cost 100
    nodes: list of nodes that cost 100

    It returns a new list that is the same as the current_list, but one
    of the indexs was changed to a random node from the nodes list
    '''
    mutated_list = list(current_list)

    i = np.random.randint(len(current_list))

    mutated_list[i] = np.random.choice(nodes)

    return mutated_list


def genetic_crossover(list_1, list_2):
    '''
    The crossover step of the genetic algorithm
    list_1 and list_2 are list of solutions that cost 100 (10 nodes)

    Return a list that contains 10 nodes randomly selected from the union of list_1 and list_2
    '''
    combined = list(set(list_1).union(set(list_2)))

    return list(np.random.permutation(combined)[:10])


def genetic_algorithm(nodes, graph, survired_ratio=0.5, mutation_ratio=0.25, N=100, n_iters=300):
    '''
    Genetic Algorithm
    Creates population of size N.
    In each iteration (there are  n_iters iterations), select the survired_ratio best solutions.
    Keep them , and combine them together to form new solutions.
    Mutate the solutiond.

    Returns the last population
    '''
    print(f"N = {N}")
    print(f"n_iter = {n_iters}")
    print(f"alpha = {survired_ratio}")
    print(f"beta = {mutation_ratio}")

    # initial population
    population = [list(np.random.choice(nodes, size=10, replace=False)) for _ in range(N)]

    # in each iteration
    for k in range(n_iters):
        # calculate scores
        pop_scores = [run_simulation(s, graph, normalize=False, num_iter=20) for s in population]

        n_survived = int(survired_ratio * N)
        n_died = N - n_survived

        sorted_indexes = sorted(range(N), key=lambda node_i: pop_scores[node_i])
        # partition_ind = np.argpartition(pop_scores, n_died)

        # witch solutions 'survive' -> continue to the next iteration
        # and which 'die' -> not continue
        # the best survired_ratio solutions are the ones that survive
        survived_indexes = sorted_indexes[n_died:]
        died_indexes = sorted_indexes[:n_died]

        if k != 0:
            print(f"Round {k} result")
            print(f"  min: {np.min(pop_scores)}")
            print(f"  max: {np.max(pop_scores)}")
            print(f"  median: {np.percentile(pop_scores, 0.5)}")
            best_ind = np.argmax(pop_scores)
            best = population[best_ind]
            best_score = pop_scores[best_ind]
            print(f"  {list(best)} -> {best_score}")

        # replaced the bad solutions with new ones crated from the good solutions
        for ind in died_indexes:
            parent_indexes = np.random.choice(survived_indexes, size=2, replace=False)

            parent1 = population[parent_indexes[0]]
            parent2 = population[parent_indexes[1]]

            child = genetic_crossover(parent1, parent2)

            population[ind] = child

        # select solutions to mutate randomly
        n_mutate = int(mutation_ratio * N)
        indexes_to_mutate = np.random.choice(N, n_mutate, replace=False)

        for i in indexes_to_mutate:
            population[i] = genetic_mutate(population[i], nodes)

    return population
     

if __name__ == '__main__':
    print("STARTING")
    influencers_submission('337604821','213203573',[3654, 2298, 2670, 640, 2295, 3448, 1969, 1051, 3851, 3040])

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

    # remove node that costs more than 100 (MUST DO TO RUN GENETIC ALGORITHM)
    common_nodes = list(set(filter(lambda node: cost_dict[node] == 100, common_nodes)))

    content = [normalize_dict_values(data_dict) for data_dict in [deg_closeness, bet_closeness]]
    common_keys = set(content[0]).union(*content[1:])
    merged_dict = {key: sum(d[key] for d in content) for key in common_keys}

    # Step 1 takes approximately ~3min to run
    # Step 2: Run greedy algorithm 5 times (can play around with the number of iterations)
    greedy_solutions = []
    greedy_iterations = 3
    for i in range(greedy_iterations):
        found = greedy_find_influencers(NoseBook_network, common_nodes, merged_dict)
        greedy_solutions.append(found)
        print("found solution ", i+1, ":", found)

    # unify all the solutions
    greedy_nodes = []
    for solution in greedy_solutions:
        greedy_nodes.extend(solution)

    # pass over greedy_solutions to step 3

    # Step 3 - Genetic Algorithm
    # remove duplicates
    greedy_nodes = set(greedy_nodes)
    random_nodes = set(random.sample(common_nodes, 20))
    pop_nodes = list(greedy_nodes.union(random_nodes))
    population = genetic_algorithm(pop_nodes, NoseBook_network, survired_ratio=0.5, mutation_ratio=0.25, N=20, n_iters=50)

    # Step 4 - Select Best Seed (influencers) set
    # Top 10
    best_solutions_10 = sorted(population, reverse=True, key=lambda s: run_simulation(s, NoseBook_network, normalize=False, num_iter=20))[:10]

    # Top 3
    best_solutions_3 = sorted(best_solutions_10, reverse=True, key=lambda s: run_simulation(s, NoseBook_network, normalize=False, num_iter=100))[:3]
    print(best_solutions_3)
    best = sorted(best_solutions_3, reverse=True, key=lambda s: run_simulation(s, NoseBook_network, normalize=False, num_iter=1000))[0]

    influencers_submission('337604821','213203573', best)

    # WE RAN OUR CODE for LONGER SO OUR BEST RESULT WAS:
    # influencers_submission('337604821','213203573',[3654, 2298, 2670, 640, 2295, 3448, 1969, 1051, 3851, 3040])
