
import ast
import numpy as np
# Path to the file
import networkx as nx
from utils import *
import time
costs = pd.read_csv(cost_path)
cost_dict = dict(zip(costs['user'], costs['cost']))
def get_influencers_cost(cost_path: str, influencers: list) -> int:
    """
    Returns the cost of the influencers you chose
    :param cost_path: A csv file containing the information about the costs
    :param influencers: A list of your influencers
    :return: Sum of costs
    """
    costs = pd.read_csv(cost_path)
    return sum([costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in influencers])


def normalize_dict_values(data_dict):
    values = np.array(list(data_dict.values()))
    # Normalize the values to the range [0, 100]
    normalized_values = 100 * (values - values.min()) / (values.max() - values.min())
    # Create a new dictionary with the normalized values
    normalized_dict = {k: v for k, v in zip(data_dict.keys(), normalized_values)}
    return normalized_dict

NoseBook_network = create_graph(NoseBook_path)
deg_closeness = nx.degree_centrality(NoseBook_network)
bet_closeness = nx.betweenness_centrality(NoseBook_network)
content = [deg_closeness, bet_closeness]
content = [normalize_dict_values(data_dict) for data_dict in content]
common_keys = set(content[0]).intersection(*content[1:])

# Step 1: Sort and rank nodes based on centrality measures
sorted_bet_closeness = sorted(bet_closeness, key=bet_closeness.get, reverse=True)
sorted_deg_closeness = sorted(deg_closeness, key=deg_closeness.get, reverse=True)

# Step 2: Select top 600 nodes from each centrality measure
top_600_bet = set(sorted_bet_closeness[:600])
top_600_deg = set(sorted_deg_closeness[:600])

# Step 3: Find union of top 600 nodes in both centrality measures
common_nodes = top_600_bet.union(top_600_deg)

