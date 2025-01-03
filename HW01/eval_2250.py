import ast
import numpy as np
from utils import *
import time
import argparse

def calc_score(influencers):
    score = []
    NoseBook_network = create_graph(NoseBook_path)
    for j in range(200):
        purchased = set(influencers)
        for i in range(6):
            purchased = buy_products(NoseBook_network, purchased)

        s = product_exposure_score(NoseBook_network, purchased)
        score.append(s)

    s = np.mean(score)
    if s > 2300:
        return s
    return 0


if __name__ == '__main__':
    cnt = 0
    print("----------------Awake-----------------")
    seeds = []
    with open("simu_sols.txt", 'r') as file:
        for line in file.readlines()[cnt:]:
            seeds.append(ast.literal_eval(line))
    if len(seeds) > 0:
        for item in seeds:
            s = calc_score(item)
            if s > 2240:
                with open('sol_2240plus.txt', 'a') as file:
                    file.write(f"{(item, s)}\n")
            cnt += 1