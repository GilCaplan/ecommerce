import ast
import numpy as np
from utils import *
import time
import argparse

def calc_score(influencers):
    score = []
    NoseBook_network = create_graph(NoseBook_path)
    for j in range(300):
        purchased = set(influencers)
        for i in range(6):
            purchased = buy_products(NoseBook_network, purchased)

        s = product_exposure_score(NoseBook_network, purchased)
        score.append(s)

    s = np.mean(score)
    if s > 2320:
        return s
    return 0


if __name__ == '__main__':
    cnt = 0
    print("----------------Awake-----------------")
    while True:
        seeds = []
        with open("simu_sols.txt", 'r') as file:
            for line in file.readlines()[cnt:]:
                seeds.append(ast.literal_eval(line))
        if len(seeds) > 0:
            for item in seeds:
                s = calc_score(item)
                if s > 0:
                    with open('final_sol.txt', 'a') as file:
                        file.write(f"{(item, s)}\n")
                cnt += 1
        else:
            print(cnt)
            print("Kind Sir it's ", time.strftime("%H:%M:%S"))
            print("----------------Taking a nap for 30 min-----------------")
            time.sleep(30 * 60)
            print("----------------Awake-----------------")