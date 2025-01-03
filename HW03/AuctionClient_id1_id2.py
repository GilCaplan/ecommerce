import copy

import numpy as np
from time import perf_counter

import scipy
import scipy.integrate as integrate
import scipy.special as special


class AuctionClient:
    def __init__(self, value, clients_num, insurances_num):
        """Initializes the AuctionClient class.

        Args:
            value (float): The value of the client for a year of insurance.
            clients_num (int): The number of clients in the simulation.
            manufacturers_num (int): The number of manufacturers in the simulation.
        """
        self.v = value
        self.clients_num = clients_num
        self.insurances_num = insurances_num

        self.last_duration = None
        self.last_price = float('inf')

        self.num_active_bidders = clients_num

        self.rv = scipy.stats.beta(2, 5)

        self.delta = 0.01
        self.cdf = self.rv.cdf(np.arange(0, 1, self.delta))

    def decide_bid(self, t, duration):
        """
        Decides the bid of the client at time t given the duration of the insurance.

        Args:
            t (int): The current time.
            duration (int): The duration of the insurance.

        Returns:
            float: The bid of the client
        """
        # if self.num_active_bidders == 0:
        #     left = 1
        # else:
        #     left = self.insurances_num - t
        # if duration <= (4 / 5):
        #     return 0
        #
        # return (duration - (4/5)) / duration * self.v
        n_rest = self.insurances_num - t - 1

        if n_rest > 0:
            # param = 6/7
            param = 3 * np.sum((1 - self.cdf ** n_rest) * self.delta)
        else:
            param = 0.5

        if duration <= param:
            return 0
        return(duration - param) / duration * self.v

    def update(self, t, price):
        """
        Updates the client based on the price
        of the winning bid at time t.

        Args:
            t (int): The current time.
            price (float): The price of the winning bid.
        """
        self.last_price = price
        if price > 0:
            self.num_active_bidders -= 1


def auction_client_creator(value, clients_num, insurances_num):
    return AuctionClient(value, clients_num, insurances_num)


class NaiveAuctionClient:
    def __init__(self, value, clients_num, insurances_num):
        """
        Initializes the NaiveAuctionClient class.
        It offers it actual value for the product.
        """
        self.value = value
        self.clients_num = clients_num
        self.manufacturers_num = insurances_num

    def decide_bid(self, t, quality):
        return self.value

    def update(self, t, price):
        pass


def naive_auction_client_creator(value, clients_num, insurances_num):
    return NaiveAuctionClient(value, clients_num, insurances_num)


def simulate_single_auction(num_of_competitors, number_of_insurances):
    """
    Simulates a single auction game.
    Args:
        your_client: your AuctionClient object
        num_of_competitors: number of competitors
        number_of_insurances: number of insurances

    Returns:

    """

    start = perf_counter()
    your_value = np.random.beta(5, 2)
    your_client = auction_client_creator(your_value, num_of_competitors + 1, number_of_insurances)
    end = perf_counter()
    if end - start > 2:
        raise Exception("The initialization of the client took too long.")

    competing_clients_list = [naive_auction_client_creator(np.random.beta(5, 2), num_of_competitors,
                                                           number_of_insurances) for _ in range(num_of_competitors)]

    active_competing_clients = copy.deepcopy(competing_clients_list)
    for t in range(number_of_insurances):

        duration = 3 * np.random.beta(2, 5)
        start = perf_counter()
        your_bid = your_client.decide_bid(t, duration)
        end = perf_counter()
        # if end - start > 0.5:
        #     raise Exception("The decision of the bid took too long.")

        competing_bid_list = [client.decide_bid(t, duration) for client in active_competing_clients]

        if len(active_competing_clients) == 0:
            if your_bid == 0:  # you didn't want to bid
                continue
            else:
                return duration * your_value

        if your_bid == 0:  # you didn't want to bid
            if len(active_competing_clients) == 1:
                second_highest_bid = 0
                active_competing_clients.pop(0)

            else:

                highest_bid = max(competing_bid_list)
                clients_with_highest_bid = [i for i, bid in enumerate(competing_bid_list) if bid == highest_bid]
                winner = np.random.choice(clients_with_highest_bid)
                second_highest_bid = max([bid for i, bid in enumerate(competing_bid_list) if i != winner])
                active_competing_clients.pop(winner)

        else:
            # Check if you are the winner of the auction. Determine tie-breaks randomly.
            highest_bid = max(your_bid, max(competing_bid_list))
            clients_with_highest_bid = [-1] if your_bid == highest_bid else []
            clients_with_highest_bid += [i for i, bid in enumerate(competing_bid_list) if bid == highest_bid]
            winner = np.random.choice(clients_with_highest_bid)
            second_highest_bid = your_bid if winner != -1 else 0
            less_then_maximum_scores = [bid for i, bid in enumerate(competing_bid_list) if i != winner]
            if len(less_then_maximum_scores) > 0:
                second_highest_bid = max(second_highest_bid, max(less_then_maximum_scores))

            # if you are the winner you get your utility and the simulation ends.
            # otherwise, the winner is removed from the list of active clients and the simulation continues.
            if winner == -1:
                return duration * (your_value - second_highest_bid)

            else:
                active_competing_clients.pop(winner)

        start = perf_counter()
        your_client.update(t, second_highest_bid)
        end = perf_counter()
        if end - start > 0.5:
            raise Exception("The update of the client took too long.")
        for client in active_competing_clients:
            client.update(t, second_highest_bid)

    return 0.5 * your_value  # if the simulation ends and you didn't win, you get default utility.


def simulate(simulations, num_of_competitors, number_of_insurances):
    """
    Simulates the auction game for the given number of rounds.

    Args:
        simulations (int): The number of simulations to run.
        num_of_competitors (int): The number of competitors in the simulation.
        number_of_insurances (int): The number of insurances in the simulation.

    Returns:
        float: The revenue of your client.
    """

    revenues_list = []
    for _ in range(simulations):
        revenue = simulate_single_auction(num_of_competitors, number_of_insurances)
        revenues_list.append(revenue)

    return np.mean(revenues_list)


VARIABLES_VALUES = [(10, 5), (5, 10), (10, 10), (20, 10), (10, 20), (20, 20)]
BASELINES = [0.3565, 0.756, 0.3565, 0.3565, 0.856, 0.3565]

if __name__ == "__main__":
    np.random.seed(0)
    for i, (num_of_competitors, number_of_insurances) in enumerate(VARIABLES_VALUES):
        # if i not in [1, 4]:
        #     continue

        result = simulate(10000, num_of_competitors, number_of_insurances)
        if result < BASELINES[i]:
            # raise Exception("You didn't beat the baseline.")
            print(f'failed {i}: {result} < {BASELINES[i]}')
        else:
            print(f'passed {i}, {result} >= {BASELINES[i]}')

        # pass
    # print("All simulations passed.")
