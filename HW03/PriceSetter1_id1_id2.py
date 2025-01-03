import numpy as np
from time import perf_counter

check = []
class PriceSetter1:

    def __init__(self, rounds):
        """
        Initialize the price setter.
        In this settings, the values of the costumers is constant and unknown in advance.

        Args:
            rounds (int): the number of rounds to simulate
        """
        self.maxPrice = 0.5
        self.minPrice = 0
        self.threshold = 1
        self.flag = False

    def set_price(self, t):
        """
        Return the price at time t.

        Args:
            t (int): the time period

        Returns:
            float: the price at time t
        """
        return self.maxPrice

    def update(self, t, outcome):
        """
        Update the price setter based on the outcome of the previous period.

        Args:
            t (int): the time period
            outcome (int): the outcome of the previous period - true if the product was sold, false otherwise
        """
        scaling_factor = 1.4 - (0.4 * (t / (t + 1)))
        if outcome:
            self.minPrice = self.maxPrice
            self.maxPrice = min(self.threshold, scaling_factor * self.maxPrice)
        else:
            self.threshold = self.maxPrice
            self.maxPrice = (self.minPrice + self.maxPrice) / 2

        if self.threshold - self.minPrice < 0.001:
            if not self.flag:
                check.append(t)
                self.flag = True
            self.maxPrice = self.minPrice
def simulate(simulations, rounds):
    """
    Simulate the game for the given number of rounds.

    Args:
        rounds (int): the number of rounds to simulate

    Returns:
        float: the revenue of the price setter
    """
    simulations_results = []
    for _ in range(simulations):
        start = perf_counter()
        price_setter = PriceSetter1(rounds)
        end = perf_counter()
        if end - start > 1:
            raise Exception("The initialization of the price setter is too slow.")
        revenue = 0
        costumer_value = np.random.uniform(0, 1)

        for t in range(rounds):
            start = perf_counter()
            price = price_setter.set_price(t)
            end = perf_counter()
            if end - start > 0.1:
                raise Exception("The set_price method is too slow.")
            if costumer_value >= price:
                revenue += price

            start = perf_counter()
            price_setter.update(t, costumer_value >= price)
            end = perf_counter()
            if end - start > 0.1:
                raise Exception("The update method is too slow.")

        simulations_results.append(revenue)

    return np.mean(simulations_results)


if __name__ == "__main__":
    np.random.seed(0)
    print(simulate(10000, 1000))
    print(np.mean(check))
