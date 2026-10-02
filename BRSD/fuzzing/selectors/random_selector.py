import random


class RandomSelector:
    def __init__(self, seed=None):
        self.name = "Random"
        self.seed = seed

    def select(self, dataset, budget):
        if self.seed is not None:
            random.seed(self.seed)
        return random.sample(range(len(dataset)), budget)
