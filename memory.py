import random
from sumtree import SumTree


class Memory:
    """
    Stores transitions as (s, a, r, s_, done) tuples using a SumTree.
    Each sample is assigned a priority which affects retrieval
    """

    def __init__(self, capacity, e=0.01, a=0.6):
        """        
        :param capacity: The maximum number of samples that can be stored
        :param e: Ensures that no sample has 0 priority
        :param a: 
        """
        self.capacity = capacity
        self.e = e
        self.a = a

        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        """
        Adds a new sample to the buffer
        :param error: The error associated with the sample
        :param sample: The sample to add 
        """
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        """
        Returns n samples from the buffer
        :param n: The number of samples to return
        """
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)
