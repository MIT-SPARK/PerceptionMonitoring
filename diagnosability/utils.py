import itertools
import math
import random
from typing import List, Tuple

import pandas as pd


def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def all_equal(iterable):
    "Returns True if all the elements are equal to each other"
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)


class Powerset:
    def __init__(self, iter, max_cardinality=None):
        self.n = len(iter)
        if max_cardinality is None:
            self.k = self.n
        else:
            self.k = max_cardinality
        self._len = self._compute_cardinality()
        it_chain = [itertools.combinations(iter, i) for i in range(0, self.k + 1)]
        self.it_chain = itertools.chain(*it_chain)

    def __iter__(self):
        return self

    def __next__(self):
        return set(next(self.it_chain))

    def random(self, sample_size=1):
        assert sample_size > 0, "Sample size must be positive"
        results = []
        iterator = iter(self)
        # Fill in the first samplesize elements:
        try:
            for _ in range(sample_size):
                results.append(iterator.__next__())
        except StopIteration:
            raise ValueError("Sample larger than population.")
        random.shuffle(results)  # Randomize their positions
        for i, v in enumerate(iterator, sample_size):
            r = random.randint(0, i)
            if r < sample_size:
                results[r] = v  # at a decreasing rate, replace random items
        if sample_size == 1:
            results = results[0]
        return results

    def _compute_cardinality(self):
        if self.n == self.k:
            return 2 ** self.n
        else:
            return sum([binom(self.n, i) for i in range(0, self.k + 1)])

    def __len__(self):
        return self._len

    def max_cardinality(self):
        return self.k


def to_bin(it):
    res = 0
    for el in it:
        res = (res << 1) | el
    return res


def split_by_fractions(dfs: List[pd.DataFrame], splits: Tuple, randomize=True):
    """Split dataset into random subset of specified fraction.

    Example:
        train, test, val = split_by_fractions([samples], (0.8, 0.1, 0.1))

    Args:
        df (pd.DataFrame): Dataset to split
        splits (list): Fractions to split dataset into.
        random_state (int, optional): [description]. Defaults to 42.

    Returns:
        Tuple[pd.Dataframe]: The splits
    """
    assert dfs, "No dataframes provided"
    assert sum(splits) == 1.0, "fractions sum is not 1.0 (fractions_sum={})".format(
        sum(splits)
    )
    assert all_equal([len(df) for df in dfs]), "All datasets must have same shape"
    N = len(dfs[0])
    indices = list(range(N))
    if randomize:
        random.shuffle(indices)
    curr_idx = 0
    remaining = N
    dff = []
    split_idxs = []
    for i in range(len(splits)):
        if splits[i] == 0:
            dff.append([[] for _ in dfs]) 
            continue
        frac = splits[i] / sum(splits[i:])
        n = int(frac * remaining)
        stop = curr_idx + n
        split_idxs.append(indices[curr_idx:stop])
        dff.append([d.iloc[indices[curr_idx:stop]] for d in dfs])
        curr_idx = stop
        remaining -= n
    return dff, split_idxs
