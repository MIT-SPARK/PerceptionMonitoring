import itertools
from typing import Callable

from dataclasses import dataclass
from typing import Callable, List


class Hyperdiagnosable:
    def __init__(
        self, nr_variables: int, tests: List[Callable], constraint: List[Callable] = []
    ) -> None:
        self.constraint = constraint
        self.tests = tests
        self.nr_variables = nr_variables

    def syndromes(self, state):
        z = [t(state) for t in self.tests]
        return set(itertools.product(*z))

    def _is_valid(self, state):
        if not self.constraint:
            return True
        return all(c(state) for c in self.constraint)

    def _print(self, msg, enabled):
        if enabled:
            print(msg)

    def states(self):
        for k in range(self.nr_variables+1):
            for f_ in itertools.combinations(range(self.nr_variables), k):
                faults = set(f_)
                state = [int(i in faults) for i in range(self.nr_variables)]
                if not self._is_valid(state):
                    continue
                yield state

    def fault_identification(self, syndrome):
        possible_states = []
        best_size = self.nr_variables + 1  # equivalent to inf
        for state in self.states():
            if syndrome in self.syndromes(state):
                card = sum(state)
                if card < best_size:
                    best_size = card
                    possible_states = [state]
                elif card == best_size:
                    possible_states.append(state)
        return possible_states

    def kappa(self, verbose=False):
        kappa = 0
        used_syndromes = set()
        for k in range(self.nr_variables+1):
            nr_valid_states = 0
            self._print(f"Testing kappa = {k}", verbose)
            for z_ in itertools.combinations(range(self.nr_variables), k):
                z = set(z_)
                state = [int(i in z) for i in range(self.nr_variables)]
                if not self._is_valid(state):
                    continue
                nr_valid_states += 1
                self._print(f"State: {state}", verbose)
                syn = self.syndromes(state)
                self._print(f"Syndromes: {syn}", verbose)
                if any(s in used_syndromes for s in syn):
                    return kappa
                used_syndromes |= syn
            if nr_valid_states > 0:
                kappa = k
            self._print("#" * 20, verbose)
        return kappa

    @staticmethod
    def OR(scope: List[int], x) -> List[int]:
        assert scope
        return [int(sum(x[i] for i in scope) > 0)]

    @staticmethod
    def WeakOR(scope: List[int], x) -> List[int]:
        assert scope
        s = sum(x[i] for i in scope)
        if s == 0:
            return [0]
        elif s == len(scope):
            return [0, 1]
        else:
            return [1]

    @staticmethod
    def UnreliableOR(scope: List[int], x) -> List[int]:
        # PASS is unreliable, FAIL grants at least one active failure mode
        assert scope
        s = sum(x[i] for i in scope)
        if s == 0:
            return [0]
        else:
            return [0, 1]

    @staticmethod
    def AnyIfAny(a: List[int], b: List[int], x: List[int]) -> List[int]:
        """Represent (||a||_1 > 0) <=> (||b||_1 > 0) or all 0s"""
        assert bool(a) and bool(b)
        sa = sum(x[i] for i in a)
        sb = sum(x[i] for i in b)
        if sa == 0 and sb == 0:
            return True
        elif sa > 0 and sb > 0:
            return True
        else:
            return False

    @staticmethod
    def AllIfAll(a: List[int], b: List[int], x: List[int]) -> List[int]:
        """Represent (||a||_1 > 0) <=> (||b||_1 > 0) or all 0s"""
        assert bool(a) and bool(b)
        sa = sum(x[i] for i in a) == len(a)
        sb = sum(x[i] for i in b) == len(b)
        if sa and sb:
            return True
        else:
            return False

    @staticmethod
    def AllEqual(a: List[int], x) -> List[int]:
        assert len(a) > 1
        return (len(a) - 1) * x[a[0]] == sum(x[i] for i in a[1:])

    @staticmethod
    def OnlyOne(a: List[int], x) -> List[int]:
        # Mutual exclusion or all 0s
        assert a
        return sum(x[i] for i in a) <= 1


if __name__ == "__main__":
    from functools import partial

    # Simple Object detection
    # constraints = [
    #     partial(Hyperdiagnosable.AllEqual, [0, 3]),
    #     partial(Hyperdiagnosable.AllEqual, [1, 4]),
    #     partial(Hyperdiagnosable.AllEqual, [2, 5]),
    # ]
    # tests = [
    #     partial(Hyperdiagnosable.WeakOR, [3, 4]),
    #     partial(Hyperdiagnosable.WeakOR, [3, 5]),
    # ]
    # h = Hyperdiagnosable(nr_variables=6, tests=tests, constraint=constraints)

    # LiDAR-based ego motion
    constraints = [
        partial(Hyperdiagnosable.AnyIfAny, [0], [1,2]),
        partial(Hyperdiagnosable.AnyIfAny, [3], [1,2]),
        partial(Hyperdiagnosable.AnyIfAny, [3], [4]),
    ]
    tests = [
        partial(Hyperdiagnosable.OR, [2]),
        partial(Hyperdiagnosable.OR, [3]),
        partial(Hyperdiagnosable.OR, [4]),
        partial(Hyperdiagnosable.OR, [1,4]),
    ]
    h = Hyperdiagnosable(nr_variables=5, tests=tests, constraint=constraints)
    # -----
    kappa = h.kappa(True)
    print(f"The dgraph is {kappa}-diagnosable")
