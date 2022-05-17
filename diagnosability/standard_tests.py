from diagnosability.factors import TestFactor
from itertools import product
import numpy as np
from typing import List, Any
from nptyping import NDArray, Float
from diagnosability.perception_system import Test
from diagnosability.perception_system import FailureMode


class IdealTest(TestFactor):
    """
    Ideal test.
    An ideal test fails if and only if at least one failure is observed.
    """

    def __init__(self, name: str, scope: List[FailureMode]):
        assert isinstance(name, str)
        assert isinstance(scope, list) and all(
            [isinstance(f, FailureMode) for f in scope]
        )
        super().__init__(Test(name=name, scope=scope), self.compute_pmf(scope))

    @staticmethod
    def compute_pmf(scope: List[FailureMode]) -> NDArray:
        """
        Computes the probability mass function of an ideal test.
        """
        cardinality = len(scope)
        pmf = []
        for x in product(*[range(2)] * (cardinality + 1)):
            num_faults = sum(x) - x[0]
            if x[0] == 1:
                # FAIL
                v = float(num_faults > 0)
            else:
                # PASS
                v = float(num_faults == 0)
            pmf.append(v)
        return np.array(pmf)


class PerfectTest(TestFactor):
    def __init__(self, name: str, scope: List[FailureMode]) -> None:
        """Perfect tests from tests scopes.

        Args:
            scopes (Dict): dictionary of the tests scopes
        """
        assert isinstance(name, str)
        assert isinstance(scope, list) and all(
            [isinstance(f, FailureMode) for f in scope]
        )
        super().__init__(Test(name=name, scope=scope), self.compute_pmf(scope))

    @staticmethod
    def compute_pmf(scope: List[FailureMode]) -> NDArray:
        cardinality = len(scope)
        pmf = []
        for x in product(*[range(2)] * (cardinality + 1)):
            num_faults = sum(x) - x[0]
            if cardinality > 1:
                if x[0] == 0:
                    # PASS
                    v = 1.0 if num_faults == 0 or num_faults == cardinality else 0.0
                else:
                    # FAIL
                    v = 1.0 if num_faults > 0 else 0.0
            else:
                if x[0] == 0:
                    # PASS
                    v = 1.0 if num_faults == 0 else 0.0
                else:
                    # FAIL
                    v = 1.0 if num_faults > 0 else 0.0
            pmf.append(v)
        return np.array(pmf)


class NoisyOrTest(TestFactor):
    def __init__(
        self, name: str, scope: List[FailureMode], pd: float = 0.99, pf: float = 0.1
    ):
        assert isinstance(name, str)
        assert isinstance(scope, list) and all(
            [isinstance(f, FailureMode) for f in scope]
        )
        super().__init__(Test(name=name, scope=scope), self.compute_pmf(scope, pd, pf))

    @staticmethod
    def compute_pmf(scope: List[FailureMode], pd: float, pf: float) -> NDArray:
        nor = lambda z, v: z + (-1) ** z * np.prod(
            [((1 - pd) ** x) * ((1 - pf) ** (1 - x)) for x in v]
        )
        return np.array(
            [nor(x[0], x[1:]) for x in product([0, 1], repeat=len(scope) + 1)]
        )


class WellBehavedTest(TestFactor):
    def __init__(
        self,
        name: str,
        scope: List[FailureMode],
        Pd: NDArray[(Any,), Float],
        Pf: NDArray[(Any,), Float],
    ) -> None:
        """Create a well behaved test from detection and false alarm probability vectors.

        Args:
            failures (List[str]): The list of failure modes
            Pd (NDArray): A vector of size |F| with detection probabilities
            Pf (NDArray): A vectors of size |F| with false alarm probabilities
        """
        assert isinstance(name, str)
        assert isinstance(scope, list) and all(
            [isinstance(f, FailureMode) for f in scope]
        )
        assert Pd.shape == (len(scope),), "Pf and Pd must be the same size as scope"
        assert Pd.shape == Pf.shape, "Incompatible size"
        self.Pd = Pd
        self.Pf = Pf
        super().__init__(Test(name=name, scope=scope), self.compute_pmf(Pd, Pf, scope))

    @staticmethod
    def compute_pmf(pd, pf, scope):
        cardinality = len(scope)
        pmf = []
        for x in product(*[range(2)] * (cardinality + 1)):
            if x[0] == 0:
                # PASS
                factors = [
                    ((1 - pd[i]) ** x[i + 1]) * ((1 - pf[i]) ** (1 - x[i + 1]))
                    for i in range(cardinality)
                ]
            else:
                # FAIL
                factors = [
                    (pf[i] ** (1 - x[i + 1])) * (pd[i] ** x[i + 1])
                    for i in range(cardinality)
                ]
            pmf.append(np.prod(factors))
        return np.array(pmf)


class RandomTest(TestFactor):
    def __init__(self, name: str, scope: List[FailureMode]) -> None:
        assert isinstance(name, str)
        assert isinstance(scope, list) and all(
            [isinstance(f, FailureMode) for f in scope]
        )
        super().__init__(
            Test(name=name, scope=scope), np.random.rand(2 ** (len(scope) + 1))
        )
