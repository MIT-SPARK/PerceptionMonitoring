from itertools import product
from numpy.core.fromnumeric import shape
from pgmpy.factors.discrete import DiscreteFactor
from typing import Callable, Dict, List, Tuple, Set, Any, Union, Optional
from nptyping import NDArray, Float
from diagnosability.perception_system import Test, Unit
import numpy as np

def AtLeastOne(a, b):
    na = len(a)
    nb = len(b)
    dens = list()
    for x in product([0, 1], repeat=na + nb):
        if sum(x) == 0 or (sum(x[:na]) >= 1 and sum(x[na:]) >= 1):
            dens.append(1.0)
        else:
            dens.append(0.0)
    return np.array(dens)


def Majority(a, b):
    na = len(a)
    nb = len(b)
    th = np.floor(nb / 2)
    dens = list()
    for x in product([0, 1], repeat=na + nb):
        if sum(x) == 0 or (sum(x[:na]) >= 1 and sum(x[na:]) > th):
            dens.append(1.0)
        else:
            dens.append(0.0)
    return np.array(dens)

class PriorFactor(DiscreteFactor):
    def __init__(self, unit: Unit, failure_probability: float = 0.5):

        super().__init__(
            [unit.varname],
            [2],
            np.array([1 - failure_probability, failure_probability]),
        )
        self._scope = unit

    def copy(self):
        return PriorFactor(self._scope, self.values[1])

    def product(self, phi1, inplace=True):
        me = DiscreteFactor(self.scope(), self.cardinality, self.values)
        other = DiscreteFactor(phi1.scope(), phi1.cardinality, phi1.values)
        return me.product(other, inplace=inplace)

    def __repr__(self):
        var_card = ", ".join(
            [f"{var}:{card}" for var, card in zip(self.variables, self.cardinality)]
        )
        return (
            f"<{type(self).__name__} representing phi({var_card}) at {hex(id(self))}>"
        )

    def __eq__(self, other, atol=1e-8):
        # if not (
        #     isinstance(self, PriorFactor)
        #     and isinstance(other, PriorFactor)
        # ):
        #     return False
        return super().__eq__(other, atol)

    def __hash__(self):
        return super().__hash__()

    def randomize(self):
        self.values = np.random.rand(*self.values.shape)


class TestFactor(DiscreteFactor):
    def __init__(
        self,
        test: Test,
        values: Optional[NDArray[(Any), Float]] = None,
    ):
        self.test = test
        if values is None:
            values = np.ones(2 ** (len(test.scope) + 1))
        elif values.size == 2 ** len(test.scope):
            variables = [unit.varname for unit in test.scope]
            cardinality = [2] * len(test.scope)
        elif values.size == 2 ** (len(test.scope) + 1):
            variables = [self.test.varname] + [unit.varname for unit in test.scope]
            cardinality = [2] * (len(test.scope) + 1)
        else:
            raise RuntimeError("The factor values have the wrong size.")
        super().__init__(variables, cardinality, values)
    
    @property
    def varname(self):
        return self.test.varname

    def product(self, phi1, inplace=True):
        me = DiscreteFactor(self.scope(), self.cardinality, self.values)
        other = DiscreteFactor(phi1.scope(), phi1.cardinality, phi1.values)
        return me.product(other, inplace=inplace)

    def copy(self):
        return TestFactor(
            self.test,
            values=np.copy(self.values).flatten(),
        )

    def __repr__(self):
        var_card = ", ".join(
            [f"{var}:{card}" for var, card in zip(self.variables, self.cardinality)]
        )
        return (
            f"<{type(self).__name__} representing phi({var_card}) at {hex(id(self))}>"
        )

    def __eq__(self, other, atol=1e-8):
        # if not (isinstance(self, TestFactor) and isinstance(other, TestFactor)):
        #     return False
        return super().__eq__(other, atol)

    def __hash__(self):
        return super().__hash__()

    def randomize(self):
        self.values = np.random.rand(*self.values.shape)


class ConstraintFactor(DiscreteFactor):
    def __init__(
        self,
        name,
        scope: List[Unit],
        values: Optional[Union[NDArray[(Any), Float], Callable]] = None,
    ):
        variables = [unit.varname for unit in scope]
        cardinality = [2] * len(scope)
        table_size = 2 ** len(scope)
        self.name = name
        if values is None:
            values = np.ones(table_size)
        else:
            if callable(values):
                values = self._compute_pmf(scope, values)
            else:
                assert values.shape == (
                    table_size,
                ), "The factor values have the wrong size."
        super().__init__(variables, cardinality, values)
        self._scope = scope

    @staticmethod
    def _compute_pmf(scope, generator_fcn):
        return np.array(
            [float(generator_fcn(x)) for x in product([0, 1], repeat=len(scope))]
        )

    def copy(self):
        values = np.copy(self.values).flatten()
        return ConstraintFactor(name=self.name, scope=self._scope, values=values)

    def product(self, phi1, inplace=True):
        me = DiscreteFactor(self.scope(), self.cardinality, self.values)
        other = DiscreteFactor(phi1.scope(), phi1.cardinality, phi1.values)
        return me.product(other, inplace=inplace)

    def __repr__(self):
        var_card = ", ".join(
            [f"{var}:{card}" for var, card in zip(self.variables, self.cardinality)]
        )
        return (
            f"<{type(self).__name__} representing phi({var_card}) at {hex(id(self))}>"
        )

    def __eq__(self, other, atol=1e-8):
        # if not (isinstance(self, TestFactor) and isinstance(other, TestFactor)):
        #     return False
        return super().__eq__(other, atol)

    def __hash__(self):
        return super().__hash__()

    def randomize(self):
        self.values = np.random.rand(*self.values.shape)


class TemporalContraintFactor(ConstraintFactor):
    pass


class MaxCardinalityFactor(DiscreteFactor):
    def __init__(self, scope: List[Unit], max_cardinality: int):
        variables = [unit.varname for unit in scope]
        cardinality = [2] * len(scope)
        values = self._compute_pmf(scope, max_cardinality)
        super().__init__(variables, cardinality, values)
        self._scope = scope
        self._max_cardinality = max_cardinality

    @property
    def max_cardinality(self):
        return self._max_cardinality

    @staticmethod
    def _compute_pmf(scope, max_cardinality):
        return np.array(
            [
                float(sum(x) <= max_cardinality)
                for x in product([0, 1], repeat=len(scope))
            ]
        )
