from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from enum import Enum
from random import randint
from typing import Dict, Iterable, List, Optional
from dataclasses import dataclass
import numpy as np
from .utils import to_bin


class TestOutcome(Enum):
    PASS = 0
    FAIL = 1

    @staticmethod
    def random(fail_prob: float = 0.5):
        return TestOutcome(1 if np.random.rand() < fail_prob else 0)


class FailureModeState(Enum):
    INACTIVE = 0
    ACTIVE = 1

    @staticmethod
    def random():
        return FailureModeState(randint(0, 1))


class Syndrome(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)
        for k, v in self.items():
            if not isinstance(v, TestOutcome):
                self[k] = TestOutcome(v)

    def __setitem__(self, key, value):
        if not isinstance(value, TestOutcome):
            self.__dict__[key] = TestOutcome(value)
        else:
            self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        """returns simple dict representation of the mapping"""
        return str(self.__dict__)

    def __repr__(self):
        """echoes class, id, & reproducible representation in the REPL"""
        return "{}, Syndrome({})".format(
            super(Syndrome, self).__repr__(), self.__dict__
        )

    def __hash__(self) -> int:
        return hash(tuple((k, v.value) for k, v in sorted(self.items())))

    def num_tests(self):
        return len(self.__dict__)

    def pretty_print(self):
        for k, v in self.__dict__.items():
            print(f"{k} <- {'FAIL (1)' if v == TestOutcome.FAIL else 'PASS (0)'}")

    def failed_tests(self):
        return set([k for k, v in self.__dict__.items() if v == TestOutcome.FAIL])

    def to_dict(self):
        return {k: v.value for k, v in self.__dict__.items()}

    def encode(self, ordering=None):
        """Return a binary encoding of the syndrome"""
        if ordering is None:
            ordering = list(self.__dict__.keys())
        bin_str = [self.__dict__[t].value for t in ordering]
        return to_bin(bin_str[::-1])

    def __eq__(self, other):
        return self.failed_tests() == other.failed_tests()

    @staticmethod
    def random(tests: Iterable[str], fail_prob: float = 0.5) -> "Syndrome":
        return Syndrome({t: TestOutcome.random(fail_prob) for t in tests})


class ProbabilisticSyndrome:
    """Wrapper uset do tag a syndrome with a probability"""

    def __init__(self, syndrome: Syndrome, probability: float) -> None:
        assert probability >= 0.0 and probability <= 1.0
        self.syndrome = syndrome
        self._probability = probability

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, probability):
        assert probability >= 0.0 and probability <= 1.0
        self._probability = probability

    def pretty_print(self):
        self.syndrome.pretty_print()
        print(f"Probability: {self._probability}")

    def __str__(self) -> str:
        """returns simple dict representation of the mapping"""
        return str((self.syndrome.__dict__, self._probability))

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        return (self.syndrome.failed_tests() == other.syndrome.failed_tests()) and (
            self._probability == other.probability
        )


class FailureStates(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)
        for k, v in self.items():
            if not isinstance(v, FailureModeState):
                self[k] = FailureModeState(v)

    def num_failure_modes(self) -> int:
        return len(self.__dict__)

    def __setitem__(self, key, value):
        if not isinstance(value, FailureModeState):
            self.__dict__[key] = FailureModeState(value)
        else:
            self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        """returns simple dict representation of the mapping"""
        return str(self.__dict__)

    def __repr__(self):
        """echoes class, id, & reproducible representation in the REPL"""
        return "{}, FailureStates({})".format(
            super(FailureStates, self).__repr__(), self.__dict__
        )

    def pretty_print(self):
        for k, v in self.__dict__.items():
            print(
                f"{k} <- {'ACTIVE (1)' if v == FailureModeState.ACTIVE else 'INACTIVE (0)'}"
            )

    def active_failures(self):
        return set(
            [k for k, v in self.__dict__.items() if v == FailureModeState.ACTIVE]
        )

    def to_dict(self):
        return {k: v.value for k, v in self.__dict__.items()}

    def encode(self, ordering=None):
        if ordering is None:
            ordering = list(self.__dict__.keys())
        bin_str = [self.__dict__[f].value for f in ordering]
        return to_bin(bin_str[::-1])

    def matching(self, other):
        assert isinstance(other, FailureStates)
        assert set(self.keys()) == set(other.keys())
        return sum([float(self[f] == other[f]) for f in self.keys()])

    def __eq__(self, other):
        assert isinstance(other, FailureStates)
        return self.active_failures() == other.active_failures()

    def subset(self, keys):
        """Takes a set of keys and returns a new FailureStates object with only those keys."""
        assert set(keys).issubset(set(self.keys()))
        return FailureStates({k: self[k] for k in keys})


@dataclass
class SystemState:
    timestamp: float
    syndrome: Syndrome  # test results
    features: Optional[Dict[str, np.ndarray]] = None  # features vector
    states: Optional[FailureStates] = None  # Failure Mode State (active/inactive)
    ground_truth: Optional[FailureStates] = None  # ground truth
    info: Optional[dict] = None  # additional info

    def __post_init__(self):
        # Validation
        assert isinstance(self.syndrome, Syndrome), "syndrome must be a Syndrome"
        syndrome_tests = set(self.syndrome.keys())
        if self.features is not None:
            assert isinstance(self.features, Dict), "features must be a dict"
            failure_modes_features = set(self.features.keys())
            assert (
                syndrome_tests - failure_modes_features
            ), "syndrome and features must be defined on different modules"
        if self.states is not None:
            assert isinstance(
                self.states, FailureStates
            ), "states must be a FailureStates"
            failure_modes_states = set(self.states.keys())
            assert (
                syndrome_tests - failure_modes_states
            ), "syndrome and states must be defined on the same modules"
        if self.ground_truth is not None:
            assert isinstance(
                self.ground_truth, FailureStates
            ), "ground_truth must be a FailureStates"
            failure_modes_ground_truth = set(self.ground_truth.keys())
            assert (
                syndrome_tests - failure_modes_ground_truth
            ), "syndrome and ground_truth must be defined on the same modules"
        if self.features is not None and self.states is not None:
            assert set(self.features.keys()) == set(
                self.states.keys()
            ), "features and states must be defined on the same modules"
        if self.features is not None and self.ground_truth is not None:
            assert set(self.features.keys()) == set(
                self.ground_truth.keys()
            ), "features and ground_truth must be defined on the same modules"
        if self.states is not None and self.ground_truth is not None:
            assert set(self.states.keys()) == set(
                self.ground_truth.keys()
            ), "states and ground_truth must be defined on the same modules"


@dataclass
class TemporalSystemState:
    states: List[SystemState]
    time: float


class DiagnosticModel(ABC):
    @abstractmethod
    def fault_identification(self, syndrome: SystemState) -> FailureStates:
        pass

    @property
    @abstractmethod
    def failure_modes(self):
        pass

    @property
    @abstractmethod
    def tests(self):
        pass
