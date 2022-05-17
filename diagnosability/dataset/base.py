import abc
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict, List, Optional, Union
from dataclasses_json import dataclass_json


class TestOutcome:
    PASS = False
    FAIL = True

    @staticmethod
    def from_bool(failure_is_active):
        return TestOutcome.FAIL if failure_is_active else TestOutcome.PASS

    @staticmethod
    def to_string(z):
        return "FAIL" if z == TestOutcome.FAIL else "PASS"

    @staticmethod
    def from_string(z):
        return TestOutcome.FAIL if z == "FAIL" else TestOutcome.PASS


@dataclass_json
@dataclass(frozen=True)
class TestResult:
    name: str
    scope: Union[List[List[str]], List[str]] = field(default_factory=list)
    result: bool = False
    confidence: float = 1.0
    timestep:Optional[int] = None

    def __repr__(self):
        return f"[{self.name}] ({self.result}) Cause: {self.cause}"


@dataclass_json
@dataclass(frozen=True)
class Endpoint:
    """Class for keeping track of an item in inventory."""

    name: str
    timestamp: float
    data: Dict[str, object] = field(default_factory=dict)

    def __repr__(self):
        metadata = f"(data: {self.data})" if self.data else ""
        return f"{self.name} at {self.timestamp} {metadata}"


@dataclass_json
@dataclass(frozen=True)
class DiagnosticTestResult:
    name: str
    timestamp: float
    endpoints: Union[List[List[Endpoint]], List[Endpoint]]
    results: Union[List[List[TestResult]], List[TestResult]]
    valid: bool = True
    temporal: bool = False

    def __repr__(self):
        ret += "- ".join([r.__repr__() for r in self.results])
        ret += "\n"
        return ret


@dataclass_json
@dataclass
class TestSet:
    # fmt: Format
    temporal: bool
    test_results: Union[
        List[List[DiagnosticTestResult]],
        List[DiagnosticTestResult],
    ]
    temporal_test_results: Optional[List[DiagnosticTestResult]] = None
