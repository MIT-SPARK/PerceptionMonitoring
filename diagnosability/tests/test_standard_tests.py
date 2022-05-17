import pytest
import numpy as np
from diagnosability.standard_tests import (
    IdealTest,
    PerfectTest,
    WellBehavedTest,
    RandomTest,
)
from diagnosability.perception_system import FailureMode
from diagnosability.factors import TestFactor


def test_ideal():
    scope = [FailureMode("f1"), FailureMode("f2")]
    test = IdealTest("t", scope)
    assert isinstance(test, TestFactor)
    assert set(test.scope()) == set([test.test.varname]) | set(
        f.varname for f in test.test.scope
    )
    assert test.values.shape == (2, 2, 2)
    assert np.array_equal(
        test.values.ravel(), np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    )


def test_perfect():
    scope = [FailureMode("f1"), FailureMode("f2")]
    test = PerfectTest("t", scope)
    assert isinstance(test, TestFactor)
    assert set(test.scope()) == set([test.test.varname]) | set(
        f.varname for f in test.test.scope
    )
    assert test.values.shape == (2, 2, 2)
    assert np.array_equal(
        test.values.ravel(), np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    )


def test_well_behaved():
    scope = [FailureMode("f1")]
    Pd = np.array([0.9])
    Pf = np.array([0.1])
    test = WellBehavedTest("t", scope=scope, Pd=Pd, Pf=Pf)
    assert isinstance(test, TestFactor)
    assert set(test.scope()) == set([test.test.varname]) | set(
        f.varname for f in test.test.scope
    )
    assert test.values.shape == (2, 2)
    np.array_equal(test.values.ravel(), np.array([0.9, 0.1, 0.1, 0.9]))

    scope = [FailureMode("f1"), FailureMode("f2")]
    Pd = np.array([0.9, 0.95])
    Pf = np.array([0.1, 0.05])
    test = WellBehavedTest("t", scope=scope, Pd=Pd, Pf=Pf)
    assert isinstance(test, TestFactor)
    assert set(test.scope()) == set([test.test.varname]) | set(
        f.varname for f in test.test.scope
    )
    assert test.values.shape == (2, 2, 2)
    np.array_equal(
        test.values.ravel(),
        np.array([0.855, 0.045, 0.095, 0.005, 0.005, 0.095, 0.045, 0.855]),
    )


def test_random():
    scope = [FailureMode("f1"), FailureMode("f2")]
    test = RandomTest("t", scope)
    assert isinstance(test, TestFactor)
    assert set(test.scope()) == set([test.test.varname]) | set(
        f.varname for f in test.test.scope
    )
    assert test.values.shape == (2, 2, 2)
