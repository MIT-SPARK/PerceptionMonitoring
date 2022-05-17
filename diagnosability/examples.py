from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph
from diagnosability.perception_system import (
    FailureMode,
    Output,
    Module,
    System,
    DefaultDatatype,
)
from diagnosability.factors import PriorFactor, ConstraintFactor
from diagnosability.standard_tests import IdealTest, PerfectTest
from diagnosability.perception_system import FunctionalModuleState
import numpy as np


def all_or_none(x):
    return all(x) or not any(x)


def nay(x):
    return not any(x)


######################## Systems ##############################


def example_system_dummy():
    # Module 1
    output_1 = Output(
        name="output 1",
        datatype=DefaultDatatype.NUMERIC,
        failure_modes=[FailureMode("failure o1")],
    )
    module_1 = Module(
        name="module 1",
        inputs=[],
        outputs=[output_1],
        failure_modes=[FailureMode("failure m1")],
    )
    # Module 2
    output_2 = Output(
        name="output 2",
        datatype=DefaultDatatype.NUMERIC,
        failure_modes=[FailureMode("failure o2")],
    )
    module_2 = Module(
        name="module 2",
        inputs=[],
        outputs=[output_2],
        failure_modes=[FailureMode("failure m2")],
    )
    # Module 3
    output_3 = Output(
        name="output 3",
        datatype=DefaultDatatype.NUMERIC,
        failure_modes=[FailureMode("failure o3")],
    )
    module_3 = Module(
        name="module 3",
        inputs=[output_1, output_2],
        outputs=[output_3],
        failure_modes=[FailureMode("failure m3")],
    )
    # SYSTEM
    sys = System([module_1, module_2, module_3])
    return sys


######################## Diagnostic Factor Graphs ########################


def example_dfg_dummy(failure_probability=0.1, alpha=0.1, beta=0.1):
    sys = example_system_dummy()
    dfg = DiagnosticFactorGraph(sys)
    # Priors
    dfg.add_factors(
        [
            PriorFactor(m, failure_probability=failure_probability)
            for m in sys.get_failure_modes(filter=System.Filter.MODULE_ONLY)
        ]
    )
    # Relationships
    relations = [
        (
            [
                sys.query("module 1 . failure m1"),
                sys.query("module 1 -> output 1 . failure o1"),
            ],
            lambda x: all_or_none(x),
        ),
        (
            [
                sys.query("module 2 . failure m2"),
                sys.query("module 2 -> output 2 . failure o2"),
            ],
            lambda x: all_or_none(x),
        ),
        (
            [
                sys.query("module 3 . failure m3"),
                sys.query("module 1 -> output 1 . failure o1"),
                sys.query("module 2 -> output 2 . failure o2"),
            ],
            np.array(
                [1.0, alpha, beta, alpha * beta, alpha * beta, 1 - alpha, 1 - beta, 1.0]
            ),
        ),
        (
            [
                sys.query("module 3 . failure m3"),
                sys.query("module 3 -> output 3 . failure o3"),
            ],
            lambda x: all_or_none(x),
        ),
    ]
    dfg.add_factors([ConstraintFactor(f"r_{i}", r[0], r[1]) for i, r in enumerate(relations)])
    # Tests
    t_scopes = {
        "t0": [
            *sys.query("module 1 -> output 1 . *"),
            *sys.query("module 2 -> output 2 . *"),
        ],
        "t1": [
            *sys.query("module 2 -> output 2 . *"),
            *sys.query("module 3 -> output 3 . *"),
        ],
    }
    tests = [PerfectTest(t, scope) for t, scope in t_scopes.items()]
    dfg.add_factors(tests)
    return dfg
