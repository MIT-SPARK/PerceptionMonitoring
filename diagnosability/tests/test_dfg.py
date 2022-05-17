# from diagnosability.examples import example_diagnostic_graph
# from pgmpy.factors.discrete.DiscreteFactor import DiscreteFactor
# import pytest
# from diagnosability import *
# import numpy as np


# def max_num_faults(data, failure_mdoes):
#     max_faults = 0
#     for _, row in data.iterrows():
#         n = sum(row[failure_mdoes])
#         if n > max_faults:
#             max_faults = n
#     return max_faults


# def test_dfg():
#     failure_modes = ["f1", "f2", "f3"]
#     p = np.array([0.5, 0.5])

#     dfg = DiagnosticFactorGraph()
#     dfg.add_failure_modes(
#         failure_modes, [FailureModePriorFactor(f, p) for f in failure_modes]
#     )
#     with pytest.raises(AssertionError) as excinfo:
#         dfg.add_failure_modes(["f1"])
#     assert dfg.num_failure_modes() == 3
#     assert set(dfg.failure_modes) == {"f1", "f2", "f3"}
#     assert dfg.variables == {"f1", "f2", "f3"}
#     assert dfg.num_tests() == 0

#     test_scope = {"t1": ["f1", "f2"], "t2": ["f2", "f3"]}
#     tests = RandomTests(dfg.failure_modes, test_scope)
#     dfg.add_tests(tests.tests, tests.factors)
#     with pytest.raises(AssertionError) as ex:
#         dfg.add_tests(["t1"])
#     assert dfg.num_tests() == 2
#     assert set(dfg.tests) == {"t1", "t2"}
#     assert set(dfg.variables) == {"f1", "f2", "f3", "t1", "t2"}
#     assert len(dfg.factors) == 5
#     assert dfg.parameters.shape == (22,)

#     dfg.add_factors([BudgetFactor(failure_modes, 1)])
#     assert len(dfg.factors) == 6
#     assert dfg.max_num_active_faults == 1

#     for f in dfg.factors:
#         assert isinstance(f, DiscreteFactor)

#     theta = dfg.parameters
#     assert theta.shape == (30,)
#     theta = np.random.random((30,))
#     dfg.parameters = theta
#     assert (dfg.parameters == theta).all()


# def test_dfg_sample():
#     balanced = True
#     dfg = example_diagnostic_graph("f3perfect")

#     fw_samples = dfg.sample(int(1e3), balanced=balanced)
#     assert fw_samples.shape[0] == int(1e3)
#     assert set(fw_samples.columns) == {"f1", "f2", "f3", "t1", "t2"}

#     dfg.add_factors([BudgetFactor(dfg.failure_modes, 2)])
#     fw_samples = dfg.sample(int(1e3), balanced=balanced)
#     assert max_num_faults(fw_samples, dfg.failure_modes) <= 2

#     p = 0.6
#     tol = 10 / 100
#     dfg = example_diagnostic_graph("f3perfect", (p, p))
#     samples = dfg.sample(int(1e4))
#     failures = (
#         samples.groupby(dfg.failure_modes)
#         .size()
#         .reset_index()
#         .rename(columns={0: "freq"})
#     )
#     failures["freq"] = failures["freq"] / samples.shape[0]
#     n = dfg.num_failure_modes()
#     for _, row in failures.iterrows():
#         num_active = sum(row[dfg.failure_modes])
#         expected_frew = p ** num_active * (1 - p) ** (n - num_active)
#         assert expected_frew / (1 + tol) <= row["freq"] <= expected_frew * (1 + tol)


# def test_dfg_fault_identification():
#     dfg = example_diagnostic_graph("f3perfect")
#     dfg.add_factors([BudgetFactor(dfg.failure_modes, 1)])

#     examples = [
#         (Syndrome({"t1": TestOutcome.PASS, "t2": TestOutcome.PASS}), set()),
#         (Syndrome({"t1": TestOutcome.FAIL, "t2": TestOutcome.PASS}), {"f1"}),
#         (Syndrome({"t1": TestOutcome.PASS, "t2": TestOutcome.FAIL}), {"f3"}),
#         (Syndrome({"t1": TestOutcome.FAIL, "t2": TestOutcome.FAIL}), {"f2"}),
#     ]
#     for ex in examples:
#         syn = ex[0]
#         expected = ex[1]
#         fi = dfg.fault_identification(syn)
#         assert fi.active_failures() == expected


# # @pytest.mark.skip(reason="Incomplete logic in class...")
# def test_diagnosability():
#     dfg = example_diagnostic_graph("f3perfect")
#     dfg.add_factors([BudgetFactor(dfg.failure_modes, 1)])
#     diagnosability = dfg.diagnosability()
#     assert diagnosability == 1.0
