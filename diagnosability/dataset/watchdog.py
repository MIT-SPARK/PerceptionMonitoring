from time import perf_counter
from diagnosability.base import (
    DiagnosticModel,
    FailureModeState,
    FailureStates,
    SystemState,
    TestOutcome,
)
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph
from diagnosability.factors import TestFactor
from diagnosability.perception_system import Module, System
from diagnosability.temporal_diagnostic_graph import TemporalDiagnosticFactorGraph
from collections import namedtuple


class Watchdog(DiagnosticModel):
    InferenceResults = namedtuple("InferenceResults", ["Inference", "Timing"])

    def __init__(self, dgraph: DiagnosticFactorGraph, ingore_reliability_score=False, ignore_modules: bool = False):
        self.model = dgraph
        self.ignore_reliability_score = ingore_reliability_score
        if isinstance(dgraph, TemporalDiagnosticFactorGraph):
            if not ignore_modules:
                self._failure_modes = dgraph.failure_modes
                self._modules = {
                    m.varname: m for sys in dgraph.temporal_systems for m in sys
                }
            else:
                self._failure_modes = {
                    f.varname
                    for sys in dgraph.temporal_systems
                    for f in sys.get_failure_modes(System.Filter.OUTPUT_ONLY)
                }
                self._modules = None
            # self._tests = dgraph.temporal_tests
        elif isinstance(dgraph, DiagnosticFactorGraph):
            if not ignore_modules:
                self._failure_modes = dgraph.failure_modes
                self._modules = {m.varname: m for m in dgraph.system}
            else:
                self._failure_modes = {
                    f.varname
                    for f in dgraph.system.get_failure_modes(System.Filter.OUTPUT_ONLY)
                }
                self._modules = None
        self._tests = dict()
        for phi in dgraph.get_factors(filter=lambda phi: isinstance(phi, TestFactor)):
            self._tests[phi.test.varname] = phi.test.scope

    def batch_fault_identification(self, dataset: DiagnosabilityDataset):
        results = []
        timings = []
        for i in range(len(dataset)):
            sample = dataset[i]
            t_start = perf_counter()
            f = self.fault_identification(sample)
            t_end = perf_counter()
            results.append(f)
            timings.append((t_end - t_start) * 1e3)
        return self.InferenceResults(results, timings)

    def _module_reliabiliy_score(self, module_name):
        if self.ignore_reliability_score:
            return 1
        if "radar" in module_name:
            return 3
        elif "fusion" in module_name:
            return 2
        elif "lidar" in module_name:
            return 1
        else:
            return 1

    def _failure_mode_score(self, failure_mode):
        if isinstance(self.model, TemporalDiagnosticFactorGraph):
            for sys in self.model.temporal_systems:
                x = failure_mode
                while not isinstance(x, Module):
                    x = sys.parent(x)
                    if x is None:
                        break
                if isinstance(x, Module):
                    return self._module_reliabiliy_score(x.name)
            raise ValueError(f"Could not find module for {failure_mode.varname}")
        else:
            x = failure_mode
            while not isinstance(x, Module):
                x = self.model.system.parent(x)
                assert (
                    x is not None
                ), f"Could not find module for {failure_mode.varname}"
            return self._module_reliabiliy_score(x.name)

    def fault_identification(self, syndrome: SystemState) -> FailureStates:
        state = FailureStates(
            {f: FailureModeState.INACTIVE for f in self.failure_modes}
        )
        for test, outcome in syndrome.syndrome.items():
            if outcome == TestOutcome.FAIL:
                scored_failure_modes = {
                    f.varname: self._failure_mode_score(f)
                    for f in self._tests[test]
                    if f.varname in self._failure_modes
                }
                min_score = min(scored_failure_modes.values())
                for f, score in scored_failure_modes.items():
                    if score == min_score:
                        state[f] = FailureModeState.ACTIVE
        if self._modules:
            for _, m in self._modules.items():
                xi = [
                    any(
                        state[f.varname] == FailureModeState.ACTIVE
                        for f in o.failure_modes
                    )
                    for o in m.outputs
                ]
                if any(xi):
                    for f in m.failure_modes:
                        state[f.varname] = FailureModeState.ACTIVE
        return state

    @property
    def failure_modes(self):
        return self.model.failure_modes

    @property
    def tests(self):
        return self.model.tests


# class Watchdog:
#     def __init__(
#         self,
#         dataset_filename: str,
#         model_config: str,
#     ):
#         self.dataset = DiagnosabilityDatasetPreprocessor(dataset_filename, model_config)
#         with open(model_config, "r") as stream:
#             self.cfg = yaml.safe_load(stream)

#     def run(self):
#         for sample in tqdm(self.dataset.data, "Samples"):
#             state = FailureStates(
#                 {
#                     f.varname: FailureModeState.INACTIVE
#                     for sys in self.dataset.dfg.temporal_systems
#                     for f in sys.get_failure_modes()
#                 }
#             )
#             for tau, window in enumerate(sample.test_results):
#                 for dtest in window:
#                     endpoints = {
#                         e.name: e.data["confidence"]
#                         if e.data["confidence"] is not None
#                         else 0.0
#                         for e in dtest.endpoints
#                     }
#                     failing_endpoint = min(endpoints, key=endpoints.get)
#                     for test in dtest.results:
#                         if test.name in self.dataset.ground_truth_test_names:
#                             continue
#                         if test.result == TestOutcome.PASS:
#                             continue
#                         ok = False
#                         x = self.dataset.dfg.temporal_systems[tau].query(
#                             self.cfg["endpoints"][failing_endpoint]
#                         )
#                         candidates = {f.varname for f in x.failure_modes}
#                         for f in test.scope:
#                             varname = (
#                                 self.dataset.dfg.temporal_systems[tau]
#                                 .query(self.cfg["failure_modes"][f])
#                                 .varname
#                             )
#                             if varname in candidates:
#                                 state[varname] = FailureModeState.ACTIVE
#                                 ok = True
#                         assert ok
#             self.data.append(state)
