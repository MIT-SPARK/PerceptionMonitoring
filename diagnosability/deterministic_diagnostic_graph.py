from diagnosability.base import (
    DiagnosticModel,
    FailureStates,
    SystemState,
    TestOutcome,
)
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset

from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from diagnosability.factors import TestFactor

from diagnosability.perception_system import System
from typing import Union, Dict, List
from collections import namedtuple
from time import perf_counter
import itertools


class DeterministicDiagnosticGraph(DiagnosticModel):
    InferenceResults = namedtuple("InferenceResults", ["Inference", "Timing"])

    def __init__(
        self,
        system: Union[System, List[System]],
        tests: Union[Dict[str, List[str]], DiagnosticFactorGraph],
        ignore_modules=False,
    ):
        self.system = system
        self.ignore_modules = ignore_modules
        if isinstance(tests, DiagnosticFactorGraph):
            self._tests = {
                phi.test.varname: {f.varname for f in phi.test.scope}
                for phi in tests.factors
                if isinstance(phi, TestFactor)
            }
        else:
            self._tests = tests
        (
            self._variables,
            self._modules_fm,
            self._outputs_fm,
            self._modules,
            self._vmap,
        ) = self._get_variables()

    @property
    def failure_modes(self):
        return {f.varname for f in self.system.get_failure_modes()}

    @property
    def tests(self):
        return set(self._tests.keys())

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

    def _get_variables(self):
        if isinstance(self.system, System):
            modules_fm = {
                v.varname
                for v in self.system.get_failure_modes(System.Filter.MODULE_ONLY)
            }
            outputs_fm = {
                v.varname
                for v in self.system.get_failure_modes(System.Filter.OUTPUT_ONLY)
            }
            modules = {m.varname: m for m in self.system}
        else:
            # Temporal Graph
            modules_fm = {
                v.varname
                for sys in self.system
                for v in sys.get_failure_modes(System.Filter.MODULE_ONLY)
            }
            outputs_fm = {
                v.varname
                for sys in self.system
                for v in sys.get_failure_modes(System.Filter.OUTPUT_ONLY)
            }
            modules = {m.varname: m for sys in self.system for m in sys}

        assert modules_fm.isdisjoint(
            outputs_fm
        ), "modules and outputs cannot be in the same set"
        variables = outputs_fm
        if not self.ignore_modules:
            variables = variables | modules_fm
        else:
            variables = outputs_fm
        variables = sorted(list(variables))
        vmap = {v: i for i, v in enumerate(variables)}
        return variables, modules_fm, outputs_fm, modules, vmap

    def fault_identification(self, syndrome: SystemState) -> FailureStates:
        solver = pywraplp.Solver.CreateSolver("SCIP")
        x = [solver.IntVar(0, 1, v) for v in self._variables]
        curr_vars = set(self._variables)
        for t, tscope in self._tests.items():
            outcome = syndrome.syndrome[t]
            scope = tscope & curr_vars  # tscope - ignore_vars
            vv = [x[self._vmap[v]] for v in scope]
            if outcome == TestOutcome.FAIL:
                # At leat one active failure mode
                solver.Add(sum(vv) >= 1)
            # else:
                # All Equal
                # solver.Add((len(vv)-1)*vv[0] == sum(vv[1:]))
        if not self.ignore_modules:
            for _, m in self._modules.items():
                all_outputs_fm = list(
                    itertools.chain(
                        *[
                            [x[self._vmap[f.varname]] for f in o.failure_modes]
                            for o in m.outputs
                        ]
                    )
                )
                module_fm = [x[self._vmap[f.varname]] for f in m.failure_modes]
                solver.Add(len(all_outputs_fm) * sum(module_fm) >= sum(all_outputs_fm))
            # for _, m in self._modules.items():
            #     m_fm = [x[self._vmap[f.varname]] for f in m.failure_modes]
            #     z = []
            #     for o in m.outputs:
            #         aux = solver.IntVar(0, 1, f"{o.varname}")
            #         xi.append(aux)
            #         z.append(aux)
            #         o_fm = [x[self._vmap[f.varname]] for f in o.failure_modes]
            #         solver.Add(len(o_fm) * aux >= sum(o_fm))
            #     solver.Add(len(z)*sum(m_fm) >= sum(z))
        solver.Minimize(sum(x))
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            print("Could not find optimal solution")
        return FailureStates(
            {v: int(x[i].solution_value()) for i, v in enumerate(self._variables)}
        )
        # else:
        # return FailureStates({v:0 for i,v in enumerate(self._variables)})
