import pickle
from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations
import yaml
from pgmpy.models import MarkovNetwork

from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph
from diagnosability.factors import *
from diagnosability.perception_system import System
from diagnosability.standard_tests import NoisyOrTest


class TemporalDiagnosticFactorGraph(DiagnosticFactorGraph):
    def __init__(
        self, dfg: DiagnosticFactorGraph, winsize: int, auto_chain: bool = False
    ):
        assert winsize > 0, "winsize must be positive"
        assert isinstance(
            dfg, DiagnosticFactorGraph
        ), "dfg must be a DiagnosticFactorGraph"
        super().__init__(dfg.system)
        self.winsize = winsize
        self.temporal_systems = [self.system]
        for _ in range(1, winsize):
            sys = self.system.copy(update_uuid=True)
            self.add_nodes_from(f.varname for f in sys.get_failure_modes())
            self.temporal_systems.append(sys)
        # Add factors (while keeping track of the tests)
        factors = [phi.copy() for phi in dfg.factors]
        self.temporal_tests = {
            phi.test.name: {0: phi.test}
            for phi in factors
            if isinstance(phi, TestFactor)
        }
        rev_query_by_varname = lambda v: self.system.rev_query(
            self.system.find_by_varname(v)
        )
        # Assign a label to each factor in the base graph
        for phi in dfg.factors:
            for t in range(1, self.winsize):
                if isinstance(phi, PriorFactor):
                    fm = self.temporal_systems[t].query(
                        rev_query_by_varname(phi.scope()[0])
                    )
                    fail_prob = phi.values[1]
                    factors.append(PriorFactor(fm, fail_prob))
                elif isinstance(phi, TestFactor):
                    test_new = deepcopy(phi.test)
                    test_new.update_uuid()
                    test_new.scope = [
                        self.temporal_systems[t].query(self.system.rev_query(v))
                        for v in phi.test.scope
                    ]
                    self.temporal_tests[phi.test.name][t] = test_new
                    factors.append(TestFactor(test_new, phi.values.ravel()))
                elif isinstance(phi, TemporalContraintFactor):
                    # Must be checked before ConstraintFactor
                    raise RuntimeError("TemporalContraintFactor not allowed")
                elif isinstance(phi, ConstraintFactor):
                    scope = [
                        self.temporal_systems[t].query(rev_query_by_varname(v))
                        for v in phi.scope()
                    ]
                    factors.append(
                        ConstraintFactor(phi.name, scope, phi.values.ravel())
                    )
                elif isinstance(phi, MaxCardinalityFactor):
                    scope = [
                        self.temporal_systems[t].query(rev_query_by_varname(v))
                        for v in phi.scope()
                    ]
                    factors.append(MaxCardinalityFactor(scope, phi.values.ravel()))
        self.add_factors(factors)
        if auto_chain:
            self.auto_temporal_chain()

    @classmethod
    def from_file(cls, filename: str, winsize: int):
        assert winsize > 1, "winsize must be greater than 1"
        with open(filename, "r") as stream:
            cfg = yaml.safe_load(stream)
        dgraph = DiagnosticFactorGraph.from_file(filename)
        tgraph = cls(dgraph, winsize=winsize, auto_chain=cfg["auto_chain"])
        # Parse temporal tests
        test_scopes = defaultdict(list)
        test_signatures = defaultdict(int)
        for testname in cfg["temporal"]:
            for fm_and_t in cfg["temporal"][testname]["scope"]:
                fm, t = fm_and_t.split("|")
                q = cfg["failure_modes"][fm]
                test_scopes[testname].append((q, int(t)))
                test_signatures[testname] = max(test_signatures[testname], int(t))
        temporal_tests = []
        for testname, step in test_signatures.items():
            for tau in range(tgraph.winsize - step):
                scope = [
                    tgraph.sys_query(q, t=tau + t) for q, t in test_scopes[testname]
                ]
                phi = NoisyOrTest(f"{testname}", scope)
                phi.test.timestep = tau
                temporal_tests.append(phi)
        tgraph.add_factors(temporal_tests)
        return tgraph

    def sys_failure_modes(self):
        modules = [
            sys.get_failure_modes(System.Filter.MODULE_ONLY)
            for sys in self.temporal_systems
        ]
        outputs = [
            sys.get_failure_modes(System.Filter.OUTPUT_ONLY)
            for sys in self.temporal_systems
        ]
        return {"modules": list(chain(*modules)), "outputs": list(chain(*outputs))}

    def auto_temporal_chain(
        self, same_state_density: float = 0.5, change_state_density: float = 0.1
    ):
        assert isinstance(self, TemporalDiagnosticFactorGraph)
        assert self.winsize > 1
        pmf = np.array(
            [
                same_state_density,
                change_state_density,
                change_state_density,
                same_state_density,
            ]
        )
        failure_modes = [
            self.system.rev_query(f) for f in self.system.get_failure_modes()
        ]
        temporal = []
        for t in range(self.winsize - 1):
            for fm in failure_modes:
                temporal.append(
                    (
                        [
                            self.sys_query(fm, t=t),
                            self.sys_query(fm, t=t + 1),
                        ],
                        pmf,
                    )
                )
        self.add_factors(
            [
                TemporalContraintFactor(f"tau_{i}", r[0], r[1])
                for i, r in enumerate(temporal)
            ]
        )

    def sys_query(self, query, t=0):
        return self.temporal_systems[t].query(query)

    def sys_rev_query(self, query, t=0):
        return self.temporal_systems[t].rev_query(query)
    
    def _sys_rev_query(self, obj):
        for t in range(self.winsize):
            q = self.sys_rev_query(obj, t=t)
            if q is not None:
                return q, t
        return None, None
    
    def sys_query_by_varname(self, varname):
        for sys in self.temporal_systems:
            unit = sys.find_by_varname(varname)
            if unit is not None:
                return unit
        return None

    def save(self, filename):
        picklefile = open(filename, "wb")
        pickle.dump(self, picklefile)
        picklefile.close()

    @classmethod
    def load(cls, filename):
        if isinstance(filename, str):
            picklefile = open(filename, "rb")
        else:
            picklefile = filename
        dfg = pickle.load(picklefile)
        picklefile.close()
        return dfg

    def to_markov_model(self, ignore_modules=False):
        mm = MarkovNetwork()
        vars = set(self.variables)
        ignore_vars = set()
        if ignore_modules:
            ignore_vars = {
                v.varname
                for sys in self.temporal_systems
                for v in sys.get_failure_modes(System.Filter.MODULE_ONLY)
            }
        mm.add_nodes_from(vars - ignore_vars)
        for factor in self.factors:
            scope = set(factor.scope()) - ignore_vars
            mm.add_edges_from(combinations(scope, 2))
        return mm
