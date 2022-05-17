import pickle
import warnings
from itertools import combinations
from pathlib import Path
from re import S
from typing import List

import pandas as pd
import yaml
from pgmpy.factors.base import factor_product
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph, MarkovNetwork

from diagnosability.base import (
    DiagnosticModel,
    FailureStates,
    Syndrome,
    SystemState,
    TestOutcome,
)
from diagnosability.diagnostic_junction_tree import DiagnosticJunctionTree
from diagnosability.factors import *
from diagnosability.fault_identification import PreprocessFaultIdentification
from diagnosability.perception_system import NamedObject, System
from diagnosability.standard_tests import NoisyOrTest


class DiagnosticFactorGraph(FactorGraph, DiagnosticModel):
    def __init__(
        self,
        system: System,
    ):
        super().__init__()
        assert isinstance(
            system, System
        ), "The system should be an instance of the class System."
        self.system = system
        self.add_nodes_from(f.varname for f in system.get_failure_modes())
        self._inference = PreprocessFaultIdentification(self)

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as stream:
            cfg = yaml.safe_load(stream)
        sys = System.from_yaml(Path(cfg["system"]))
        dfg = cls(sys)
        if cfg["with_priors"]:
            dfg.add_factors(
                [
                    PriorFactor(m, failure_probability=0.1)
                    for m in sys.get_failure_modes(filter=System.Filter.MODULE_ONLY)
                ]
            )
        fmode = {f: sys.query(q) for f, q in cfg["failure_modes"].items()}
        test_scopes = {
            t: [fmode[f] for f in cfg["tests"][t]["scope"]] for t in cfg["tests"]
        }
        tests = [NoisyOrTest(t, scope) for t, scope in test_scopes.items()]
        dfg.add_factors(tests)
        rels = []
        for r in cfg["relations"]:
            rel_def = cfg["relations"][r]
            assert (set(rel_def["A"]) & set(rel_def["B"])) == set()
            a = [sys.query(q) for q in rel_def["A"]]
            b = [sys.query(q) for q in rel_def["B"]]
            scope = a + b
            if rel_def["type"] == "AtLeastOne":
                values = AtLeastOne(a, b)
            elif rel_def["type"] == "Majority":
                values = Majority(a, b)
            else:
                raise ValueError("Unknown relation type: {}".format(rel_def["type"]))
            rels.append(ConstraintFactor(name=r, scope=scope, values=values))
        dfg.add_factors(rels)
        return dfg

    @property
    def variables(self) -> Set[str]:
        """Returns all the variables in the graph.

        Returns:
            Set[str]: list of variables labels
        """
        return set([n for n in self.nodes if not isinstance(n, DiscreteFactor)])

    @property
    def tests(self):
        return set(
            [phi.test.varname for phi in self.factors if isinstance(phi, TestFactor)]
        )

    @property
    def failure_modes(self):
        return self.variables - self.tests

    def sys_failure_modes(self):
        return {
            "modules": self.system.get_failure_modes(System.Filter.MODULE_ONLY),
            "outputs": self.system.get_failure_modes(System.Filter.OUTPUT_ONLY),
        }

    def sys_query_by_varname(self, varname):
        for phi in self.get_factors(filter=lambda phi: isinstance(phi, TestFactor)):
            if phi.test.varname == varname:
                return phi.test
        return self.system.find_by_varname(varname)

    def sys_query(self, query):
        for phi in self.get_factors(filter=lambda phi: isinstance(phi, TestFactor)):
            if phi.test.name == query:
                return phi.test
        return self.system.query(query)

    def sys_rev_query(self, obj: NamedObject):
        for phi in self.get_factors(filter=lambda phi: isinstance(phi, TestFactor)):
            if phi.test is obj:
                return f"{phi.test.name}"
        return self.system.rev_query(obj)
    
    def _sys_rev_query(self, obj: NamedObject):
        # Used for exporting to dict
        return self.sys_rev_query(obj), 0

    def __str__(self):
        variables = self.variables
        s = f"VARIABLES ({len(variables)}):\n"
        for v in variables:
            s += f"\t{v}\n"
        s += f"FACTORS ({len(self.factors)}):\n"
        for f in self.factors:
            if isinstance(f, TestFactor):
                t = "Test       "
            elif isinstance(f, PriorFactor):
                t = "Prior      "
            elif isinstance(f, TemporalContraintFactor):
                t = "Temporal   "
            elif isinstance(f, ConstraintFactor):
                t = "Constraint "
            elif isinstance(f, MaxCardinalityFactor):
                t = "MaxCard    "
            else:
                t = "Unknown    "
            s += f"\t{t} on {f.scope()}\n"
        return s

    def add_factors(self, factors: List[DiscreteFactor]):
        assert all(
            [isinstance(phi, DiscreteFactor) for phi in factors]
        ), "All factors must be DiscreteFactors (or derived)."
        for phi in factors:
            for v in phi.scope():
                self.add_node(v)
            self.add_node(phi)
            self.add_edges_from([(v, phi) for v in phi.scope()])
            super().add_factors(phi)
        self._inference.invalidate()

    def copy(self):
        cpy = DiagnosticFactorGraph(self.system)
        cpy.add_factors([phi.copy() for phi in self.factors])
        return cpy

    def fault_identification(self, syndrome: SystemState) -> FailureStates:
        assert isinstance(
            syndrome, SystemState
        ), "The syndrome should be an instance of the class Syndrome."
        return self._inference.most_probable_failures(syndrome.syndrome)

    def get_factors(self, filter: Callable = lambda phi: True):
        return [phi for phi in self.factors if filter(phi)]

    def get_factor_product(self, filter: Callable = lambda phi: True):
        factors = self.get_factors(filter)
        if not factors:
            return None
        return factor_product(*factors)

    def sample(self, num_samples: int, balanced=True) -> pd.DataFrame:
        if balanced:
            phi = self.get_factor_product(
                filter=lambda phi: not isinstance(phi, PriorFactor)
            )
            return phi.sample(num_samples)
        else:
            # TODO: This sampling strategy might be slow and memory intensive
            # gibbs = GibbsSampling(self.to_markov_model())
            # return gibbs.sample(size=num_samples)
            phi = self.get_factor_product()
            return phi.sample(num_samples)

    def check_model(self):
        super().check_model()
        # TODO: should we check also for factor type? e.g. failure mode prior has 1 failure mode in its scope etc..
        # variables = self.variables
        return True

    def to_diagnostic_junction_tree(self):
        return DiagnosticJunctionTree(self)

    ## I/O
    def save(self, filename):
        picklefile = open(filename, "wb")
        pickle.dump(self, picklefile)
        picklefile.close()

    def to_markov_model(self, ignore_modules=False):
        mm = MarkovNetwork()
        vars = set(self.variables)
        ignore_vars = set()
        if ignore_modules:
            ignore_vars = {
                v.varname
                for v in self.system.get_failure_modes(System.Filter.MODULE_ONLY)
            }
        mm.add_nodes_from(vars - ignore_vars)
        for factor in self.factors:
            scope = set(factor.scope()) - ignore_vars
            mm.add_edges_from(combinations(scope, 2))
        return mm

    def to_dict(self, include_modules=True):
        """Export variables and factors to a dictionary. Useful to dum a DFG to json.

        Examples:
        with open("dfg.json", "w") as outfile:
            json.dump(dfg.to_dict(), outfile)
        """
        variables = []
        tests = []
        relationships = []
        fmodes = self.sys_failure_modes()
        index = 0
        vars_ = fmodes["outputs"]
        for v in vars_:
            q, w = self._sys_rev_query(v)
            v_dict = {
                "name": v.name,
                "varname": v.varname,
                "type": "output",
                "prior": None,
                "index": index,
                "query": q,
                "window": w,
                "severity": v.severity,
            }
            variables.append(v_dict) 
            index += 1
        if include_modules:
            vars_ = fmodes["modules"]
            for v in vars_:
                q, w = self._sys_rev_query(v)
                v_dict = {
                    "name": v.name,
                    "varname": v.varname,
                    "type": "module",
                    "prior": None,
                    "index": index,
                    "query": q,
                    "window": w,
                    "severity": v.severity,
                }
                variables.append(v_dict)
                index += 1
        var_varnames = {v["varname"] for v in variables}
        for idx, phi in enumerate(self.factors):
            if isinstance(phi, TestFactor):
                scope = {v.varname for v in phi.test.scope}
                marginalize = scope - var_varnames
                if marginalize:
                    f = phi.marginalize(marginalize, inplace=False)
                else:
                    f = phi
                if not f.scope():
                    continue
                t = {
                    "type": "test",
                    "name": phi.test.name,
                    "varname": phi.test.varname,
                    "scope": [v for v in f.scope() if v in scope],
                    "densities": f.values.ravel().tolist(),
                }
                tests.append(t)
            elif isinstance(phi, PriorFactor) and include_modules:
                # Should set the prior...
                basevar = phi.scope()[0]
                for v in variables:
                    if v["varname"] == basevar:
                        v["prior"] = phi.values.ravel()[1]
                        break
            # elif isinstance(phi, TemporalContraintFactor):
            #     scope = set(phi.scope())
            #     marginalize = scope - var_varnames
            #     if marginalize:
            #         f = phi.marginalize(marginalize, inplace=False)
            #     else:
            #         f = phi
            #     if not f.scope():
            #         continue
            #     r = {
            #         "type": "temporal",
            #         "name": f"r{len(relationships)}",
            #         "scope": f.scope(),
            #         "densities": f.values.ravel().tolist(),
            #     }
            #     relationships.append(r)
            elif isinstance(phi, ConstraintFactor):
                scope = set(phi.scope())
                marginalize = scope - var_varnames
                if marginalize:
                    f = phi.marginalize(marginalize, inplace=False)
                else:
                    f = phi
                if not f.scope():
                    continue
                rtype = (
                    "temporal"
                    if isinstance(phi, TemporalContraintFactor)
                    else "constraint"
                )
                r = {
                    "type": rtype,
                    "name": phi.name,
                    "varname": f"rel_{idx}",
                    "scope": f.scope(),
                    "densities": f.values.ravel().tolist(),
                }
                relationships.append(r)
            elif isinstance(phi, MaxCardinalityFactor):
                continue
        return {"variables": variables, "tests": tests, "relationships": relationships}

    @classmethod
    def load(cls, filename):
        if isinstance(filename, str):
            picklefile = open(filename, "rb")
        else:
            picklefile = filename
        dfg = pickle.load(picklefile)
        picklefile.close()
        return dfg
