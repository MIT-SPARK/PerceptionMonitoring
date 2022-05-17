import pickle
from typing import Dict, List, Optional

import pandas as pd
from pgmpy.models import JunctionTree

from diagnosability.base import DiagnosticModel, FailureStates, Syndrome, SystemState
from diagnosability.fault_identification import PreprocessFaultIdentification


class DiagnosticJunctionTree(JunctionTree, DiagnosticModel):
    """
    A class for constructing a diagnostic junction tree from a diagnostic model.
    """

    def __init__(
        self,
        diagnostic_model: DiagnosticModel,
        junction_tree: Optional[JunctionTree] = None,
    ):
        """
        Initialize a diagnostic junction tree from a diagnostic model.

        :param diagnostic_model: A diagnostic model.
        :param fault_identification_engine: A fault identification engine.
        """
        self._diagnostic_model = diagnostic_model
        if junction_tree is None:
            junction_tree = self._diagnostic_model.to_junction_tree()
        else:
            assert isinstance(junction_tree, JunctionTree)
            # TODO: check variables match
        super(JunctionTree, self).__init__(junction_tree.edges())
        self.add_nodes_from(junction_tree.nodes())
        self.add_factors(*junction_tree.factors)
        self._inference = PreprocessFaultIdentification(self)

    @staticmethod
    def fit_dfg_from_data(diagnostic_model: DiagnosticModel, samples: pd.DataFrame):
        # from pgmpy.estimators import MaximumLikelihoodEstimator
        from diagnosability.bayesian_mle import MaximumLikelihoodEstimator

        bn = diagnostic_model.to_markov_model().to_bayesian_model()
        bn.fit(samples, estimator=MaximumLikelihoodEstimator)
        return DiagnosticJunctionTree(diagnostic_model, bn.to_junction_tree())

    @property
    def failure_modes(self) -> List[str]:
        return self._diagnostic_model.failure_modes

    @property
    def tests(self) -> List[str]:
        return self._diagnostic_model.tests

    @property
    def variables(self) -> List[str]:
        """Returns all the variables in the graph (F \\cup T).

        Returns:
            List[str]: list of variables labels
        """
        return self._diagnostic_model.variables

    def calibrate(self, use_max: bool = True, force: bool = False):
        """Calibrate inference engine."""
        self._inference.calibrate(use_max=use_max, force=force)

    def fault_identification(self, syndrome: SystemState) -> FailureStates:
        """Performs fault identification given a syndrome."""
        assert isinstance(
            syndrome, SystemState
        ), "The syndrome should be an instance of the class Syndrome."
        return self._inference.most_probable_failures(syndrome.syndrome)

    def __str__(self):
        variables = self.variables
        s = f"VARIABLES ({len(variables)}):\n"
        for v in variables:
            s += f"\t{v}\n"
        s += f"NODES ({len(self.nodes)}):\n"
        for n in self.nodes:
            s += f"\t{n}\n"
        s += f"FACTORS ({len(self.factors)}):\n"
        for f in self.factors:
            s += f"\tFactor on {f.scope()}\n"
        s += f"EDGES ({len(self.edges)}):\n"
        for u, v in self.edges:
            s += f"\t{u} -> {v}\n"
        return s

    ## I/O
    def save(self, filename):
        picklefile = open(filename, "wb")
        pickle.dump(self, picklefile)
        picklefile.close()

    @classmethod
    def load(cls, filename):
        picklefile = open(filename, "rb")
        dfg = pickle.load(picklefile)
        picklefile.close()
        return dfg
