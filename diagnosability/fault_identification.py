from pgmpy.models import FactorGraph
from diagnosability.utils import Powerset
from pgmpy.inference import BeliefPropagation
from itertools import product
from diagnosability.base import *


class FaultIdentificationEngine:
    class Mode(Enum):
        NONE = 0
        QUERY = 1
        MAP = 2

    def __init__(self, model):
        self._mode = self.Mode.NONE
        self._model = model
        self._belief_propagation = None

    def invalidate(self):
        """Invalidate current calibration. Call this function every time the model changes."""
        self._mode = self.Mode.NONE

    def calibrate(self, use_max=True, force=False):
        req_mode = self.Mode.MAP if use_max else self.Mode.QUERY
        if (
            not force
            and self._belief_propagation is not None
            and self._mode == req_mode
        ):
            return
        self._belief_propagation = BeliefPropagation(self._model)
        if req_mode == self.Mode.MAP:
            self._belief_propagation.max_calibrate()
            self._mode = self.Mode.MAP
        else:
            self._belief_propagation.calibrate()
            self._mode = self.Mode.QUERY

    def most_probable_failures(self, syndrome):
        self.calibrate(use_max=True)
        fi = self.map_query(self._model.failure_modes, syndrome.to_dict())
        return FailureStates({f: FailureModeState(v) for f, v in fi.items()})

    def most_probable_syndrome(self, fault_states):
        self.calibrate(use_max=True)
        syn = self.map_query(self._model.tests, fault_states.to_dict())
        return Syndrome({t: TestOutcome(v) for t, v in syn.items()})

    def query(self, variables, evidence):
        self.calibrate(use_max=False)
        return self._belief_propagation.query(variables, evidence, show_progress=False)

    def map_query(self, variables, evidence):
        self.calibrate(use_max=True)
        return self._belief_propagation.map_query(
            variables, evidence, show_progress=False
        )


class PreprocessFaultIdentification:
    class Mode(Enum):
        NONE = 0
        QUERY = 1
        MAP = 2

    def __init__(self, model):
        self._mode = self.Mode.NONE
        self._model = model
        self._fg = None
        self._belief_propagation = None
        self._last_syndrome = None
        self._map_cache = {}

    def invalidate(self, clear_cache=True):
        """Invalidate current calibration. Call this function every time the model changes."""
        self._mode = self.Mode.NONE
        if clear_cache:
            self._map_cache = {}

    def _reduce_dfg(self, syndrome):
        fg = FactorGraph()
        evidence = set(syndrome.keys())
        for v in self._model.variables:
            if v not in evidence:
                fg.add_node(v)
        for phi in self._model.factors:
            evidence_in_scope = set(phi.scope()) & evidence
            if evidence_in_scope:
                # Reduce factor
                evidence_values = [(v, syndrome[v].value) for v in evidence_in_scope]
                factor = phi.reduce(evidence_values, inplace=False)
            else:
                factor = phi.copy()
            fg.add_factors(factor)
            fg.add_node(factor)
            fg.add_edges_from([(v, factor) for v in factor.scope()])
        return fg

    def calibrate(self, syndrome, use_max=True, force=False):
        req_mode = self.Mode.MAP if use_max else self.Mode.QUERY
        if (
            not force
            and self._last_syndrome is not None
            and syndrome == self._last_syndrome
            and self._belief_propagation is not None
            and self._mode == req_mode
        ):
            return
        # Graph reduction
        self._fg = self._reduce_dfg(syndrome)
        # BP calibration
        self._belief_propagation = BeliefPropagation(self._fg)
        if use_max:
            self._belief_propagation.max_calibrate()
            self._mode = self.Mode.MAP
        else:
            self._belief_propagation.calibrate()
            self._mode = self.Mode.QUERY

    def most_probable_failures(self, syndrome):
        self.calibrate(syndrome=syndrome, use_max=True)
        if syndrome in self._map_cache:
            fs = self._map_cache[syndrome]
        else:
            fi = self.map_query(variables=self._model.failure_modes)
            fs = FailureStates({f: v for f, v in fi.items()})
            self._map_cache[syndrome] = fs
        return fs

    def most_probable_syndrome(self, fault_states):
        return None

    def query(self, variables, evidence=None):
        assert self._mode == self.Mode.QUERY
        return self._belief_propagation.query(variables, evidence, show_progress=False)

    def map_query(self, variables, evidence=None):
        assert self._mode == self.Mode.MAP
        return self._belief_propagation.map_query(
            variables, evidence, show_progress=False
        )
