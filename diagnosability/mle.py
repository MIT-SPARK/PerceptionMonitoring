from pgmpy.factors.base import factor_divide
from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph
from itertools import product
import numpy as np
from math import exp, log
from scipy import optimize
import diagnosability
import pandas as pd

from tqdm import tqdm
from time import perf_counter

import logging
logger = logging.getLogger(__name__)
FORMAT = "[%(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)

class MLEFit():
    """MLE parameter estimation for a factor graph.
    """

    def __init__(self, factograph: diagnosability.DiagnosticFactorGraph,
                 data: pd.DataFrame) -> None:
        # assert isinstance(factograph, DiagnosticFactorGraph), "The factor graph must be an instance of a DiagnosticFactorGraph."
        self.fg = factograph
        self.fg_0 = self.fg .copy()
        self.data = data
        self.x_0 = None
        self.freq = self._compute_frequencies()

    def fit(self, ftol=1e-8, gtol=1e-5):
        """Start optimization.

        ftol : float
            The iteration stops when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
        gtol : float
            The iteration will stop when ``max{|proj g_i | i = 1, ..., n} <= gtol``
            where ``g_i`` is the i-th component of the projected gradient.
        """
        dim = self.fg.parameters.shape[0]
        bounds = tuple([(0.0, 1.0)] * dim)
        self.x_0 = self.fg.parameters
        res = optimize.minimize(self.loss,
                                self.x_0,
                                method='L-BFGS-B',
                                bounds=bounds,
                                options={'gtol': gtol, 'ftol': ftol, "disp": True})
        self.fg.parameters = res.x
        return res

    def loss(self, theta):
        logger.debug('Start') 
        t0 = perf_counter()
        self.fg.parameters = theta
        E = np.sum(theta * self.freq)
        lnZ = self._approx_log_partition_function()
        t1 = perf_counter()
        logger.debug(f"done {t1-t0}")
        return lnZ - E

    def _log_partition_function(self):
        logger.debug('Start') 
        t0 = perf_counter()
        Z = 0
        variables = self.fg.variables
        for assignment in product(*[range(2) for _ in variables]):
            x = dict(zip(variables, assignment))
            Z_factor = 0
            for factor in self.fg.factors:
                x_scoped = {k: x[k] for k in factor.scope()}
                Z_factor += factor.get_value(**x_scoped)
            Z += exp(Z_factor)
        t1 = perf_counter()
        logger.debug(f"done {t1-t0}")
        return np.log(Z)

    def _approx_log_partition_function(self, n=1000):
        Z = 0
        variables = self.fg.variables
        k = 0
        for assignment in product(*[range(2) for _ in variables]):
            x = dict(zip(variables, assignment))
            Z_factor = 0
            for factors in zip(self.fg.factors, self.fg_0.factors):
                x_scoped = {k: x[k] for k in factors[0].scope()}
                Z_factor += factors[0].get_value(**x_scoped) - factors[1].get_value(**x_scoped)
            Z += exp(Z_factor)
            k += 1
            if k >= n:
                break
        return np.log(Z/k)


    def _compute_frequencies(self):
        freq = []
        for factor in self.fg.factors:
            for assignment in product(
                    *[range(card) for card in factor.cardinality]):
                query = dict(zip(factor.scope(), assignment))
                _, est = self._df_select(query)
                freq.append(est)
        return np.array(freq)

    def _df_select(self, query):
        query_str = " and ".join(f"{k} == {v}" for k, v in query.items())
        masked_df = self.data.query(query_str)
        perc = masked_df.shape[0] / self.data.shape[0]
        return masked_df, perc