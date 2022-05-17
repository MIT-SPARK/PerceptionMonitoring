from typing import List
from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph, Syndrome, TestOutcome
from diagnosability.factors import TestFactor
from diagnosability.utils import Powerset


def all_syndromes(dfg: DiagnosticFactorGraph) -> List[Syndrome]:
    # tests = {
    #     phi.test.name: phi.test.varname
    #     for phi in dfg.get_factors(filter=lambda f: isinstance(f, TestFactor))
    # }
    tests = dfg.tests
    syndromes = []
    for failing_tests in Powerset(tests):
        syndromes.append(
            Syndrome({t: TestOutcome(int(t in failing_tests)) for t in tests})
        )
    return syndromes


def random_syndrome(dfg: DiagnosticFactorGraph) -> Syndrome:
    return Syndrome({t: TestOutcome.random() for t in dfg.tests})

# def failure_mode_only_dfg(dfg):
#     from diagnosability.diagnostic_factor_graph import IgnoreFactorsRules
#     from pgmpy.models import FactorGraph

#     dfg.check_model()
#     # Create the truncated factor graph (omit test factors)
#     fg = FactorGraph()
#     fg.add_nodes_from(dfg.failure_modes)
#     factors = dfg.get_factors(
#         [
#             IgnoreFactorsRules.IGNORE_TEST_FACTORS,
#             IgnoreFactorsRules.IGNORE_ON_TESTS,
#         ],
#         mode="all",
#     )
#     fg.add_nodes_from(factors)
#     fg.add_factors(*factors)
#     # Add edges (if it makes sense)
#     edges = [(v, phi) for phi in factors for v in phi.scope()]
#     fg.add_edges_from(edges)
#     fg.check_model()
#     return fg

# def approx_sample(dfg: DiagnosticGraph, num_samples: int):
#     import pandas as pd
#     from pgmpy.sampling import GibbsSampling
        
#     # # TODO: raise warning if failure mode are not independend
#     priors = dfg.get_priors()    
#     failure_samples = pd.concat([phi.sample(num_samples) for phi in priors.values() if phi is not None], axis=1)

#     # Get temporal constraints (if needed)
#     temporal_failures = {k for k, v in priors.items() if v is None}
#     if temporal_failures:
#         for phi in dfg.factors:
#             scope = set(phi.scope())
#             if ...


#     test_factors = dfg.get_test_factors()
#     test_samples = []
#     for g in failure_samples.groupby(dfg.failure_modes):
#         count = g[1].shape[0]
#         fstate = tuple(zip(dfg.failure_modes, g[0]))
#         local_test_samples = []
#         for t, phi in test_factors.items():
#             phi_scope = set(phi.scope())
#             rphi = phi.copy()
#             rphi.reduce([e for e in fstate if e[0] in phi_scope])
#             tmp = rphi.sample(count)
#             tmp = pd.DataFrame(tmp, columns=[t])
#             tmp = tmp.set_index(g[1].index)
#             local_test_samples.append(tmp)
#         test_samples.append(pd.concat(local_test_samples, axis=1))
#     test_samples = pd.concat(test_samples, axis=0)
#     return pd.concat([failure_samples, test_samples], axis=1)

# def approx_sample(dfg: DiagnosticGraph, num_samples: int):
#     from diagnosability.diagnostic_factor_graph import IgnoreFactorsRules
#     import pandas as pd
#     priors = {
#         f: dfg.find_factors(
#             [f],
#             [
#                 IgnoreFactorsRules.IGNORE_BUDGET_FACTORS,
#                 IgnoreFactorsRules.IGNORE_TEST_FACTORS,
#             ],
#         ) for f in dfg.failure_modes
#     }
#     # failure_factors = dfg._get_joint_factor(
#     #     omit_rules=[
#     #         IgnoreFactorsRules.IGNORE_TEST_FACTORS,
#     #         IgnoreFactorsRules.IGNORE_ON_TESTS,
#     #     ]
#     # )
#     # failure_samples = failure_factors.sample(int(num_samples))
#     test_factors = dfg.get_test_factors()
#     test_samples = []
#     for g in failure_samples.groupby(dfg.failure_modes):
#         count = g[1].shape[0]
#         fstate = tuple(zip(dfg.failure_modes, g[0]))
#         local_test_samples = []
#         for t, phi in test_factors.items():
#             phi_scope = set(phi.scope())
#             rphi = phi.copy()
#             rphi.reduce([e for e in fstate if e[0] in phi_scope])
#             tmp = rphi.sample(count)
#             tmp = pd.DataFrame(tmp, columns=[t])
#             tmp = tmp.set_index(g[1].index)
#             local_test_samples.append(tmp)
#         test_samples.append(pd.concat(local_test_samples, axis=1))
#     test_samples = pd.concat(test_samples, axis=0)
#     return pd.concat([failure_samples, test_samples], axis=1)
