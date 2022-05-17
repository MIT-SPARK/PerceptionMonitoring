from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph
from diagnosability.factors import TestFactor
from diagnosability.temporal_diagnostic_graph import TemporalDiagnosticFactorGraph


def remap_diagnostic_graph(graph_1, graph_2):
    if isinstance(graph_1, TemporalDiagnosticFactorGraph):
        assert isinstance(graph_2, TemporalDiagnosticFactorGraph)
        assert graph_1.winsize == graph_2.winsize, "winsize mismatch"
        varname_map = dict()
        for t in range(graph_1.winsize):
            for f in graph_1.temporal_systems[t].get_failure_modes():
                q = graph_1.temporal_systems[t].rev_query(f)
                varname_map[f.varname] = graph_2.temporal_systems[t].query(q).varname
        for tname in graph_1.temporal_tests:
            for k, v in graph_1.temporal_tests[tname].items():
                varname_map[v.varname] = graph_2.temporal_tests[tname][k].varname
        # Temporal tests
        test_1 = [
            phi.test
            for phi in graph_1.get_factors(lambda x: isinstance(x, TestFactor))
            if phi.test.varname not in varname_map.keys()
        ]
        test_2 = [
            phi.test
            for phi in graph_2.get_factors(lambda x: isinstance(x, TestFactor))
            if phi.test.varname not in varname_map.values()
        ]
        assert len(test_1) == len(test_2), "Different sets of tests"
        used = set()
        for t1 in test_1:
            found = False
            for i, t2 in enumerate(test_2):
                if t1.name == t2.name and t1.timestep == t2.timestep:
                    assert i not in used, f"Duplicate test names {t1.name}"
                    varname_map[t1.varname] = t2.varname
                    used.add(i)
                    found = True
                    break
            assert found, f"Test {t1.name} not found"
        return varname_map
    elif isinstance(graph_1, DiagnosticFactorGraph):
        assert isinstance(graph_2, DiagnosticFactorGraph)
        varname_map = dict()
        # Failure Modes
        for f in graph_1.system.get_failure_modes():
            q = graph_1.system.rev_query(f)
            varname_map[f.varname] = graph_2.system.query(q).varname
        # Tests
        test_1 = graph_1.get_factors(lambda x: isinstance(x, TestFactor))
        test_2 = graph_2.get_factors(lambda x: isinstance(x, TestFactor))
        assert len(test_1) == len(test_2), "Different sets of tests"
        used = set()
        for t1 in test_1:
            found = False
            for i, t2 in enumerate(test_2):
                if t1.test.name == t2.test.name:
                    assert i not in used, "Duplicate test names"
                    varname_map[t1.test.varname] = t2.test.varname
                    used.add(i)
                    found = True
                    break
            assert found, f"Test {t1.test.name} not found"
        return varname_map
    else:
        raise RuntimeError("Unknown graph type")