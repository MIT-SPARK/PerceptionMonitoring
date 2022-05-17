import pytest
from diagnosability.examples import example_system_dummy


def test_valid_query():
    sys = example_system_dummy()
    # fmt: off
    queries = [
        ("module 1", sys["module 1"]),
        ("module 1 -> *", list(sys["module 1"].outputs)),
        ("module 1 . *", list(sys["module 1"].failure_modes)),
        ("module 1 . failure m1", sys["module 1"].failure_modes["failure m1"]),
        ("module 1 -> output 1", sys["module 1"].outputs["output 1"]),
        ("module 1 -> output 1 . failure o1", sys["module 1"].outputs["output 1"].failure_modes["failure o1"]),
        ("module 1 -> output 1 . *", list(sys["module 1"].outputs["output 1"].failure_modes)),
    ]
    # fmt: on
    for query in queries:
        q = sys.query(query[0])
        assert q == query[1]


def test_invalid_query():
    sys = example_system_dummy()
    # fmt: off
    queries = [
        "module 6",
        "module 6 -> *",
        "module 5 . *",
        "module 1 . failure x",
        "module 1 -> output 6",
        "module 1 -> output 1 . failure xx",
        "module 1 -> output 3 . *",
    ]
    # fmt: on
    for query in queries:
        with pytest.raises(KeyError):
            sys.query(query)
