from rich.console import Console
from rich.table import Table
from rich import box
from rich import print as rprint
from diagnosability.dataset.base import TestSet, TestOutcome
import msgpack
from dacite import from_dict


class ApolloDiagnosabilityDataset:
    def __init__(self):
        self.console = Console()
        self.data = []

    @property
    def suffix(self) -> str:
        return ".dat"

    def append(self, test_results: TestSet):
        self.data.append(test_results)

    def save(self, filename: str):
        if not self.data:
            print("No data to save.")
            return
        datum = [d.to_dict() for d in self.data]
        enc = msgpack.packb(datum, use_bin_type=True)
        with open(filename, "wb") as outfile:
            outfile.write(enc)

    @classmethod
    def load(cls, filename: str) -> "ApolloDiagnosabilityDataset":
        dataset = cls()
        print(f"Loading dataset from {filename}")
        with open(filename, "rb") as data_file:
            enc = data_file.read()
        dec = msgpack.unpackb(enc, raw=False)
        dataset.data = [from_dict(data_class=TestSet, data=d) for d in dec]
        assert all(d.temporal for d in dataset.data) or all(
            not d.temporal for d in dataset.data
        ), "All samples must be temporal or all non-temporal"
        print(f"Loaded {len(dataset)} samples")
        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def printall(self):
        for test_results in self.data:
            self.print_sample(test_results)
        print(f"Printed {len(self.data)} samples")

    def _print_sample(self, test_set):
        assert not test_set.temporal
        table = Table(expand=True, show_lines=True)
        table.add_column("Timestamp", justify="left", no_wrap=True)
        table.add_column("Test Name", justify="left", no_wrap=True)
        table.add_column("Endpoints", justify="left", no_wrap=True)
        table.add_column("Test Results")
        for test_ret in test_set.test_results:
            endpoints = [f"{e.name} @ {e.timestamp}" for e in test_ret.endpoints]
            test_table = Table(
                show_footer=False,
                expand=True,
                box=box.SIMPLE,
            )
            test_table.add_column("Outcome", justify="center", no_wrap=True)
            test_table.add_column("Name", justify="left", no_wrap=True)
            test_table.add_column("Confidence", justify="center", no_wrap=True)
            test_table.add_column("Scope")
            for t in test_ret.results:
                c = "green" if t.result == TestOutcome.PASS else "red"
                test_table.add_row(
                    f"[{c}]{TestOutcome.to_string(t.result)}[/{c}]",
                    t.name,
                    f"{t.confidence*100:.1f}%" if t.confidence is not None else "N/A",
                    "- " + "\n- ".join(t.scope),
                )
            table.add_row(
                f"{test_ret.timestamp}",
                test_ret.name,
                "- " + "\n- ".join(endpoints),
                test_table,
            )
        self.console.print(table)

    def _print_temporal_sample(self, test_set):
        assert test_set.temporal
        window_size = len(test_set.test_results)
        for tau in range(window_size):
            table = Table(title=f"Instantaneous @ {tau}", expand=True, show_lines=True)
            table.add_column("Timestamp", justify="left", no_wrap=True)
            table.add_column("Test Name", justify="left", no_wrap=True)
            table.add_column("Endpoints", justify="left", no_wrap=True)
            table.add_column("Test Results")
            test_results = test_set.test_results[tau]
            for test_ret in test_results:
                endpoints = [f"{e.name} @ {e.timestamp}" for e in test_ret.endpoints]
                test_table = Table(
                    show_footer=False,
                    expand=True,
                    box=box.SIMPLE,
                )
                test_table.add_column("Outcome", justify="center", no_wrap=True)
                test_table.add_column("Name", justify="left", no_wrap=True)
                test_table.add_column("Confidence", justify="center", no_wrap=True)
                test_table.add_column("Scope")
                for t in test_ret.results:
                    c = "green" if t.result == TestOutcome.PASS else "red"
                    test_table.add_row(
                        f"[{c}]{TestOutcome.to_string(t.result)}[/{c}]",
                        t.name,
                        f"{t.confidence*100:.1f}%"
                        if t.confidence is not None
                        else "N/A",
                        "- " + "\n- ".join(t.scope),
                    )
                table.add_row(
                    f"{test_ret.timestamp}",
                    test_ret.name,
                    "- " + "\n- ".join(endpoints),
                    test_table,
                )
            self.console.print(table)
        if test_set.temporal_test_results is not None:
            table = Table(title="Temporal", expand=True, show_lines=True)
            table.add_column("Timestamp", justify="left", no_wrap=True)
            table.add_column("Test Name", justify="left", no_wrap=True)
            table.add_column("Endpoints", justify="left", no_wrap=True)
            table.add_column("Test Results")
            for test_ret in test_set.temporal_test_results:
                endpoints = [
                    f"({t}) {e.name} @ {e.timestamp}"
                    for t, z in enumerate(test_ret.endpoints)
                    for e in z
                ]
                test_table = Table(
                    show_footer=False,
                    expand=True,
                    box=box.SIMPLE,
                )
                test_table.add_column("Outcome", justify="center", no_wrap=True)
                test_table.add_column("Name", justify="left", no_wrap=True)
                test_table.add_column("Confidence", justify="center", no_wrap=True)
                test_table.add_column("Scope")
                for t in test_ret.results:
                    c = "green" if t.result == TestOutcome.PASS else "red"
                    scope = [
                        f"({t}) {name}" for t, z in enumerate(t.scope) for name in z
                    ]
                    test_table.add_row(
                        f"[{c}]{TestOutcome.to_string(t.result)}[/{c}]",
                        t.name,
                        f"{t.confidence*100:.1f}%"
                        if t.confidence is not None
                        else "N/A",
                        "- " + "\n- ".join(scope),
                    )
                table.add_row(
                    f"{test_ret.timestamp}",
                    test_ret.name,
                    "- " + "\n- ".join(endpoints),
                    test_table,
                )
            self.console.print(table)

    def print_sample(self, test_results: TestSet, clear: bool = False):
        if clear:
            self.console.clear()
        if test_results.temporal:
            self._print_temporal_sample(test_results)
        else:
            self._print_sample(test_results)
