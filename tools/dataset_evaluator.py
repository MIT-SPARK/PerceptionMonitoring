from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.evaluator import (
    DiagnosticTestsEvaluator,
    FaultDetectionEvaluator,
    Stats,
)
from pathlib import Path
import argparse
from rich import print
from rich.console import Console

from diagnosability.perception_system import System


def file_path(path):
    p = Path(path)
    if p.is_file():
        return p
    else:
        raise argparse.ArgumentTypeError(f"{path} does not exist")

def average_number_of_faults(dataset, vars):
    f1 = []
    for sample in dataset:
        f1.append(sum([sample.ground_truth[v].value for v in vars]))
    return sum(f1) / len(f1)

def main():
    parser = argparse.ArgumentParser(
        description="Analize dataset and computes statistics on the tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset", type=file_path, help="Path to diagnosability dataset."
    )
    parser.add_argument(
        "--include-test-stats", action="store_true", help="Include test statistics."
    )
    args = parser.parse_args()

    console = Console()
    dataset = DiagnosabilityDataset.load(args.dataset)

    console.print("Dataset", style="bold")
    print(f"Path: {args.dataset}")
    print(f"Size: {len(dataset)}")
    print(f"Model type: {dataset.model.__class__.__name__}")
    print(f"Temporal: {dataset.is_temporal}")
    if dataset.is_temporal:
        print(f"Window size: {dataset.model.winsize}")
    print(f"Num. Failure Modes: {len(dataset.model.failure_modes)}")
    if dataset.is_temporal:
        # Temporal Graph
        modules_fm = {
            v.varname
            for sys in dataset.model.temporal_systems
            for v in sys.get_failure_modes(System.Filter.MODULE_ONLY)
        }
        outputs_fm = {
            v.varname
            for sys in dataset.model.temporal_systems
            for v in sys.get_failure_modes(System.Filter.OUTPUT_ONLY)
        }
    else:
        modules_fm = {
            v.varname
            for v in dataset.model.system.get_failure_modes(System.Filter.MODULE_ONLY)
        }
        outputs_fm = {
            v.varname
            for v in dataset.model.system.get_failure_modes(System.Filter.OUTPUT_ONLY)
        }
    print(f"  - Modules: {len(modules_fm)}")
    print(f"  - Output:  {len(outputs_fm)}")
    print(f"Num. Tests: {len(dataset.model.tests)}")
    print(f"Avg. Num. Failures: {average_number_of_faults(dataset, modules_fm|outputs_fm)}")
    print("")

    console.print("Fault detection", style="bold")
    detection_cm = FaultDetectionEvaluator.evaluate(dataset)
    print(Stats.all(detection_cm))

    if args.include_test_stats:
        console.print("\nTest statistics", style="bold")
        cm = DiagnosticTestsEvaluator.evaluate(dataset)
        for t in cm:
            print(t)
            print(Stats.all(cm[t]))
            print("")


if __name__ == "__main__":
    main()
