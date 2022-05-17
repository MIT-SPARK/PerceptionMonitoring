#!/usr/bin/env python3

import argparse
from unittest import result
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix as compute_confusion_matrix
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from tqdm import tqdm
from rich import print
from diagnosability.evaluator import Stats


TEMPORAL_EVALUATION_LAST_ONLY = True


def get_run(run_id, dataset_file, results_filename):
    client = MlflowClient()
    run = client.get_run(run_id)
    artifacts_uri = urlparse(run.info.artifact_uri)
    basepath = Path(artifacts_uri.path)
    dataset_folder = Path(run.data.params["dataset"])
    dataset_path = dataset_folder / dataset_file
    dataset = DiagnosabilityDataset.load(dataset_path)
    inference = None
    for artifact in client.list_artifacts(run_id):
        if artifact.path == results_filename:
            with open(basepath / artifact.path) as f:
                inference = pd.read_csv(f)
                break
    return dataset, inference


def main(run_id, dataset_filename, results_filename):
    dataset, results = get_run(run_id, dataset_filename, results_filename)
    assert result is not None, "No results found"
    mdl = dataset.model
    if dataset.is_temporal and TEMPORAL_EVALUATION_LAST_ONLY:
        vars = [f.varname for f in mdl.temporal_systems[-1].get_failure_modes()]
    else:
        vars = mdl.failure_modes
    gt, pred = [], []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        # Did the system experience at least one failure?
        gt.append(int(any(sample.ground_truth[v].value for v in vars)))
        # Did at least one test fail?
        pred.append(int(any(results.iloc[i][v] for v in vars)))
    tn, fp, fn, tp = compute_confusion_matrix(gt, pred).ravel()
    cm = {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}
    print(Stats.all(cm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "id",
        action="store",
        type=str,
        help="Run uuid.",
    )
    parser.add_argument(
        "--split", action="store", type=str, default="test", help="Split type [train|test|validation]."
    )
    args = parser.parse_args()
    main(
        run_id=args.id,
        dataset_filename=f"{args.split}.pkl",
        results_filename=f"{args.split}_inference.csv",
    )
    # main(
    #     run_id="e45f59f8e9e64f31b0fbb46439046c17", #"fbf7ea65faf04a89a6eab1fa55ecc567",
    #     dataset_filename="test.pkl",
    #     results_filename="test_inference.csv",
    # )
