#!/usr/bin/env python3

import argparse
import os
import pprint
import re
from pathlib import Path
from urllib.parse import urlparse
from black import out
from mlflow.tracking import MlflowClient

from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset, DiagnosabilityDatasetPreprocessor
from diagnosability.gnn import *
from diagnosability.gnn.datamodule import DiagnosticModelAdapter
from diagnosability.gnn.gnn_diagnostic_model import GNNDiagnosticModel
from diagnosability.percival_adapter import Percival
from diagnosability.gnn.graph_convolution import GraphConvolution
from rich import print

def get_run_info(run_id):
    """
    Return a list of all artifacts for a given run.
    """
    date_fmt = "\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    mle_fmt = re.compile(f"dfg_{date_fmt}.pkl")
    client = MlflowClient()
    run = client.get_run(run_id)
    artifacts_uri = urlparse(run.info.artifact_uri)
    artifacts = dict()
    for artifact in client.list_artifacts(run_id):
        if mle_fmt.match(artifact.path):
            artifacts["dfg"] = Path(artifacts_uri.path, artifact.path)
        elif artifact.path.endswith(".ckpt"):
            artifacts["gnn"] = Path(artifacts_uri.path, artifact.path)
    artifacts["net"] = globals()[run.data.params["model"]]
    return {"params": run.data.params, "artifacts": artifacts}


def get_model(run_id):
    """
    Return a model for a given run.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    artifacts_uri = urlparse(run.info.artifact_uri)
    basepath = Path(artifacts_uri.path)
    dataset_folder = Path(run.data.params["dataset"])
    dataset_path = dataset_folder / "train.pkl"
    dataset = DiagnosabilityDataset.load(dataset_path)
    ignore_modules = bool(run.data.params["ignore_modules"].lower() == "true")
    for artifact in client.list_artifacts(run_id):
        if artifact.path.endswith(".ckpt"):
            gnn = GraphConvolution.load_from_checkpoint(basepath / artifact.path)
            adapter = DiagnosticModelAdapter(
                dataset.model, ignore_modules=ignore_modules
            )
            return GNNDiagnosticModel(adapter, gnn), dataset.model
        elif artifact.path == "checkpoint.json":
            dfg = Percival(dataset.model, include_modules=not ignore_modules)
            dfg.load_params(basepath / artifact.path)
            return dfg, dataset.model


def postprocess(
    run_id: str,
    config : Path,
    dataset_path: Path,
    outfile: Path,
    force: bool=False,
    verbose=False,
):
    def _print(msg):
        if verbose:
            print(msg)
    if not dataset_path.is_file():
        _print(f"{dataset_path} does not exist.")
        return False
    if outfile.is_file():
        if force:
            outfile.unlink()
        else:
            print(f"{outfile} already exists.")
            return False
    
    model, base_model = get_model(run_id)
    modelname = model.__class__.__name__
    _print(f"Type: {modelname}")

    _print(f"Loading dataset `{dataset_path}`...")
    preprocessor = DiagnosabilityDatasetPreprocessor(dataset_path, config)
    dataset = preprocessor.bake().to_diagnosability_dataset()
    dataset.change_model(base_model)
    _print(f"Dataset loaded with {len(dataset)} samples.")

    _print(f"Running inference...")
    results = model.batch_fault_identification(dataset)
    dataset.set_fault_identification_results(results.Inference)
    _print(f"Done.")

    _print(f"Eporting results to `{outfile}`...")
    dataset.export(outfile)
    _print(f"Completed.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        action="store",
        type=str,
        help="Run uuid.",
    )
    parser.add_argument("--config", action="store", type=Path, help="Config file.")
    parser.add_argument(
        "--tests", action="store", type=Path, help="File to test results."
    )
    parser.add_argument(
        "--output", action="store", type=Path, help="File to export results."
    )
    parser.add_argument('--force', help='Override', action='store_true')
    args = parser.parse_args()

    postprocess(
        run_id=args.id,
        config=args.config,
        dataset_path=args.tests,
        outfile=args.output,
        force=args.force,
        verbose=True,
    )
    # main(
    #     run_id="384749f86d0d4127a051d1221d10ff2f",
    #     config="/home/antonap/sparklab/diagnosability/configs/dfg_obstacle_detection_without_mismatch_type.yaml",
    #     dataset_path="/home/antonap/sparklab/apollo-master/data/bag/stopped_vehicle_curved_2022-03-06-22-12-41_ok/log_regular_adaptive.dat",
    #     outfile="/home/antonap/sparklab/apollo-master/data/bag/stopped_vehicle_curved_2022-03-06-22-12-41_ok/inference.zip",
    # )
