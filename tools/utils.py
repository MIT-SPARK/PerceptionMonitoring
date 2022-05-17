import re
from pathlib import Path
from urllib.parse import urlparse

from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.gnn.datamodule import DiagnosticModelAdapter
from diagnosability.gnn.gnn_diagnostic_model import GNNDiagnosticModel
from diagnosability.gnn.graph_convolution import GraphConvolution
from diagnosability.percival_adapter import Percival
from mlflow.tracking import MlflowClient

def artifact(run_id, artifact_name, client=None):
    if client is None:
        client = MlflowClient()
    run = client.get_run(run_id)
    artifacts_uri = urlparse(run.info.artifact_uri)
    basepath = Path(artifacts_uri.path)
    for artifact in client.list_artifacts(run_id):
        if artifact.path == artifact_name:
            return basepath / artifact.path
    return None

def load_model(run_id, client=None):
    """
    Return a model for a given run.
    """
    if client is None:
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
    return None, dataset.model
