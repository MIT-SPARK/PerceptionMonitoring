from pathlib import Path
from urllib.parse import urlparse
import json
from mlflow.tracking import MlflowClient
import re

def load_artifact(client, run_id, filename, skiprows=0):
    run = client.get_run(run_id)
    artifacts_uri = urlparse(run.info.artifact_uri)
    ret = None
    for artifact in client.list_artifacts(run_id):
        if artifact.path == filename:
            if ".json" in filename:
                with open(Path(artifacts_uri.path, artifact.path)) as f:
                    ret = json.load(f)
                    break
            else:
                ret = Path(artifacts_uri.path, artifact.path)
                break
    return ret

def get_run_params(run_id):
    """
    Return a list of all artifacts for a given run.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    
    return {"params": run.data.params}