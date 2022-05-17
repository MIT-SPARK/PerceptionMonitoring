#!/usr/bin/env python3

import argparse
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
from pathlib import Path
import pprint
import pandas as pd


def load_artifact(client, label, run_id, filename):
    run = client.get_run(run_id)
    artifacts_uri = urlparse(run.info.artifact_uri)
    stats = None
    for artifact in client.list_artifacts(run_id):
        if artifact.path == filename:
            with open(Path(artifacts_uri.path, artifact.path)) as f:
                return pd.read_csv(f, skiprows=1, names=[label])
    raise ValueError(f"No artifact called {filename} found in run {run_id}")


def main(run_ids, filename, output_dir):
    client = MlflowClient()
    # Regular
    timing = [
        load_artifact(client, label, run_id, filename)
        for label, run_id in run_ids["regular"].items()
    ]
    result = pd.concat(timing, axis=1)
    result.to_csv(output_dir / "timing_regular.csv", index=False)
    # Temporal
    timing = [
        load_artifact(client, label, run_id, filename)
        for label, run_id in run_ids["temporal"].items()
    ]
    result = pd.concat(timing, axis=1)
    result.to_csv(output_dir / "timing_temporal.csv", index=False)


if __name__ == "__main__":
    # fmt: off
    config = {
        "regular": {
            "Factor Graph": "fbf7ea65faf04a89a6eab1fa55ecc567",
            "Deterministic": "87fa09c3f1ef4948917c8305094cf978",
            "Baseline": "8e6a42f9bd1f49629fda62c2ac0ceb44",
            "GCN": "fe8ee59b3944468fb2ff1dd7bf6b8fff",
            "GCNII": "5d92465f8f8744db8bd8e7261ec0454d",
            "GIN": "76ab25dfb1254a6bbae355ca931d0472",
            "GraphSAGE": "5e11b91ad7ea468caf450c8fbb7757fe",
        },
        "temporal": {
            "Factor Graph": "e45f59f8e9e64f31b0fbb46439046c17",
            "Deterministic": "7a437b313c01490d8d4ba1fe0c170fbf",
            "Baseline": "1c3d24061ff346f396d47b243dd5fb6a",
            "GCN": "a2d6ae11651b49c8a0779ae3b06d5f92",
            "GCNII": "194f0dd19619479eabee1d0c5a7460f3",
            "GIN": "ad7e17d924914c5c9f849fb17753ffed",
            "GraphSAGE": "61eb4881809a4eb382dc03f5243fc233",
        },
    }
    # fmt: on
    main(
        run_ids=config,
        filename="test_timing.csv",
        output_dir=Path("temporary"),
    )
