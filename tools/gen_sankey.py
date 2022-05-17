#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient

from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.perception_system import System
import pandas as pd
from tqdm import tqdm

tui_cols, _ = os.get_terminal_size(0)


def get_results_file(run_id):
    """
    Return a list of all artifacts for a given run.
    """
    date_fmt = "\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
    results_fmt = re.compile(f"dfg_test_results_{date_fmt}.csv")
    # dfg_fmt = re.compile(f"dfg_{date_fmt}.pkl")
    client = MlflowClient()
    run = client.get_run(run_id)
    dataset_folder = Path(run.data.params["dataset"]) / "test.pkl"
    artifacts_uri = urlparse(run.info.artifact_uri)
    # dfg_filepah = None
    results_filepath = None
    for artifact in client.list_artifacts(run_id):
        if results_fmt.match(artifact.path):
            results_filepath = Path(artifacts_uri.path, artifact.path)
        # elif dfg_fmt.match(artifact.path):
        # dfg_filepah = Path(artifacts_uri.path, artifact.path)
    return dataset_folder, results_filepath


def main(
    run_id: str,
    filename: str,
):
    dataset_filepath, results_filepath = get_results_file(run_id)
    # dfg = DiagnosticFactorGraph.load(dfg_filepath)
    dataset = DiagnosabilityDataset.load(dataset_filepath)
    results = pd.read_csv(results_filepath)
    model = dataset.model
    fm_output = sorted(
        [f.varname for f in model.system.get_failure_modes(System.Filter.OUTPUT_ONLY)]
    )
    sankey = []
    sankey = open(filename, "w")

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        assert sample.ground_truth is not None, "The dataset must have ground truth"
        y_hat = {v: results.iloc[i][v] for v in fm_output}
        y = {v: sample.ground_truth[v].value for v in fm_output}
        for v in fm_output:
            var = model.sys_query_by_varname(v)
            query = model.sys_rev_query(var)
            name = query
            if y[v] == 1 and y_hat[v] == 0:
                # print(f"{v} is a false negative")
                # sankey.append("FN", v)
                sankey.write(f"FN,{name}\n")
            elif y[v] == 0 and y_hat[v] == 1:
                # print(f"{v} is a false positive")
                # sankey.append("FP", v)
                sankey.write(f"FP,{name}\n")
            elif y[v] == 1 and y_hat[v] == 1:
                # print(f"{v} is a true positive")
                # sankey.append("TP", v)
                sankey.write(f"TP,{name}\n")
            elif y[v] == 0 and y_hat[v] == 0:
                # print(f"{v} is a true negative")
                # sankey.append("TN", v)
                sankey.write(f"TN,{name}\n")
    sankey.close()


## Example:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "id",
        action="store",
        type=str,
        help="Run uuid.",
    )
    parser.add_argument(
        "filename",
        action="store",
        type=str,
        default="sankey.txt",
        help="Output filename.",
    )
    args = parser.parse_args()

    main(
        run_id=args.id,
        filename=args.filename,
    )
#    main(
        # run_id="77a18cd6c57645ff9762ad82fce8f0cb",
        # filename="/home/antonap/sparklab/sankey.txt",
    # )
