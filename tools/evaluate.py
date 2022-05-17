#!/usr/bin/env python3

import argparse
import os
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
# from diagnosability.gnn import *
from rich import print
from tools.evaluation_utils import ModelWithResults, evaluate_on_dataset
from tools.utils import load_model

tui_cols, _ = os.get_terminal_size(0)

def evaluate(
    run_id: str,
    dataset_path: str,
):
    model, base_model = load_model(run_id)
    modelname = model.__class__.__name__
    print(f"Type: {modelname}")

    print(f"Loading dataset `{dataset_path}`...")
    dataset = DiagnosabilityDataset.load(dataset_path)
    print(f"Dataset loaded with {len(dataset)} samples.")
    dataset.change_model(base_model)

    results = model.batch_fault_identification(dataset)

    if dataset.is_temporal:
        sys = dataset.model.temporal_systems
    else:
        sys = dataset.model.system

    stats = evaluate_on_dataset(
        system=sys,
        models={"modelname": ModelWithResults(model, results.Inference)},
        dataset=dataset,
        ignore_modules=False,
    )
    print(stats)


## Example:
# poetry run python3 evaluate.py a4946db83e654de687819ff29370fed6
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "id",
        action="store",
        type=str,
        help="Run uuid.",
    )
    parser.add_argument(
        "filename", action="store", type=str, help="Filename."
    )
    args = parser.parse_args()

    evaluate(
        run_id=args.id,
        dataset_path=args.filename,
    )
    # main(
    #     run_id="642bf130d39b4814bffe6b2fe13af8b8",
    #     dataset_path="/home/antonap/sparklab/dataset/aij/regular/test.pkl",
    # )
