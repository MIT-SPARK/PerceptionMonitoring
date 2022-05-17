#!/usr/bin/env python3

import argparse
from pathlib import Path

from pytorch_lightning.loggers import MLFlowLogger

from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.deterministic_diagnostic_graph import DeterministicDiagnosticGraph
from tools.evaluation_utils import ModelWithResults, evaluate_on_dataset
import tempfile
import os
import pandas as pd

def to_dataframe(inference):
    return pd.DataFrame([i.to_dict() for i in inference])

def main(
    dataset_dir: str,
    ignore_modules: bool = False,
):
    dataset_folder = Path(dataset_dir)
    train_dataset_path = dataset_folder.joinpath("train.pkl")
    validation_dataset_path = dataset_folder.joinpath("validation.pkl")
    test_dataset_path = dataset_folder.joinpath("test.pkl")

    # Loading dataset
    print(f"Loading train dataset `{train_dataset_path}`...")
    train_dataset = DiagnosabilityDataset.load(train_dataset_path)
    print(f"Loading validation dataset `{validation_dataset_path}`...")
    validation_dataset = DiagnosabilityDataset.load(validation_dataset_path)
    print(f"Loading test dataset `{test_dataset_path}`...")
    test_dataset = DiagnosabilityDataset.load(test_dataset_path)

    if train_dataset.is_temporal:
        sys = train_dataset.model.temporal_systems
    else:
        sys = train_dataset.model.system

    mlf_logger = MLFlowLogger(experiment_name="Deterministic")
    mlf_logger.log_hyperparams(
        {
            "samples": len(train_dataset) + len(validation_dataset) + len(test_dataset),
            "train_num_samples": len(train_dataset),
            "validation_num_samples": len(validation_dataset),
            "test_num_samples": len(test_dataset),
            "dataset": dataset_folder,
            "ignore_modules": ignore_modules,
        }
    )

    # Baseline
    ddg = DeterministicDiagnosticGraph(
        sys, train_dataset.model, ignore_modules=ignore_modules
    )
    train_results = ddg.batch_fault_identification(train_dataset)
    test_results = ddg.batch_fault_identification(test_dataset)
    validation_results = ddg.batch_fault_identification(validation_dataset)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Evaluation
    train_stats = evaluate_on_dataset(
        system=sys,
        models={"deterministic": ModelWithResults(ddg, train_results.Inference)},
        dataset=train_dataset,
        ignore_modules=ignore_modules,
    )
    test_stats = evaluate_on_dataset(
        system=sys,
        models={"deterministic": ModelWithResults(ddg, test_results.Inference)},
        dataset=test_dataset,
        ignore_modules=ignore_modules,
    )
    validation_stats = evaluate_on_dataset(
        system=sys,
        models={"deterministic": ModelWithResults(ddg, validation_results.Inference)},
        dataset=validation_dataset,
        ignore_modules=ignore_modules,
    )

    # Saving artifacts
    # fmt: off
    tmp = tempfile.TemporaryDirectory()
    train_inference_csv = os.path.join(tmp.name, f"train_inference.csv")
    test_inference_csv = os.path.join(tmp.name, f"test_inference.csv")
    validation_inference_csv = os.path.join(tmp.name, f"validation_inference.csv")
    train_timing_csv = os.path.join(tmp.name, f"train_timing.csv")
    test_timing_csv = os.path.join(tmp.name, f"test_timing.csv")
    validation_timing_csv = os.path.join(tmp.name, f"validation_timing.csv")
    # Save train 
    to_dataframe(train_results.Inference).to_csv(train_inference_csv, index=False)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, train_inference_csv)
    pd.DataFrame(train_results.Timing).to_csv(train_timing_csv, index=False)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, train_timing_csv)
    # Save test timing results
    to_dataframe(test_results.Inference).to_csv(test_inference_csv, index=False)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, test_inference_csv)
    pd.DataFrame(test_results.Timing).to_csv(test_timing_csv, index=False)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, test_timing_csv)
    # Save validation timing results
    to_dataframe(validation_results.Inference).to_csv(validation_inference_csv, index=False)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, validation_inference_csv)
    pd.DataFrame(validation_results.Timing).to_csv(validation_timing_csv, index=False)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, validation_timing_csv)
    # Save stats
    mlf_logger.experiment.log_dict(mlf_logger.run_id, train_stats, "train_stats.json")
    mlf_logger.experiment.log_dict(mlf_logger.run_id, test_stats, "test_stats.json")
    mlf_logger.experiment.log_dict(mlf_logger.run_id, validation_stats, "validation_stats.json")
    # fmt: on

    mlf_logger.finalize(status="FINISHED")


## Example:
# poetry run python3 deterministic.py  -d /home/antonap/sparklab/dataset/simple
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        action="store",
        type=str,
        required=True,
        help="Path to dataset folder.",
    )
    args = parser.parse_args()

    main(dataset_dir=args.dataset)
