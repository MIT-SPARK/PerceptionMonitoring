#!/usr/bin/env python3

import argparse
from operator import index
import os, time
import tempfile
from pathlib import Path

import gin
from pytorch_lightning.loggers import MLFlowLogger

from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from tools.evaluation_utils import evaluate_on_dataset, ModelWithResults

# from diagnosability.gnn import *
# from diagnosability.gnn.datamodule import FaultIdentificationDataModule
from diagnosability.percival_adapter import Percival
from diagnosability.deterministic_diagnostic_graph import DeterministicDiagnosticGraph

CHECKPOINTS_DIR = "models_checkpoints/"


@gin.configurable("Main", denylist=["dataset_dir"])
def main(
    dataset_dir: str,
    debug: bool = False,
    ignore_modules: bool = False,
    cd_gibbs_sweeps: int = 1000,
    cd_batch_size: int = 0,
    cd_stepsize: float = 0.01,
    cd_steps: int = 30,
    ml_max_iterations: int = 100,
    ml_tolerance: float = 1e-8,
    ssvm_tolerance: float = 1.0e-8,
    ssvm_regularization: float = 1,
    ssvm_opt_method: str = "bmrm",
    ssvm_iterations: int = 100,
    dfg_randomize: bool = False,
):
    dataset_folder = Path(dataset_dir)
    train_dataset_path = dataset_folder.joinpath("train.pkl")
    validation_dataset_path = dataset_folder.joinpath("validation.pkl")
    test_dataset_path = dataset_folder.joinpath("test.pkl")
    strtime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Loading dataset
    print(f"Loading train dataset `{train_dataset_path}`...")
    train_dataset = DiagnosabilityDataset.load(train_dataset_path)
    print(f"Loading validation dataset `{validation_dataset_path}`...")
    validation_dataset = DiagnosabilityDataset.load(validation_dataset_path)
    print(f"Loading test dataset `{test_dataset_path}`...")
    test_dataset = DiagnosabilityDataset.load(test_dataset_path)
    # datamodule = FaultIdentificationDataModule(
    #     train_dataset, validation_dataset, test_dataset, ignore_modules=ignore_modules
    # )

    if train_dataset.is_temporal:
        sys = train_dataset.model.temporal_systems
    else:
        sys = train_dataset.model.system

    # Setting up trainer (& Co.)
    mlf_logger = MLFlowLogger(experiment_name="Factor Graph")
    mlf_logger.log_hyperparams(
        {
            "samples": len(train_dataset) + len(validation_dataset) + len(test_dataset),
            "train_num_samples": len(train_dataset),
            "validation_num_samples": len(validation_dataset),
            "test_num_samples": len(test_dataset),
            "dataset": dataset_folder,
            "cd_gibbs_sweeps": cd_gibbs_sweeps,
            "cd_batch_size": cd_batch_size,
            "cd_stepsize": cd_stepsize,
            "cd_steps": cd_steps,
            "ml_max_iterations": ml_max_iterations,
            "ml_tolerance": ml_tolerance,
            "ssvm_tolerance": ssvm_tolerance,
            "ssvm_regularization": ssvm_regularization,
            "ssvm_opt_method": ssvm_opt_method,
            "ssvm_iterations": ssvm_iterations,
            "dfg_randomize": dfg_randomize,
            "ignore_modules": ignore_modules,
        }
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## Diagnostic Factor Graph (trained on training dataset)
    dfg = Percival(train_dataset.model, include_modules=not ignore_modules)
    if not debug:
        dfg_train_results, dfg_test_results = dfg.train(
            train_dataset,
            test_dataset=test_dataset,
            cd_gibbs_sweeps=cd_gibbs_sweeps,
            cd_batch_size=cd_batch_size,
            cd_stepsize=cd_stepsize,
            cd_iterations=cd_steps,
            ml_max_iterations=ml_max_iterations,
            ml_tolerance=ml_tolerance,
            ssvm_tolerance=ssvm_tolerance,
            ssvm_regularization=ssvm_regularization,
            ssvm_opt_method=ssvm_opt_method,
            ssvm_iterations=ssvm_iterations,
            randomize=dfg_randomize,
        )

    ## Deterministic Diagnostic Graph
    # ddg = DeterministicDiagnosticGraph(
    #     sys, train_dataset.model, ignore_modules=ignore_modules
    # )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Evaluation

    # Train dataset
    train_stats = evaluate_on_dataset(
        system=sys,
        models={"dfg": ModelWithResults(dfg, dfg_train_results.Inference)},
        dataset=train_dataset,
        ignore_modules=ignore_modules,
    )
    # Test Dataset
    test_stats = evaluate_on_dataset(
        system=sys,
        models={"dfg": ModelWithResults(dfg, dfg_test_results.Inference)},
        dataset=test_dataset,
        ignore_modules=ignore_modules,
    )

    # Saving artifacts
    # fmt: off
    if not debug:
        tmp = tempfile.TemporaryDirectory()
        dfg_model_path = os.path.join(tmp.name, f"trained_model.json")
        dfg_model_params_path = os.path.join(tmp.name, f"checkpoint.json")
        dfg_train_inference_csv = os.path.join(tmp.name, f"train_inference.csv")
        dfg_train_timing_csv = os.path.join(tmp.name, f"train_timing.csv")
        dfg_test_inference_csv = os.path.join(tmp.name, f"test_inference.csv")
        dfg_test_timing_csv = os.path.join(tmp.name, f"test_timing.csv")
        # Save model
        dfg.save_model(dfg_model_path)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, dfg_model_path)
        dfg.save_params(dfg_model_params_path)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, dfg_model_params_path)
        # Save train inference results
        dfg_train_results.Inference.to_csv(dfg_train_inference_csv, index=False)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, dfg_train_inference_csv)
        # Save train timing results
        dfg_train_results.Timing.to_csv(dfg_train_timing_csv, index=False)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, dfg_train_timing_csv)
        # Save test inference results
        dfg_test_results.Inference.to_csv(dfg_test_inference_csv, index=False)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, dfg_test_inference_csv)
        # Save test timing results
        dfg_test_results.Timing.to_csv(dfg_test_timing_csv, index=False)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, dfg_test_timing_csv)
    # Save train & test stats
    mlf_logger.experiment.log_dict(mlf_logger.run_id, train_stats, "train_stats.json")
    mlf_logger.experiment.log_dict(mlf_logger.run_id, test_stats, "test_stats.json")
    # fmt: on

    mlf_logger.finalize(status="FINISHED")


## Example:
# poetry run python3 tools/train_factor_graph.py -c configs/train_factor_graph.gin -d /home/antonap/sparklab/dataset/simple
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="store",
        type=str,
        required=True,
        help="Path to dataset folder.",
    )
    args = parser.parse_args()

    gin.parse_config_file(args.config)
    main(dataset_dir=args.dataset)

    # gin.parse_config_file("configs/train_factor_graph.gin")
    # main(dataset_dir="/home/antonap/sparklab/dataset/temporal")
