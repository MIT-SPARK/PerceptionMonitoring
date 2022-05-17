#!/usr/bin/env python3

import argparse
from distutils.command.install import HAS_USER_SITE
from pathlib import Path

import gin
from matplotlib.style import available
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.gnn.graph_convolution import GraphConvolution
from diagnosability.gnn.datamodule import (
    FaultIdentificationDataModule,
)
from diagnosability.gnn.gnn_diagnostic_model import GNNDiagnosticModel
from tools.evaluation_utils import ModelWithResults, evaluate_on_dataset
import os
import tempfile
import pandas as pd

CHECKPOINTS_DIR = "models_checkpoints/"


def to_dataframe(inference):
    return pd.DataFrame([i.to_dict() for i in inference])


@gin.configurable("Main", denylist=["dataset_dir"])
def main(
    dataset_dir: str,
    gpus: int = 1,
    debug: bool = False,
    max_epochs: int = 20,
    weighted_syndrome: bool = False,
    unique_features:bool = False,
    ignore_modules: bool = False,
):
    assert gpus >= 0 and gpus <= torch.cuda.device_count(), "Not enough GPUs."
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
    datamodule = FaultIdentificationDataModule(
        train_dataset,
        validation_dataset,
        test_dataset,
        ignore_modules=ignore_modules,
        weighted_syndrome=weighted_syndrome,
        unique_features=unique_features,
    )

    if train_dataset.is_temporal:
        sys = train_dataset.model.temporal_systems
    else:
        sys = train_dataset.model.system

    # Init NN
    model = GraphConvolution(datamodule.num_features, datamodule.num_classes)
    if gpus < 2:
        train_accelerator = None
    else:
        print(f"Using {gpus} gpus")
        train_accelerator = "ddp"

    # Setting up trainer (& Co.)
    mlf_logger = MLFlowLogger(experiment_name=model.hparams["conv_layer"])
    mlf_logger.log_hyperparams(
        {
            "samples": len(train_dataset) + len(validation_dataset) + len(test_dataset),
            "train_num_samples": len(train_dataset),
            "validation_num_samples": len(validation_dataset),
            "test_num_samples": len(test_dataset),
            "dataset": dataset_folder,
            "batch_size": datamodule.batch_size,
            "max_epochs": max_epochs,
            "model": model.__class__.__name__,
            "ignore_modules": ignore_modules,
            "weighted_syndrome": weighted_syndrome,
            "unique_feature": unique_features,
        }
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR,
        save_top_k=1,
        verbose=False,
        monitor="val_metric",
        mode="max",
    )
    trainer = Trainer(
        gpus=gpus,
        accelerator=train_accelerator,
        max_epochs=max_epochs,
        fast_dev_run=debug,
        track_grad_norm=2,
        callbacks=[checkpoint_callback],
        logger=mlf_logger,
        weights_summary="full",
        auto_lr_find=True,
    )
    # Find learning rate
    trainer.tune(model, datamodule=datamodule)
    # Training NN
    trainer.fit(model, datamodule=datamodule)
    # Testing
    # if not debug:
    # trainer.test(datamodule=datamodule)

    ## GNN-based Diagnostic Model
    gnn = GNNDiagnosticModel(datamodule.adapter, model)
    train_results = gnn.batch_fault_identification(train_dataset)
    test_results = gnn.batch_fault_identification(test_dataset)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Evaluation
    # Train dataset
    train_stats = evaluate_on_dataset(
        system=sys,
        models={"gnn": ModelWithResults(gnn, train_results.Inference)},
        dataset=train_dataset,
        ignore_modules=ignore_modules,
    )
    # Test Dataset
    test_stats = evaluate_on_dataset(
        system=sys,
        models={"gnn": ModelWithResults(gnn, test_results.Inference)},
        dataset=test_dataset,
        ignore_modules=ignore_modules,
    )

    # Saving artifacts
    # fmt: off
    if not debug:
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, checkpoint_callback.best_model_path)
        tmp = tempfile.TemporaryDirectory()
        train_inference_csv = os.path.join(tmp.name, f"train_inference.csv")
        test_inference_csv = os.path.join(tmp.name, f"test_inference.csv")
        train_timing_csv = os.path.join(tmp.name, f"train_timing.csv")
        test_timing_csv = os.path.join(tmp.name, f"test_timing.csv")
        # Save train inference/timing results
        to_dataframe(train_results.Inference).to_csv(train_inference_csv, index=False)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, train_inference_csv)
        pd.DataFrame(train_results.Timing).to_csv(train_timing_csv, index=False)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, train_timing_csv)
        # Save test inference/timing results
        to_dataframe(test_results.Inference).to_csv(test_inference_csv, index=False)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, test_inference_csv)
        pd.DataFrame(test_results.Timing).to_csv(test_timing_csv, index=False)
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, test_timing_csv)
        # Save train & test stats
        mlf_logger.experiment.log_dict(mlf_logger.run_id, train_stats, "train_stats.json")
        mlf_logger.experiment.log_dict(mlf_logger.run_id, test_stats, "test_stats.json")
        tmp.cleanup()
    # fmt: on
    mlf_logger.finalize(status="FINISHED")


## Example:
# poetry run python3 train.py -c configs/deep_gcn_v1.gin -d /home/antonap/sparklab/apollo-master/data/bag/cyclist_2021-12-01-11-05-17_failed
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

    # gin.parse_config_file("configs/train_gcn.gin")
    # main(dataset_dir="/home/antonap/sparklab/dataset/aij/temporal")
