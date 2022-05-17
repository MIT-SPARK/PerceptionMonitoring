from sre_constants import FAILURE
import click
from diagnosability.dataset.diagnosability_dataset import (
    DiagnosabilityDataset,
    DiagnosabilityDatasetPreprocessor,
)
from pathlib import Path
import os
from enum import Enum


class OpResult:
    class Type(Enum):
        SUCCESS = 0
        FAILURE = 1
        SKIP = 2

    def __init__(self, type: Type, message: str = None):
        self.type = type
        self.message = message


DEFAULT_PROCESSED_FILENAME = "dataset.zip"


def _preprocess(dataset, config, input, force):
    if isinstance(dataset, Path):
        dataset_folder = dataset
    else:
        dataset_folder = Path(dataset)
    if not dataset_folder.is_dir():
        return OpResult(OpResult.Type.FAILURE, "Dataset path is not a directory.")
    dataset_path = dataset_folder / input
    if not dataset_path.is_file():
        return OpResult(OpResult.Type.FAILURE, f"Log file `{input}` does not exist.")
    output_filepath = dataset_folder / DEFAULT_PROCESSED_FILENAME
    if output_filepath.is_file():
        if force:
            output_filepath.unlink()
        else:
            return OpResult(
                OpResult.Type.SKIP, f"{DEFAULT_PROCESSED_FILENAME} already exists."
            )
    config_path = Path(config)
    if not config_path.is_file():
        return OpResult(OpResult.Type.FAILURE, "Config file does not exist.")
    dataset = DiagnosabilityDatasetPreprocessor(dataset_path, config_path)
    dataset.bake()
    outfile = dataset.export(output_filepath)
    return OpResult(OpResult.Type.SUCCESS, f"Dataset exported to {outfile}.")


def _merge(datasets, output, force, train, validation, test):
    print(f"Merging {len(datasets)} datasets.")
    assert (
        train + validation + test == 1.0
    ), "Train, validation and test ratios must sum to 1.0"
    out_folder = Path(output)
    if not out_folder.is_dir():
        print(f"Creating output folder {out_folder}")
        os.makedirs(out_folder)

    train_filepath = out_folder.joinpath("train.pkl")
    val_filepath = out_folder.joinpath("validation.pkl")
    test_filepath = out_folder.joinpath("test.pkl")
    if not force:
        if train > 0.0 and train_filepath.is_file():
            return OpResult(OpResult.Type.FAILURE, "train.pkl already exists.")
        if validation > 0.0 and val_filepath.is_file():
            return OpResult(OpResult.Type.FAILURE, "validation.pkl already exists.")
        if test > 0.0 and test_filepath.is_file():
            return OpResult(OpResult.Type.FAILURE, "train.pkl already exists.")

    dataset = DiagnosabilityDataset()
    for dataset_filepath in datasets:
        dataset_path = Path(dataset_filepath)
        if not dataset_path.is_file():
            return OpResult(
                OpResult.Type.FAILURE, f"Dataset file does not exist: {dataset_path}"
            )
        print(f"Loading {dataset_path}")
        dataset.append(dataset_filepath)
    print(f"Num. samples: {len(dataset)}")
    print(f"Splitting into ({train}, {validation}, {test})")
    train_d, val_d, test_d = dataset.split((train, validation, test))

    if len(train_d) > 0:
        print(f"Train: {len(train_d)} -> {train_filepath}")
        train_d.save(train_filepath)
    if len(val_d) > 0:
        print(f"Validation: {len(val_d)} -> {val_filepath}")
        val_d.save(val_filepath)
    if len(test_d) > 0:
        print(f"Test: {len(test_d)} -> {test_filepath}")
        test_d.save(test_filepath)

    return OpResult(OpResult.Type.SUCCESS, "Dataset merged.")


@click.group()
def cli():
    pass


# @cli.command()
# @click.option("-d", "--dataset", type=str, required=True, help="Dataset folder")
# @click.option("-c", "--config", type=str, required=True, help="Config file")
# @click.option("-f", "--force", is_flag=True, help="Force overwrite")
# def preprocess(dataset, config, force):
#     _preprocess(dataset, config, force)


@cli.command()
# fmt: off
@click.option("-d", "--datasets", multiple=True, type=str, required=True, help="Dataset files")
@click.option("-o", "--output", type=str, required=True, help="Output folder")
@click.option("-f", "--force", is_flag=True, help="Force overwrite")
@click.option("--train", type=float, required=False, default=0.8, help="Train ratio")
@click.option("--validation", type=float, required=False, default=0.1, help="Validation ratio")
@click.option("--test", type=float, required=False, default=0.1, help="Test ratio")
# fmt: on
def merge(datasets, output, force, train, validation, test):
    _merge(datasets, output, force, train, validation, test)


@cli.command()
# fmt: off
@click.option("-d", "--dir", type=str, required=True, help="Folder containing datasets")
@click.option("-c", "--config", type=str, required=True, help="Config file")
@click.option("-i", "--input", type=str, required=True, help="Log filename")
@click.option("-o", "--output", type=str, required=True, help="Output folder")
@click.option("-f", "--force", is_flag=True, default=False, help="Force reprocess")
@click.option("--train", type=float, required=False, default=0.8, help="Train ratio")
@click.option("--validation", type=float, required=False, default=0.1, help="Validation ratio")
@click.option("--test", type=float, required=False, default=0.1, help="Test ratio")
# fmt: on
def process(dir, config, input, output, force, train, validation, test):
    print(f"Process datasets {dir}.")
    assert (
        train + validation + test == 1.0
    ), "Train, validation and test ratios must sum to 1.0"
    folder = Path(dir)
    if not folder.is_dir():
        print("Error, datasets folder does not exist.")
        exit(1)
    dd = []
    for d in folder.iterdir():
        if d.is_dir():
            print(f"Preprocessing {d}")
            r = _preprocess(d, config, input, force)
            if r.type == OpResult.Type.FAILURE:
                print(f"💥 {r.message}")
            else:
                print(f"✅ {r.message}")
                dd.append(d / DEFAULT_PROCESSED_FILENAME)
        print("")
    if dd:
        r = _merge(dd, output, force, train, validation, test)
        if r.type == OpResult.Type.FAILURE:
            print(f"💥 {r.message}")
        else:
            print(f"✅ {r.message}")
    else:
        print("No datasets to merge.")


if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    # print("Waiting for debugger attach")
    # ptvsd.wait_for_attach()

    cli()
