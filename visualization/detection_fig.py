#!/usr/bin/env python3

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from adjustText import adjust_text
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.evaluator import Stats
from mlflow.tracking import MlflowClient
from rich import print
from sklearn.metrics import confusion_matrix as compute_confusion_matrix
from diagnosability.perception_system import System
from tools.utils import artifact, load_model

from enum import Enum


class VarType(Enum):
    ALL = 0
    MODULES = 1
    OUTPUTS = 2


def compute_stats(model_or_results, dataset_path, last_only=False, vartype=VarType):
    dataset = DiagnosabilityDataset.load(dataset_path)
    mdl = dataset.model
    if isinstance(model_or_results, pd.DataFrame):
        results = model_or_results
    else:
        results = model_or_results.batch_fault_identification(dataset)
        results = pd.DataFrame([i.to_dict() for i in results.Inference])
    if dataset.is_temporal:
        if last_only:
            if vartype == VarType.ALL:
                vars = [f.varname for f in mdl.temporal_systems[-1].get_failure_modes()]
            else:
                vars = [
                    f.varname
                    for f in mdl.temporal_systems[-1].get_failure_modes(
                        System.Filter.MODULE_ONLY
                        if vartype == VarType.MODULES
                        else System.Filter.OUTPUT_ONLY
                    )
                ]
        else:
            if vartype == VarType.ALL:
                vars = mdl.failure_modes
            else:
                raise NotImplementedError("Not implemented yet")
    else:
        if vartype == VarType.ALL:
            vars = mdl.failure_modes
        else:
            vars = [
                f.varname
                for f in mdl.system.get_failure_modes(
                    System.Filter.MODULE_ONLY
                    if vartype == VarType.MODULES
                    else System.Filter.OUTPUT_ONLY
                )
            ]
    gt, pred = [], []
    for i in range(len(dataset)):
        # Did the system experience at least one failure mode?
        gt.append(int(any(dataset[i].ground_truth[v].value for v in vars)))
        # Was at least one failure mode reported as active?
        pred.append(int(any(results.iloc[i][v] for v in vars)))
    tn, fp, fn, tp = compute_confusion_matrix(gt, pred).ravel()
    cm = {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}
    return Stats.all(cm)


def preprocess(config, dgraph_type, vartype):
    algos = config["Algorithms"]
    dataset_filename = config["Plots"]["Detection"]["dataset"]
    inference_artifact = config["Plots"]["Detection"]["inference"]
    last_only = config["Plots"]["Detection"]["last_only"]
    detection_stats = dict()
    # label_len = max(len(l) for l in algos)+1
    client = MlflowClient()
    for label, alg in algos.items():
        run_id = alg[dgraph_type]
        run = client.get_run(run_id)
        results_file = artifact(run_id, inference_artifact, client=client)
        dataset_filepath = Path(run.data.params["dataset"]) / dataset_filename
        if results_file is not None:
            inference = pd.read_csv(results_file)
            detection_stats[label] = compute_stats(
                inference, dataset_filepath, last_only=last_only, vartype=vartype
            )
        else:
            model, _ = load_model(run_id, client=client)
            detection_stats[label] = compute_stats(
                model, dataset_filepath, last_only=last_only, vartype=vartype
            )
    return detection_stats


def main(cfg):
    vartype = VarType.ALL
    algos = cfg["Algorithms"]
    marker_size = cfg["Plots"]["Detection"]["marker_size"]
    font_size = cfg["Plots"]["Detection"]["font_size"]
    stats = {
        "regular": preprocess(cfg, "regular", vartype),
        "temporal": preprocess(cfg, "temporal", vartype),
    }
    # PLOT
    with plt.style.context(["science"]):
        fig, (ax_o, ax_m) = plt.subplots(1, 2, figsize=(10, 3))
        # Regular
        xx, yy, colors, labels, markers = [], [], [], [], []
        for label, stat in stats["regular"].items():
            xx.append(stat["precision"] * 100)
            yy.append(stat["recall"] * 100)
            colors.append(mcolors.to_rgba_array([algos[label]["color"]]))
            labels.append(label)
            markers.append(algos[label]["marker"])
            ax_m.scatter(
                stat["precision"] * 100,
                stat["recall"] * 100,
                c=mcolors.to_rgba_array([algos[label]["color"]]),
                marker=algos[label]["marker"],
                s=marker_size,
            )
        # ax_m.scatter(xx, yy, c=colors, marker="o", s=marker_size)
        ax_m.set_xlabel("Precision (\%)")
        ax_m.set_ylabel("Recall (\%)")
        ax_m.set_title("Regular")
        reg_text = [
            ax_m.text(x, y, label, fontsize=font_size)
            for label, x, y in zip(labels, xx, yy)
        ]
        # Temporal
        xx, yy, colors, labels, markers = [], [], [], [], []
        for label, stat in stats["temporal"].items():
            xx.append(stat["precision"] * 100)
            yy.append(stat["recall"] * 100)
            colors.append(mcolors.to_rgba_array([algos[label]["color"]]))
            labels.append(label)
            markers.append(algos[label]["marker"])
            ax_o.scatter(
                stat["precision"] * 100,
                stat["recall"] * 100,
                c=mcolors.to_rgba_array([algos[label]["color"]]),
                marker=algos[label]["marker"],
                s=marker_size,
            )
        # ax_o.scatter(xx, yy, c=colors, marker="o", s=marker_size)
        ax_o.set_xlabel("Precision (\%)")
        ax_o.set_ylabel("Recall (\%)")
        ax_o.set_title("Temporal")
        tem_text = [
            ax_o.text(x, y, label, fontsize=font_size)
            for label, x, y in zip(labels, xx, yy)
        ]
    adjust_text(
        reg_text,
        lim=int(1e4),
        expand_text=(2, 2),
        expand_points=(2, 2),
        force_text=(0.01, 0.25),
        force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle="-", lw=1),
        ax=ax_m,
    )
    adjust_text(
        tem_text,
        lim=int(5e3),
        expand_text=(2, 2),
        expand_points=(2, 2),
        force_text=(0.01, 0.25),
        force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle="-", lw=1),
        ax=ax_o,
    )
    fig.savefig(f"temporary/fault_detection_{vartype.name.lower()}.pdf")


if __name__ == "__main__":
    with open("visualization/config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    main(cfg)
