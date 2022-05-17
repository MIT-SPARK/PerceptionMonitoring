#!/usr/bin/env python3

from mlflow.tracking import MlflowClient
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.colors as mcolors
import yaml
from visualization.utils import load_artifact
import argparse

def generate_plot(config, dgraph_type):
    print(f"Generating {dgraph_type}")
    algos = config["Algorithms"]
    artifact_filename = config["Artifact"]
    marker_size = config["Plots"]["ROC"]["marker_size"]
    font_size = config["Plots"]["ROC"]["font_size"]
    # Collect results from MLflow
    client = MlflowClient()
    modules = np.zeros((len(algos), 2))
    outputs = np.zeros((len(algos), 2))
    colors, labels, markers = [], [], []
    for i, label in enumerate(algos.keys()):
        key = algos[label]["key"]
        run_id = algos[label][dgraph_type]
        stats = load_artifact(client, run_id, artifact_filename)
        if stats is None:
            raise Exception(f"No stats found for {label}")
        modules[i, :] = [
            stats[key]["modules"]["precision"] * 100,
            stats[key]["modules"]["recall"] * 100,
        ]
        outputs[i, :] = [
            stats[key]["outputs"]["precision"] * 100,
            stats[key]["outputs"]["recall"] * 100,
        ]
        labels.append(label)
        colors.append(mcolors.to_rgba_array([algos[label]["color"]]))
        markers.append(algos[label]["marker"])
    # PLOT
    with plt.style.context(["science"]):
        fig, (ax_o, ax_m) = plt.subplots(1, 2, figsize=(10, 3))
        # Modules
        for i in range(len(labels)):
            ax_m.scatter(
                modules[i, 0],
                modules[i, 1],
                c=colors[i],
                marker=markers[i],
                s=marker_size,
            )
        ax_m.set_xlabel("Precision (\%)")
        ax_m.set_ylabel("Recall (\%)")
        ax_m.set_title("Modules")
        mlabels = [
            ax_m.text(x, y, label, fontsize=font_size)
            for label, x, y in zip(labels, modules[:, 0], modules[:, 1])
        ]
        # Outputs
        for i in range(len(labels)):
            ax_o.scatter(
                outputs[i, 0],
                outputs[i, 1],
                c=colors[i],
                marker=markers[i],
                s=marker_size,
            )
        ax_o.set_xlabel("Precision (\%)")
        ax_o.set_ylabel("Recall (\%)")
        ax_o.set_title("Outputs")
        olabels = [
            ax_o.text(x, y, label, fontsize=font_size)
            for label, x, y in zip(labels, outputs[:, 0], outputs[:, 1])
        ]
    adjust_text(
        mlabels,
        lim=int(5e3),
        expand_points=(2, 2),
        force_text=(0.2, 0.7),
        arrowprops=dict(arrowstyle="-", lw=1),
        ax=ax_m,
    )
    adjust_text(
        olabels,
        lim=int(5e3),
        expand_points=(2, 2),
        force_text=(0.2, 0.7),
        arrowprops=dict(arrowstyle="-", lw=1),
        ax=ax_o,
    )
    fig.savefig(f"temporary/precision_recall_{dgraph_type}.pdf")


def main(config_filename):
    with open(config_filename, "r") as stream:
        cfg = yaml.safe_load(stream)
    generate_plot(config=cfg, dgraph_type="regular")
    generate_plot(config=cfg, dgraph_type="temporal")


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
    args = parser.parse_args()
    main(args.config)
