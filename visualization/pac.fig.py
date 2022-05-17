#!/usr/bin/env python3

from black import Path
from mlflow.tracking import MlflowClient
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.perception_system import System
from visualization.utils import load_artifact


mark_every = 20

def bound(h_est, confidence, D, F):
    delta = 1 - confidence
    return h_est + F * np.sqrt(np.log(2 / delta) / (2 * D))


def generate_plot(config, dgraph_type):
    print(f"Generating {dgraph_type}")
    algos = config["Algorithms"]
    artifact_filename = config["Artifact"]
    font_size = config["Plots"]["Bounds"]["font_size"]
    line_width = config["Plots"]["Bounds"]["line_width"]
    confidence = np.linspace(0.9, 1 - (1e-12), 100)
    bbox_to_anchor = (-0.3, 1.0, 2.8, 0.25)
    # Collect results from MLflow
    client = MlflowClient()
    modules = np.zeros((len(algos), len(confidence)))
    outputs = np.zeros((len(algos), len(confidence)))
    colors, labels, markers = [], [], []
    legend_cols = len(algos) if len(algos) % 2 == 0 else len(algos) + 1
    legend_cols = int(legend_cols)
    for i, label in enumerate(algos):
        key = algos[label]["key"]
        run_id = algos[label][dgraph_type]
        params = client.get_run(run_id).data.params
        dataset_folder = Path(params["dataset"])
        dataset_path = dataset_folder / "train.pkl"
        dataset = DiagnosabilityDataset.load(dataset_path)
        stats = load_artifact(client, run_id, artifact_filename)
        if stats is None:
            raise Exception(f"No stats found for {label}")
        # fmodes = 
        num_modules_fmodes = len(dataset.model.system.get_failure_modes(System.Filter.MODULE_ONLY))
        num_outputs_fmodes = len(dataset.model.system.get_failure_modes(System.Filter.OUTPUT_ONLY))
        num_samples = len(dataset)
        modules[i, :] = bound(
            stats[key]["modules"]["HammingDistance"],
            confidence,
            num_samples,
            num_modules_fmodes,
        )
        outputs[i, :] = bound(
            stats[key]["outputs"]["HammingDistance"],
            confidence,
            num_samples,
            num_outputs_fmodes,
        )
        labels.append(label)
        colors.append(mcolors.to_rgba_array([algos[label]["color"]]))
        markers.append(algos[label]["marker"])
    # PLOT
    with plt.style.context(["science"]):
        fig, (ax_o, ax_m) = plt.subplots(1, 2, figsize=(10, 3))
        # Modules
        for i, label in enumerate(algos):
            ax_m.plot(
                confidence * 100,
                modules[i, :],
                label=label,
                color=colors[i],
                linewidth=line_width,
                marker=markers[i],
                markevery=mark_every,
            )
        ax_m.axvline(x=100, linestyle="dashed", color="xkcd:light grey")
        ax_m.set(xlabel="Confidence (\%)", ylabel="Bound")  # , fontsize=font_size)
        
        # Outputs
        for i, label in enumerate(algos):
            ax_o.plot(
                confidence * 100,
                outputs[i, :],
                label=label,
                color=colors[i],
                linewidth=line_width,
                marker=markers[i],
                markevery=mark_every,
            )
        ax_o.axvline(x=100, linestyle="dashed", color="xkcd:light grey")
        ax_o.set(xlabel="Confidence (\%)", ylabel="Bound")  # , fontsize=font_size)
        ax_o.legend(
                bbox_to_anchor=bbox_to_anchor,
                loc="upper left",
                mode="expand",
                borderaxespad=1,
                ncol=legend_cols,
                fontsize=font_size,
                handlelength=1,
            )
    fig.savefig(f"temporary/pac_bounds_{dgraph_type}.pdf")


def main():
    with open("visualization/config_alt.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    generate_plot(config=cfg, dgraph_type="regular")
    generate_plot(config=cfg, dgraph_type="temporal")


if __name__ == "__main__":
    main()
