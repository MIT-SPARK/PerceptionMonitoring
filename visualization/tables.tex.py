#!/usr/bin/env python3

from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
from string import Template
from visualization.utils import load_artifact
import yaml
from visualization.detection_fig import preprocess as compute_detection_stats
from visualization.detection_fig import VarType
from diagnosability.evaluator import Stats
from pathlib import Path
import pandas as pd
import argparse

accuracy_table_template_single = Template(
    """
\\begin{table}
  \hspace*{-3cm}
  \smaller
  \\begin{tabular}{ |l||c|c|c|c|c|c|c|c|c|c|c|c| }
    \hline
    \multirow{3}{*}{\\textbf{Algorithm}} & \multicolumn{6}{c|}{\\textbf{Regular}} & \multicolumn{6}{c|}{\\textbf{Temporal}}\\\\
    & \multicolumn{3}{c|}{Fault Identification} & \multicolumn{3}{c|}{Fault Detection} & \multicolumn{3}{c|}{Fault Identification} & \multicolumn{3}{c|}{Fault Detection} \\\\
    & All & Outputs & Modules & All & Outputs & Modules  & All & Outputs & Modules & All & Outputs & Modules  \\\\
    \hline
$body   \hline
  \end{tabular}
  \caption{Fault Identification and Detection Accuracy for each algorithm.}
  \label{tab:accuracies}
\end{table}
"""
)

accuracy_table_template = Template(
    """
\\begin{table}
  \hspace*{-2cm}
  \smaller
  \\begin{tabular}{ |l||c|c|c|c|c|c| }
    \hline
    \multirow{2}{*}{\\textbf{Algorithm}} & \multicolumn{3}{c|}{\\textbf{Regular}} & \multicolumn{3}{c|}{\\textbf{Temporal}}\\\\
    & All & Outputs & Modules & All & Outputs & Modules \\\\\
    \hline
    $body
   \hline
  \end{tabular}
  \caption{$title Accuracy for each algorithm.}
  \label{tab:$label}
\end{table}
"""
)

timing_table_template = Template(
    """
\\begin{table}
  \hspace*{-1.7cm}
  \smaller{
  \\begin{tabular}{ l $cols }
    & $labels \\\\
    \hline
    $body
  \end{tabular}
  }
  \caption{Runtime for Fault Identification in milliseconds. Average and standard deviation.}
  \label{tab:timing}
\end{table}
"""
)


def sumstats(cm1, cm2):
    scm = {k: cm1[k] + cm2[k] for k in cm1.keys()}
    return Stats.all(scm)


def generate_accuracy_table(config, single=False):
    algos = config["Algorithms"]
    artifact_filename = config["Artifact"]
    client = MlflowClient()
    det = {
        "regular": {
            "outputs": compute_detection_stats(config, "regular", VarType.OUTPUTS),
            "modules": compute_detection_stats(config, "regular", VarType.MODULES),
        },
        "temporal": {
            "outputs": compute_detection_stats(config, "temporal", VarType.OUTPUTS),
            "modules": compute_detection_stats(config, "temporal", VarType.MODULES),
        },
    }
    label_len = max(len(l) for l in algos) + 1
    if single:
        body = ""
        for label, alg in algos.items():
            key = alg["key"]
            reg = load_artifact(client, alg["regular"], artifact_filename)
            temp = load_artifact(client, alg["temporal"], artifact_filename)
            if reg is None or temp is None:
                raise Exception(f"No stats found for {label}")
            dreg = sumstats(
                det["regular"]["modules"][label]["confusion_matrix"],
                det["regular"]["outputs"][label]["confusion_matrix"],
            )
            dtemp = sumstats(
                det["temporal"]["modules"][label]["confusion_matrix"],
                det["temporal"]["outputs"][label]["confusion_matrix"],
            )
            ireg = sumstats(
                reg[key]["modules"]["confusion_matrix"],
                reg[key]["outputs"]["confusion_matrix"],
            )
            itemp = sumstats(
                temp[key]["modules"]["confusion_matrix"],
                temp[key]["outputs"]["confusion_matrix"],
            )
            entry = (
                f"\t{label.ljust(label_len, ' ')} & "
                f"{ireg['accuracy']*100:.2f} & "
                f"{reg[key]['outputs']['accuracy']*100:.2f} & "
                f"{reg[key]['modules']['accuracy']*100:.2f} & "
                f"{dreg['accuracy']*100:.2f} & "
                f"{det['regular']['outputs'][label]['accuracy']*100:.2f} & "
                f"{det['regular']['modules'][label]['accuracy']*100:.2f} & "
                f"{itemp['accuracy']*100:.2f} & "
                f"{temp[key]['outputs']['accuracy']*100:.2f} & "
                f"{temp[key]['modules']['accuracy']*100:.2f} & "
                f"{dtemp['accuracy']*100:.2f} &"
                f"{det['temporal']['outputs'][label]['accuracy']*100:.2f} & "
                f"{det['temporal']['modules'][label]['accuracy']*100:.2f} \\\\"
                "\n"
            )
            body += entry
        print(accuracy_table_template_single.substitute({"body": body}))
    else:
        # Fault Identification
        body = ""
        for label, alg in algos.items():
            key = alg["key"]
            reg = load_artifact(client, alg["regular"], artifact_filename)
            temp = load_artifact(client, alg["temporal"], artifact_filename)
            if reg is None or temp is None:
                raise Exception(f"No stats found for {label}")
            ireg = sumstats(
                reg[key]["modules"]["confusion_matrix"],
                reg[key]["outputs"]["confusion_matrix"],
            )
            itemp = sumstats(
                temp[key]["modules"]["confusion_matrix"],
                temp[key]["outputs"]["confusion_matrix"],
            )
            entry = (
                f"\t{label.ljust(label_len, ' ')} & "
                f"{ireg['accuracy']*100:.2f} & "
                f"{reg[key]['outputs']['accuracy']*100:.2f} & "
                f"{reg[key]['modules']['accuracy']*100:.2f} & "
                f"{itemp['accuracy']*100:.2f} & "
                f"{temp[key]['outputs']['accuracy']*100:.2f} & "
                f"{temp[key]['modules']['accuracy']*100:.2f} \\\\ "
                "\n"
            )
            body += entry
        print(
            accuracy_table_template.substitute(
                {
                    "title": "Fault Identification",
                    "label": "fault_identification",
                    "body": body,
                }
            )
        )
        # Fault Detection
        body = ""
        for label, alg in algos.items():
            key = alg["key"]
            reg = load_artifact(client, alg["regular"], artifact_filename)
            temp = load_artifact(client, alg["temporal"], artifact_filename)
            if reg is None or temp is None:
                raise Exception(f"No stats found for {label}")
            dreg = sumstats(
                det["regular"]["modules"][label]["confusion_matrix"],
                det["regular"]["outputs"][label]["confusion_matrix"],
            )
            dtemp = sumstats(
                det["temporal"]["modules"][label]["confusion_matrix"],
                det["temporal"]["outputs"][label]["confusion_matrix"],
            )
            entry = (
                f"\t{label.ljust(label_len, ' ')} & "
                f"{dreg['accuracy']*100:.2f} & "
                f"{det['regular']['outputs'][label]['accuracy']*100:.2f} & "
                f"{det['regular']['modules'][label]['accuracy']*100:.2f} & "
                f"{dtemp['accuracy']*100:.2f} &"
                f"{det['temporal']['outputs'][label]['accuracy']*100:.2f} & "
                f"{det['temporal']['modules'][label]['accuracy']*100:.2f} \\\\ "
                "\n"
            )
            body += entry
        print(
            accuracy_table_template.substitute(
                {
                    "title": "Fault Detection",
                    "label": "fault_detection",
                    "body": body,
                }
            )
        )


def generate_timing_table(config):
    algos = config["Algorithms"]
    artifact_filename = config["TimingArtifact"]
    client = MlflowClient()
    label_len = max([len("Regular"), len("Temporal")]) + 1
    labels = algos.keys()  # fix an ordering
    template = {"cols": "c" * len(algos), "labels": " & ".join(labels), "body": ""}
    for dgraph in ["regular", "temporal"]:
        timing_cols = []
        for label, alg in algos.items():
            artifact_path = load_artifact(client, alg[dgraph], artifact_filename)
            assert isinstance(artifact_path, Path)
            with open(artifact_path) as f:
                timing_cols.append(pd.read_csv(f, skiprows=1, names=[label]))
        timing = pd.concat(timing_cols, axis=1)
        template["body"] += f"\t{dgraph.title().ljust(label_len, ' ')} & "
        template["body"] += " & ".join(
            f"{timing[label].mean():.2f} ({timing[label].std():.2f})"
            for label in labels
        )
        template["body"] += " \\\\ \n"
    print(timing_table_template.substitute(template))


def main(config_file):
    with open(config_file, "r") as stream:
        cfg = yaml.safe_load(stream)
    generate_accuracy_table(config=cfg, single=False)
    generate_timing_table(config=cfg)


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
