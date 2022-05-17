#!/usr/bin/env python3

from termios import CINTR
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
from string import Template
import pprint
from visualization.utils import load_artifact
import yaml

table_template = Template("""
\\begin{table}
  \hspace*{-5em}
  \\begin{tabular}{ |l||c|c|c|c|c|c|c|c| }
    \hline
    \multirow{2}{*}{\\textbf{Algorithm}} & \multicolumn{2}{c|}{\\textbf{Accuracy}} & \multicolumn{2}{c|}{\\textbf{Precision}} & \multicolumn{2}{c|}{\\textbf{Recall}}& \multicolumn{2}{c|}{\\textbf{F-Score}}\\\\
    & Modules & Outputs & Modules & Outputs & Modules & Outputs & Modules & Outputs\\\\
    \hline
$body   \hline
  \end{tabular}
  \caption{Fault Identification Results in $title \DGraphs}
  \label{tab:fault_identification_results_$name}
\end{table}
"""
)

def generate_precision_recall_table(config, dgraph_type):
    algos = config["Algorithms"]
    artifact_filename = config["Artifact"]
    client = MlflowClient()
    body = ""
    # hamming = {"modules": dict(), "outputs": dict()}
    label_len = max(len(l) for l in algos)+1
    for label, alg in algos.items():
        key = alg["key"]
        run_id = alg[dgraph_type]
        stats = load_artifact(client, run_id, artifact_filename)
        if stats is None:
            raise Exception(f"No stats found for {label}")
        entry = (
            f"\t{label.ljust(label_len, ' ')} & "
            f"{stats[key]['modules']['accuracy']*100:.2f} & "
            f"{stats[key]['outputs']['accuracy']*100:.2f} & "
            f"{stats[key]['modules']['precision']*100:.2f} & "
            f"{stats[key]['outputs']['precision']*100:.2f} & "
            f"{stats[key]['modules']['recall']*100:.2f} & "
            f"{stats[key]['outputs']['recall']*100:.2f} & "
            f"{stats[key]['modules']['f1']*100:.2f} & "
            f"{stats[key]['outputs']['f1']*100:.2f} \\\\"
            "\n"
        )
        body += entry
        # hamming["modules"][label] = stats[key]['modules']['HammingDistance']
        # hamming["outputs"][label] = stats[key]['outputs']['HammingDistance']
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(hamming)
    print(table_template.substitute({"body": body, "name": dgraph_type, "title": str.title(dgraph_type)}))

def main():
    with open("visualization/config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    generate_precision_recall_table(config=cfg, dgraph_type="regular")
    print("="*50)
    generate_precision_recall_table(config=cfg, dgraph_type="temporal")


if __name__ == "__main__":
    main()