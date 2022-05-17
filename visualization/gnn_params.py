from mlflow.tracking import MlflowClient
from rich import print
import yaml

def print_params(config):
    algos = config["Algorithms"]
    # Collect results from MLflow
    client = MlflowClient()
    params = dict()
    for label in algos:
        reg = client.get_run(algos[label]["regular"]).data.params
        temp = client.get_run(algos[label]["temporal"]).data.params
        if "model" in reg and reg["model"] == "GraphConvolution":
            params[label] = {
                "regular": {
                    "layers:": reg["num_layers"],
                    "max_epochs": reg["max_epochs"],
                    "batch_size": reg["batch_size"],
                    "hidden_channels": reg["hidden_channels"],
                },
                "temporal": {
                    "layers:": temp["num_layers"],
                    "max_epochs": temp["max_epochs"],
                    "batch_size": temp["batch_size"],
                    "hidden_channels": temp["hidden_channels"],
                },
            }
    print(params)

def main():
    with open("visualization/config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    print_params(config=cfg)


if __name__ == "__main__":
    main()
