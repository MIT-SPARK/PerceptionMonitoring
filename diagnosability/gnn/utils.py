import tempfile
from os import path

import numpy as np
import pandas as pd
import torch
from diagnosability.base import FailureStates, Syndrome
from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph
from diagnosability.utils import Powerset
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class EarlyStoppingCondition(EarlyStopping):
    def __init__(self, min_val=0.0, **kwargs):
        super(EarlyStoppingCondition, self).__init__(**kwargs)
        self.min_val = min_val

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)
        if current >= self.min_val:
            self._run_early_stopping_check(trainer)
        else:
            self.wait_count = 0
            pass

    def on_train_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current = logs.get(self.monitor)
        if current >= self.min_val:
            self._run_early_stopping_check(trainer)
        else:
            self.wait_count = 0
            pass


# def compare(
#     adapter: DatasetAdapter, model: LightningModule, save_to_file: bool = False
# ) -> float:
#     num: float = 0
#     ok: float = 0
#     artifact = None
#     if save_to_file:
#         temp_dir = tempfile.mkdtemp()
#         artifact_name = path.join(temp_dir, "fault_identification_comparison.txt")
#         artifact = open(artifact_name, "w")
#     for failed_tests in Powerset(adapter.dfg.tests):
#         num += 1
#         syndrome = Syndrome({t: int(t in failed_tests) for t in adapter.dfg.tests})
#         fi_dfg = adapter.dfg.fault_identification(syndrome)
#         data = adapter.syndrome_to_data(syndrome).to(model.device)
#         with torch.no_grad():
#             y_hat = model(data)
#         var_state = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
#         fi_gnn = FailureStates(
#             {f: var_state[idx] for f, idx in adapter.failures_idx.items()}
#         )
#         msg = f"Syndrome: {syndrome.failed_tests()} -> DFG: {fi_dfg.active_failures()}, GNN: {fi_gnn.active_failures()}\n"
#         print(msg, end="")
#         if artifact is not None:
#             artifact.write(msg)
#         ok += float(fi_dfg == fi_gnn)
#     accuracy = ok / num
#     if artifact is not None:
#         artifact.write(f"Accuracy: {accuracy}")
#         artifact.close()
#     if save_to_file:
#         return accuracy, artifact_name
#     else:
#         return accuracy


def estimate_failure_modes_features_from_data(
    dfg: DiagnosticFactorGraph, samples: pd.DataFrame
):
    failures_samples = samples[dfg.failure_modes]
    x = failures_samples.apply(pd.Series.value_counts, axis=0)
    f_features = dict()
    for f in dfg.failure_modes:
        p = x.iloc[1][f] / samples.shape[0]
        if np.isnan(p):
            # TODO: does it make sense?
            f_features[f] = np.array([0,0], dtype=np.float32)
        else:
            f_features[f] = np.array([1 - p, p], dtype=np.float32)
    return f_features

