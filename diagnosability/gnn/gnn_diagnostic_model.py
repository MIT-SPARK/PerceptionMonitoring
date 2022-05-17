from typing import Dict

import torch
from diagnosability.base import DiagnosticModel, FailureStates, SystemState
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.gnn.datamodule import DiagnosticModelAdapter
from pytorch_lightning.core.lightning import LightningModule
from collections import namedtuple
from time import perf_counter


class GNNDiagnosticModel(DiagnosticModel):

    InferenceResults = namedtuple("InferenceResults", ["Inference", "Timing"])

    def __init__(self, adapter: DiagnosticModelAdapter, model: LightningModule) -> None:
        assert isinstance(
            adapter, DiagnosticModelAdapter
        ), "adapter must be a DiagnosticModelAdapter"
        assert isinstance(model, LightningModule), "model must be a LightningModule"
        self.adapter = adapter
        self.model = model

    @property
    def failure_modes(self):
        return self.adapter.failure_modes

    @property
    def tests(self):
        return self.adapter.tests

    def batch_fault_identification(self, dataset: DiagnosabilityDataset):
        results = []
        timing = []
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                data = self.adapter.to_data(sample)
                t_start = perf_counter()
                y_hat = self.model(data)
                t_end = perf_counter()
                var_state = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
                results.append(
                    FailureStates(
                        {
                            f: var_state[idx]
                            for f, idx in self.adapter.failures_idx.items()
                        }
                    )
                )
                timing.append((t_end - t_start)*1e3)
        return self.InferenceResults(results, timing)

    def fault_identification(self, syndrome: SystemState) -> FailureStates:
        data = self.adapter.to_data(syndrome)
        with torch.no_grad():
            y_hat = self.model(data)
        var_state = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
        return FailureStates(
            {f: var_state[idx] for f, idx in self.adapter.failures_idx.items()}
        )
