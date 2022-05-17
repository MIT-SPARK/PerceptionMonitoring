from typing import Dict, Optional, Tuple

import gin
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diagnosability.base import Syndrome
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph
from diagnosability.factors import TestFactor
from diagnosability.gnn.utils import estimate_failure_modes_features_from_data
from networkx.relabel import relabel_nodes
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.utils import to_undirected

from diagnosability.perception_system import Module, System
from diagnosability.temporal_diagnostic_graph import TemporalDiagnosticFactorGraph


class DiagnosticModelAdapter:
    def __init__(
        self,
        dfg: DiagnosticFactorGraph,
        ignore_modules: bool = False,
        weighted_syndrome=False,
        unique_features=False,
    ) -> None:
        assert isinstance(
            dfg, DiagnosticFactorGraph
        ), "Only DiagnosticFactorGraphs are supported"
        self.dfg = dfg
        mn = dfg.to_markov_model(ignore_modules=ignore_modules)
        self.ordering = sorted(mn.nodes)
        self.tests_idx = {t: self.ordering.index(t) for t in self.dfg.tests}
        self.failures_idx = {t: self.ordering.index(t) for t in self.ordering}
        self.weighted_syndrome = weighted_syndrome
        self.unique_features = unique_features
        if not ignore_modules:
            if isinstance(dfg, TemporalDiagnosticFactorGraph):
                self.modules_failure_modes = [
                    f.varname
                    for sys in self.dfg.temporal_systems
                    for f in sys.get_failure_modes(System.Filter.MODULE_ONLY)
                ]
            else:
                self.modules_failure_modes = [
                    f.varname
                    for f in self.dfg.system.get_failure_modes(
                        System.Filter.MODULE_ONLY
                    )
                ]
        else:
            self.modules_failure_modes = []
        if isinstance(dfg, TemporalDiagnosticFactorGraph):
            self.outputs_failure_modes = [
                f.varname
                for sys in self.dfg.temporal_systems
                for f in sys.get_failure_modes(System.Filter.OUTPUT_ONLY)
            ]
        else:
            self.outputs_failure_modes = [
                f.varname
                for f in self.dfg.system.get_failure_modes(System.Filter.OUTPUT_ONLY)
            ]
        self.fmodes_parents = dict()
        for f in self.outputs_failure_modes:
            p = self.find_parent(f)
            self.fmodes_parents[f] = p.varname
        for f in self.modules_failure_modes:
            p = self.find_parent(f)
            self.fmodes_parents[f] = p.varname
        parents_id = {x: i for i, x in enumerate(set(self.fmodes_parents.values()))}
        self.fmodes_parents_idx = {
            f: parents_id[self.fmodes_parents[f]] for f in self.fmodes_parents
        }
        mn_int = relabel_nodes(mn, dict(zip(self.ordering, range(len(self.ordering)))))
        self.edge_index = torch.LongTensor(list(mn_int.edges)).t().contiguous()
        self.edge_index = to_undirected(self.edge_index)

    def find_parent(self, f):
        if isinstance(self.dfg, TemporalDiagnosticFactorGraph):
            for sys in self.dfg.temporal_systems:
                x = sys.find_by_varname(f)
                if x is None:
                    continue
                x = sys.parent(x)
                if x is None:
                    continue
                return x
            raise ValueError(f"Could not find parent for {f}")
        else:
            x = self.dfg.sys_query_by_varname(f)
            if x is None:
                raise ValueError(f"Could not find {f}")
            x = self.dfg.system.parent(x)
            if x is None:
                raise ValueError(f"Could not find parent for {f}")
            return x

    @property
    def num_features(self) -> int:
        return 2

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def failure_modes(self):
        return set(self.outputs_failure_modes) | set(self.modules_failure_modes)

    @property
    def tests(self):
        return set(self.dfg.tests)

    def test_scopes(self):
        factors = self.dfg.get_factors(filter=lambda phi: isinstance(phi, TestFactor))
        assert len(factors) == len(self.dfg.tests), "Cannot find all test factors"
        return {
            phi.test.varname: [f.varname for f in phi.test.scope] for phi in factors
        }

    @staticmethod
    def one_hot_enc(outcome):
        t = np.zeros(2, dtype=np.float32)
        t[outcome] = 1.0
        return t

    def syndrome_to_features(self, syndrome: Syndrome, info=None) -> torch.Tensor:
        if self.weighted_syndrome:
            assert info is not None, "Weighed syndrome requires info"
            weights = {
                t: min(
                    x if x is not None else 1.0 for x in info[t]["endpoints"].values()
                )
                for t in info
            }
        else:
            weights = {t: 1 for t in info}
        return {t: weights[t] * self.one_hot_enc(o.value) for t, o in syndrome.items()}

    def ideal_test_results(self, ground_truth):
        # ideal_test = lambda f: self.one_hot_enc(int(sum(f) > 0))
        ideal_test = lambda f: int(sum(f) > 0)
        test_scope = self.test_scopes()
        return {
            t: ideal_test([ground_truth[f].value for f in test_scope[t]])
            for t in self.dfg.tests
        }

    # def to_data(self, syndrome, failure_modes_features, ground_truth=None):
    def to_data(self, sample):
        if self.unique_features:
            failure_modes_features = {
                f: np.array([self.fmodes_parents_idx[f], self.failures_idx[f]])
                for f in sample.features.keys()
            }
        else:
            failure_modes_features = sample.features
        # ----
        ground_truth = sample.ground_truth
        syndrome = sample.syndrome
        features = {
            **failure_modes_features,
            **self.syndrome_to_features(syndrome, sample.info),
        }
        x = torch.from_numpy(np.array([features[k] for k in self.ordering])).type(
            torch.float
        )
        if ground_truth is not None:
            expected_results = {
                **{k: v for k, v in ground_truth.to_dict().items()},
                **self.ideal_test_results(ground_truth),
            }
            y = torch.tensor(
                [expected_results[k] for k in self.ordering], dtype=torch.long
            )
        else:
            y = None
        return Data(
            edge_index=self.edge_index,
            x=x,
            y=y,
            syndrome=syndrome,
            test_idx=self.tests_idx,
            failure_modes_idx=self.failures_idx,
            modules_failure_modes=self.modules_failure_modes,
            outputs_failure_modes=self.outputs_failure_modes,
            ordering=self.ordering,
        )


@gin.configurable
class FaultIdentificationDataModule(pl.LightningDataModule):
    @gin.configurable(denylist=["train", "validation", "test"])
    def __init__(
        self,
        train: DiagnosabilityDataset,
        validation: DiagnosabilityDataset,
        test: DiagnosabilityDataset,
        batch_size: int = 8,
        ignore_modules: bool = False,
        weighted_syndrome: bool = False,
        unique_features: bool = False,
    ) -> None:
        super().__init__()
        self.adapter = DiagnosticModelAdapter(
            train.model,
            ignore_modules=ignore_modules,
            weighted_syndrome=weighted_syndrome,
            unique_features=unique_features,
        )
        self.train = FaultIdentificationDataset(self.adapter, train)
        self.val = FaultIdentificationDataset(self.adapter, test)
        self.test = FaultIdentificationDataset(self.adapter, validation)
        self.batch_size = batch_size

    @property
    def num_features(self) -> int:
        return self.adapter.num_features

    @property
    def num_classes(self) -> int:
        return self.adapter.num_classes

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class FaultIdentificationDataset(Dataset):
    def __init__(
        self,
        adapter: DiagnosticModelAdapter,
        dataset: DiagnosabilityDataset,
    ) -> None:
        super(FaultIdentificationDataset, self).__init__(None, None, None)
        self.adapter = adapter
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.adapter.to_data(self.dataset[idx])
        # system_state = self.dataset[idx]
        # return self.adapter.to_data(
        #     system_state.syndrome,
        #     system_state.features,
        #     system_state.ground_truth,
        # )

    @property
    def num_features(self) -> int:
        return self.adapter.num_features

    @property
    def num_classes(self) -> int:
        return self.adapter.num_classes
