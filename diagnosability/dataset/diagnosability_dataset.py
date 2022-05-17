from collections import defaultdict, ChainMap
import pickle
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from diagnosability.base import DiagnosticModel, FailureStates, Syndrome, SystemState
from diagnosability.dataset.apollo import ApolloDiagnosabilityDataset
from diagnosability.dataset.base import TestOutcome
from diagnosability.dataset.utils import remap_diagnostic_graph
from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph
from diagnosability.factors import TestFactor
from diagnosability.perception_system import FailureMode, System
from diagnosability.temporal_diagnostic_graph import TemporalDiagnosticFactorGraph
from diagnosability.utils import split_by_fractions
from tqdm import tqdm
from itertools import groupby


@dataclass
class Sample:
    system_state: SystemState
    features: Dict[str, float]


def all_equal(lst):
    g = groupby(lst)
    return next(g, True) and not next(g, False)


def strictly_increasing(lst):
    return all(x < y for x, y in zip(lst, lst[1:]))


class DiagnosabilityDatasetPreprocessor(ApolloDiagnosabilityDataset):
    """Preprocesses Apollo Diagnosability Dataset into Pandas DataFrames.
    Example:
    >>> from diagnosability.dataset.preprocessor import DiagnosabilityDatasetPreprocessor
    >>> dataset = DiagnosabilityDatasetPreprocessor(dataset_path, config_path)
    >>> dataset.bake()
    >>> dataset.export(filename)
    """

    def __init__(self, dataset_filename: str, model_config: str):
        ApolloDiagnosabilityDataset.__init__(self)
        # Parse config file
        print(f"Loading config from {model_config}")
        with open(model_config, "r") as stream:
            self.model_cfg = yaml.safe_load(stream)
        # Load dataset
        tmp = ApolloDiagnosabilityDataset.load(dataset_filename)
        self.data = tmp.data
        # ts = [np.min([y.timestamp for y in x.test_results[-1]]) for x in self.data]
        assert len(self.data) > 0, "Dataset must have at least one sample."
        winsize = len(self.data[0].test_results) if self.data[0].temporal else 0
        # Instantiate DFG
        if winsize > 0:
            assert all(
                len(s.test_results) == winsize for s in self.data
            ), "All samples must have the same window size."
            self.dfg = TemporalDiagnosticFactorGraph.from_file(
                model_config, winsize=winsize
            )
        else:
            self.dfg = DiagnosticFactorGraph.from_file(model_config)
        assert self.dfg.system.has_oracle(), "Underlying system myst have an oracle."
        self._baked = None
        # Init
        with open(model_config, "r") as stream:
            cfg = yaml.safe_load(stream)
        if winsize > 0:
            self._init_temporal(cfg)
        else:
            self._init(cfg)

    def _init_temporal(self, cfg):
        self.ground_truth_test_names = cfg["ground_truth_tests"]
        self.failure_mode_map = dict()
        for fm, query in cfg["failure_modes"].items():
            self.failure_mode_map[fm] = [
                self.dfg.sys_query(query, i) for i in range(self.dfg.winsize)
            ]
        # self._validate_failure_mode_map()
        self.temporal_test_map = defaultdict(dict)
        for phi in self.dfg.get_factors(filter=lambda phi: isinstance(phi, TestFactor)):
            test = phi.test
            if test.timestep is not None:
                self.temporal_test_map[test.name][test.timestep] = phi.varname
        self.endpoints = dict()
        for unit, query in cfg["endpoints"].items():
            if query is not None:
                self.endpoints[unit] = [
                    self.dfg.sys_query(query, i) for i in range(self.dfg.winsize)
                ]

    def _init(self, cfg):
        # TODO: not tested
        # Create the map dataset_failure_mode -> DFG_failure_mode
        self.ground_truth_test_names = cfg["ground_truth_tests"]
        self.failure_mode_map = dict()
        for fm, query in cfg["failure_modes"].items():
            self.failure_mode_map[fm] = self.dfg.sys_query(query)
        self.endpoints = dict()
        for unit, query in cfg["endpoints"].items():
            if query is not None:
                self.endpoints[unit] = self.dfg.sys_query(query)

    def bake(self):
        if all(s.temporal for s in self.data):
            self.is_temporal = True
            return self._bake_temporal()
        elif all(not s.temporal for s in self.data):
            self.is_temporal = False
            return self._bake_simple()
        else:
            raise ValueError(
                "Dataset must contain either all temporal or all non-temporal samples."
            )

    def to_diagnosability_dataset(self):
        if self._baked is None:
            self.bake()
        dataset = DiagnosabilityDataset()
        dataset.is_temporal = self.is_temporal
        dataset.model = self.dfg
        dataset.timestamp = self._baked["timestamp"]
        dataset.samples = self._baked["dataset"]
        dataset.features = self._baked["features"]
        dataset.test_info = self._baked["test_info"]
        dataset.default_features = dataset.default_features_estimator(
            dataset.model, dataset.samples
        )
        return dataset

    def _bake_temporal(self):
        dataset, features, test_info, timestamps = [], [], [], []
        for sample in tqdm(self.data, "Samples"):
            var_states = dict()
            test_confidence = dict()
            tinfo = dict()
            ts = 0
            for tau, window in enumerate(sample.test_results):
                for dtest in window:
                    if not dtest.valid:
                        continue
                    for test in dtest.results:
                        if test.name in self.ground_truth_test_names:
                            varname = self.failure_mode_map[test.scope[0]][tau].varname
                        else:
                            varname = self.dfg.temporal_tests[test.name][tau].varname
                            test_confidence[varname] = test.confidence
                            tinfo[varname] = {
                                "endpoints": {
                                    e.name: e.data["confidence"]
                                    for e in dtest.endpoints
                                },
                                "timestamp": dtest.timestamp,
                            }
                        var_states[varname] = int(test.result)
                ts = max(ts, dtest.timestamp)
            for dtest in sample.temporal_test_results:
                for test in dtest.results:
                    try:
                        varname = self.temporal_test_map[test.name][test.timestep]
                    except KeyError:
                        raise KeyError(f"Error accessing test `{test.name}`.")
                    var_states[varname] = int(test.result)
                    test_confidence[varname] = test.confidence
                    tinfo[varname] = {"timestamp": dtest.timestamp, "endpoints": dict()}
                    for k in range(len(dtest.endpoints)):
                        for e in dtest.endpoints[k]:
                            ename = (
                                self.dfg.temporal_systems[k]
                                .query(self.model_cfg["endpoints"][e.name])
                                .varname
                            )
                            tinfo[varname]["endpoints"][ename] = e.data["confidence"]
                    # tinfo[varname] = {
                    #     "endpoints": [
                    #         {e.name: e.data["confidence"] for e in x}
                    #         for x in dtest.endpoints
                    #     ],
                    #
                    # }
                ts = max(ts, dtest.timestamp)
            z = [
                self.dfg.temporal_systems[t].oracle(evidence=var_states)
                for t in range(self.dfg.winsize)
            ]
            modules_states = dict(ChainMap(*z))
            modules_states = {k: int(v) for k, v in modules_states.items()}
            dataset.append({**var_states, **modules_states})
            features.append(test_confidence)
            timestamps.append(ts)
            test_info.append(tinfo)
        # Create pandas dataframe
        assert strictly_increasing(
            timestamps
        ), "Timestamps are not strictly increasing."
        self._baked = {
            "timestamp": timestamps,
            "dataset": pd.DataFrame(dataset),
            "features": pd.DataFrame(features),
            "test_info": test_info,
        }
        return self

    def _bake_simple(self):
        test_map = {
            phi.test.name: phi.test
            for phi in self.dfg.get_factors(lambda x: isinstance(x, TestFactor))
        }
        dataset, features, test_info, timestamps = [], [], [], []
        for sample in tqdm(self.data, "Samples"):
            # sample: List[DiagnosticTestResult]
            var_states, test_confidence, tinfo = dict(), dict(), dict()
            ts = 0
            for diagnostic_test in sample.test_results:
                if not diagnostic_test.valid:
                    continue
                for test in diagnostic_test.results:
                    if test.name in self.ground_truth_test_names:
                        varname = self.failure_mode_map[test.scope[0]].varname
                    else:
                        varname = test_map[test.name].varname
                        test_confidence[varname] = test.confidence
                        tinfo[varname] = {
                            "endpoints": {
                                self.dfg.system.query(
                                    self.model_cfg["endpoints"][e.name]
                                ).varname: e.data["confidence"]
                                for e in diagnostic_test.endpoints
                            },
                            "timestamp": diagnostic_test.timestamp,
                        }
                    outcome = int(test.result)
                    var_states[varname] = outcome
                ts = max(ts, diagnostic_test.timestamp)
            modules_states = self.dfg.system.oracle(evidence=var_states)
            modules_states = {k: int(v) for k, v in modules_states.items()}
            dataset.append({**var_states, **modules_states})
            features.append(test_confidence)
            timestamps.append(ts)
            test_info.append(tinfo)
        # Create pandas dataframe
        assert strictly_increasing(
            timestamps
        ), "Timestamps are not strictly increasing."
        self._baked = {
            "timestamp": timestamps,
            "dataset": pd.DataFrame(dataset),
            "features": pd.DataFrame(features),
            "test_info": test_info,
        }
        return self

    def export(self, output_filepath):
        assert self._baked is not None, "Dataset is not baked. Bake before exporting."
        outfile = Path(output_filepath)
        if outfile.is_file():
            raise ValueError("Output filename already exists.")
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirpath = Path(tmpdirname)
            pd.DataFrame(self._baked["timestamp"], columns=["timestamp"]).to_csv(
                tmpdirpath / "timestamp.csv", index=False
            )
            self._baked["dataset"].to_csv(tmpdirpath / "dataset.csv", index=False)
            self._baked["features"].to_csv(tmpdirpath / "features.csv", index=False)
            with open(tmpdirpath / "metadata.json", "w") as f:
                json.dump({"temporal": self.is_temporal}, f)
            with open(tmpdirpath / "test_info.json", "w") as f:
                json.dump(self._baked["test_info"], f)
            self.dfg.save(tmpdirpath / "dfg.pkl")
            # Zip the files
            with zipfile.ZipFile(outfile, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for f in tmpdirpath.glob("**/*"):
                    zip_file.write(f, f.relative_to(tmpdirpath))
        return outfile


class DiagnosabilityDataset:
    def __init__(
        self,
        dataset_file: Optional[str] = None,
        model: Optional[DiagnosticFactorGraph] = None,
        default_features: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.timestamp = list()
        self.samples = pd.DataFrame()
        self.features = pd.DataFrame()
        self.test_info = list()
        self.identification = None
        self.default_features = default_features
        self.is_temporal = False

        if dataset_file is not None:
            self.append(dataset_file)

    def get_samples(self, include_modules=True):
        if include_modules:
            return self.samples
        else:
            modules_vars = {
                v.varname
                for v in self.model.system.get_failure_modes(System.Filter.MODULE_ONLY)
            }
            cols = set(self.samples.columns) - modules_vars
            return self.samples[cols]

    def load_datasets(self, dataset_filepaths: List[str]):
        assert isinstance(dataset_filepaths, list), "Datasets filepaths must be a list."
        for dataset_filepath in dataset_filepaths:
            self.append(dataset_filepath)
        self.default_features_estimator(self.model, self.samples)

    def append(
        self,
        filename: str,
    ):
        archive = zipfile.ZipFile(filename, "r")
        ts = pd.read_csv(archive.open("timestamp.csv"))["timestamp"].values.tolist()
        samples = pd.read_csv(archive.open("dataset.csv"))
        features = pd.read_csv(archive.open("features.csv"))
        with archive.open("metadata.json") as f:
            metadata = json.load(f)
        with archive.open("test_info.json") as f:
            test_info = json.load(f)
        if metadata["temporal"]:
            dfg = TemporalDiagnosticFactorGraph.load(archive.open("dfg.pkl"))
            self.is_temporal = True
        else:
            dfg = DiagnosticFactorGraph.load(archive.open("dfg.pkl"))
        if self.model is None:
            self.model = dfg
        else:
            # Remap dataset to current model
            fm_map = remap_diagnostic_graph(dfg, self.model)
            samples = samples.rename(columns={c: fm_map[c] for c in samples.columns})
            features = features.rename(columns={c: fm_map[c] for c in features.columns})
            for i in range(len(test_info)):
                test_info[i] = {fm_map[k]: v for k, v in test_info[i].items()}
        self._validate_data(samples, features)
        if self.default_features is None:
            self.default_features = self.default_features_estimator(self.model, samples)
        self.samples = pd.concat([self.samples, samples], axis=0)
        self.features = pd.concat([self.features, features], axis=0)
        self.timestamp.extend(ts)
        self.test_info.extend(test_info)
        assert all_equal(
            [
                len(self.samples),
                len(self.features),
                len(self.timestamp),
                len(self.test_info),
            ]
        ), "Size disagreement."

    def change_model(self, model: DiagnosticFactorGraph):
        fm_map = remap_diagnostic_graph(self.model, model)
        self.samples = self.samples.rename(
            columns={c: fm_map[c] for c in self.samples.columns}
        )
        self.features = self.features.rename(
            columns={c: fm_map[c] for c in self.features.columns}
        )
        self.default_features = {fm_map[k]: v for k, v in self.default_features.items()}
        self.model = model
        for i in range(len(self.test_info)):
            self.test_info[i] = {fm_map[k]: v for k, v in self.test_info[i].items()}

    def _to_system_state(self, idx):
        sample, features = self.samples.iloc[idx], self.features.iloc[idx]
        info = self.test_info[idx]
        f_features = self.default_features.copy()
        for varname, feature in features.items():
            if feature in self.model.failure_modes:
                f_features[varname] = self.feature_encoder(feature)
        syndrome = Syndrome({t: sample[t] for t in self.model.tests})
        ground_truth = FailureStates({f: sample[f] for f in self.model.failure_modes})
        if self.identification is not None:
            state = FailureStates(
                {f: self.identification.iloc[idx][f] for f in self.model.failure_modes}
            )
        else:
            state = None
        return SystemState(
            timestamp=self.timestamp[idx],
            syndrome=syndrome,
            features=f_features,
            states=state,
            ground_truth=ground_truth,
            info=info,
        )

    def _validate_data(self, samples, features):
        assert isinstance(self.model, DiagnosticFactorGraph) or isinstance(
            self.model, TemporalDiagnosticFactorGraph
        ), "Only DiagnosticModel is supported."
        assert len(samples) == len(features), "Datset size mismatch"
        assert self.model.failure_modes.issubset(
            set(samples.columns)
        ), "Dataset columns do not contain all failure modes"
        assert self.model.tests.issubset(
            set(samples.columns)
        ), "Features columns do not contain all tests"

    def set_fault_identification_results(self, identification):
        assert len(identification) == len(self.samples), "Identification size mismatch"
        assert all(
            f in identification.columns for f in self.model.failure_modes
        ), "Missing columns in identification"
        self.identification = identification

    @staticmethod
    def default_features_estimator(dfg: DiagnosticFactorGraph, samples: pd.DataFrame):
        failures_samples = samples[dfg.failure_modes]
        z = failures_samples.sum(axis=0) / failures_samples.shape[0]
        return {c: np.array([1 - z[c], z[c]]) for c in dfg.failure_modes}

    @staticmethod
    def feature_encoder(feature_value: float) -> np.ndarray:
        if np.isnan(feature_value):
            return np.array([0.0, 0.0], dtype=np.float32)
        else:
            return np.array([feature_value, 1 - feature_value], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._to_system_state(idx)

    def get_test_info(self, idx):
        return self.test_info[idx]

    def split(self, splits):
        assert all(s >= 0.0 for s in splits), "Splits must be positive."
        splits = list(splits)
        dataset_splits = []
        df_splits, idxs = split_by_fractions(
            dfs=[self.samples, self.features], splits=splits, randomize=True
        )
        samples, features, test_info, timestamps = [], [], [], []
        for i in range(len(splits)):
            timestamps.append([self.timestamp[x] for x in idxs[i]])
            samples.append(df_splits[i][0])
            features.append(df_splits[i][1])
            test_info.append([self.test_info[x] for x in idxs[i]])
        # for d, f, i, t in zip(samples, features, test_info, timestamps):
        for k in range(len(splits)):
            ds = DiagnosabilityDataset()
            ds.model = self.model
            # ds.samples, ds.features, ds.test_info, ds.timestamps = d, f, i, t
            ds.timestamp = timestamps[k]
            ds.samples = samples[k]
            ds.features = features[k]
            ds.test_info = test_info[k]
            ds.default_features = self.default_features
            ds.is_temporal = self.is_temporal
            dataset_splits.append(ds)
        return tuple(dataset_splits)

    ## I/O
    def save(self, filename):
        picklefile = open(filename, "wb")
        pickle.dump(self, picklefile)
        picklefile.close()

    @classmethod
    def load(cls, filename):
        # if isinstance(filename, str):
        picklefile = open(filename, "rb")
        # else:
        # picklefile = filename
        dataset = pickle.load(picklefile)
        picklefile.close()
        return dataset

    def export(self, outfile):
        """Export a simplified dataset."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirpath = Path(tmpdirname)
            with open(tmpdirpath / "model.json", "w") as modelfile:
                json.dump(self.model.to_dict(), modelfile)
            samples = self.samples.copy()
            samples.insert(0, "Timestamp", self.timestamp, True)
            samples.to_csv(tmpdirpath / "samples.csv", index=False)
            if self.identification is not None:
                self.identification.to_csv(tmpdirpath / "identification.csv", index=False)
            with zipfile.ZipFile(outfile, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for f in tmpdirpath.glob("**/*"):
                    zip_file.write(f, f.relative_to(tmpdirpath))
