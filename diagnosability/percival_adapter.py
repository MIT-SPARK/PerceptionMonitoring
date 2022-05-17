import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional
from numpy import double

import pandas as pd
from sklearn import datasets

from diagnosability.base import DiagnosticModel, FailureStates, SystemState
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.diagnostic_factor_graph import DiagnosticFactorGraph
from enum import Enum
from collections import namedtuple
from shutil import copyfile


class Percival(DiagnosticModel):
    class InferenceAlgorithm(Enum):
        Diffusion = ("diffusion",)
        BruteForce = ("brute_force",)
        NaiveMeanField = ("naive_mean_field",)
        BeliefPropagation = ("belief_propagation",)
        LinearProgramming = ("linear_programming",)
        SimulatedAnnealing = ("simulated_annealing",)

        def to_str(self):
            return self.value[0]

    InferenceResults = namedtuple("InferenceResults", ["Inference", "Timing"])

    def __init__(self, base_model: DiagnosticFactorGraph, include_modules=True):
        assert isinstance(
            base_model, DiagnosticFactorGraph
        ), "base_model must be a DiagnosticFactorGraph"
        super().__init__()
        self.base_model = base_model
        self.trained_params = None
        self.train_bin = Path(os.getcwd(), r"bazel-bin/percival/app/train")
        self.inference_bin = Path(os.getcwd(), r"bazel-bin/percival/app/inference")
        self.include_modules = include_modules
        self._temp_dir = tempfile.TemporaryDirectory()
        assert (
            self.train_bin.is_file()
        ), f"Train binary not found `{self.train_bin}`, did you compile it?"
        assert (
            self.inference_bin.is_file()
        ), f"Inference binary not found `{self.inference_bin}`, did you compile it?"

    def __del__(self):
        self._temp_dir.cleanup()

    @property
    def failure_modes(self):
        return self.base_model.failure_modes

    @property
    def tests(self):
        return self.base_model.tests

    def save_model(self, output_filepath):
        with open(output_filepath, "w") as outfile:
            json.dump(
                self.base_model.to_dict(include_modules=self.include_modules),
                outfile,
            )

    def save_params(self, output_filepath):
        if self.trained_params is not None:
            copyfile(self.trained_params, output_filepath)

    def load_params(self, input_filepath):
        self.trained_params = Path(self._temp_dir.name, "checkpoint.json")
        copyfile(input_filepath, self.trained_params)

    def fault_identification(
        self,
        syndrome: SystemState,
        algorithm: InferenceAlgorithm = InferenceAlgorithm.BeliefPropagation,
        tolerance: float = 1e-8,
        max_iterations: int = 100,
    ) -> FailureStates:
        dd = DiagnosabilityDataset()
        syn_data = syndrome.syndrome.to_dict()
        dd.samples = pd.DataFrame(data=syn_data, index=[0])
        dd.model = self.base_model
        results = self.batch_fault_identification(
            dd, algorithm, tolerance, max_iterations
        )
        assert len(results) == 1, "Expected a single result."
        results = results.iloc[0]
        state = {}
        for f in self.failure_modes:
            try:
                state[f] = results[f]
            except KeyError:
                continue
        return FailureStates(state)

    def train(
        self,
        dataset: DiagnosabilityDataset,
        test_dataset: Optional[DiagnosabilityDataset] = None,
        cd_gibbs_sweeps: int = 1000,
        cd_batch_size: int = 0,
        cd_stepsize: double = 0.1,
        cd_iterations: int = 30,
        ml_max_iterations: int = 100,
        ml_tolerance: float = 1e-8,
        ssvm_tolerance: float = 1.0e-8,
        ssvm_regularization: float = 1,
        ssvm_opt_method: str = "bmrm",
        ssvm_iterations: int = 100,
        randomize: bool = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            mdl_base = Path(tmpdirname, "model.json")
            train_data = Path(tmpdirname, "train.csv")
            self.save_model(mdl_base)
            dataset.get_samples(include_modules=self.include_modules).to_csv(
                train_data, index=False
            )
            cmd = (
                f"{self.train_bin} --model {mdl_base} --dataset {train_data} --output {tmpdirname} "
                f"--ssvm_tolerance {ssvm_tolerance} --ssvm_regularization {ssvm_regularization} --ssvm_opt_method {ssvm_opt_method} --ssvm_iterations {ssvm_iterations} "
                f"--cd_gibbs_sweeps {cd_gibbs_sweeps} --cd_batch_size {cd_batch_size} --cd_stepsize {cd_stepsize} --cd_iterations {cd_iterations} "
                f"--ml_max_iterations {ml_max_iterations} --ml_tolerance {ml_tolerance} "
                f"--randomize {randomize} "
                "--verbose"
            )
            if test_dataset is not None:
                test_data = Path(tmpdirname, "test.csv")
                test_dataset.get_samples(include_modules=self.include_modules).to_csv(
                    test_data, index=False
                )
                cmd += f" --test_dataset {test_data}"
            ret = subprocess.run(cmd, shell=True)
            if ret.returncode != 0:
                print(f"[ERROR] Train failed with return code {ret.returncode}")
            # Collecing output
            output_mdl = Path(tmpdirname, "trained_model.json")
            output_params = Path(tmpdirname, "checkpoint.json")
            train_results_filename = Path(tmpdirname, "train_inference.csv")
            train_timing_filename = Path(tmpdirname, "train_timing.csv")
            self.trained_params = Path(self._temp_dir.name, "checkpoint.json")
            copyfile(output_params, self.trained_params)
            train_results = pd.read_csv(train_results_filename)
            train_timing = pd.read_csv(train_timing_filename)
            train_ret = Percival.InferenceResults(train_results, train_timing / 1e6)
            if test_dataset is not None:
                test_results_filename = Path(tmpdirname, "test_inference.csv")
                test_timing_filename = Path(tmpdirname, "test_timing.csv")
                test_results = pd.read_csv(test_results_filename)
                test_timing = pd.read_csv(test_timing_filename)
                test_ret = Percival.InferenceResults(test_results, test_timing / 1e6)
            else:
                test_ret = None
            return (train_ret, test_ret)

    def batch_fault_identification(
        self,
        dataset: DiagnosabilityDataset,
        algorithm: InferenceAlgorithm = InferenceAlgorithm.BeliefPropagation,
        tolerance: float = 1e-8,
        max_iterations: int = 100,
    ) -> pd.DataFrame:
        results = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            mdl_file = Path(tmpdirname, "model.json")
            params_file = Path(tmpdirname, "checkpoint.json")
            dataset_file = Path(tmpdirname, "test.csv")
            results_file = Path(tmpdirname, "inference.csv")
            timing_file = Path(tmpdirname, "timing.csv")
            self.save_model(mdl_file)
            if self.trained_params:
                copyfile(self.trained_params, params_file)
            dataset.get_samples(include_modules=self.include_modules).to_csv(
                dataset_file, index=False
            )
            cmd = f"{self.inference_bin} --model {mdl_file} --dataset {dataset_file} --output {tmpdirname}"
            if self.trained_params is not None:
                cmd += f" --params {params_file}"
            ret = subprocess.run(cmd, shell=True)
            assert (
                ret.returncode == 0
            ), "Something went wrong while running the inference binary."
            results = pd.read_csv(results_file)
            timing = pd.read_csv(timing_file)
            ret = Percival.InferenceResults(results, timing / 1e6)
        return ret
