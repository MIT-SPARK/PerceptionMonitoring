from tempfile import TemporaryFile
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from sklearn import datasets
from torchmetrics.metric import Metric
from diagnosability.base import FailureStates, Syndrome, DiagnosticModel
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
import torch
from tqdm import tqdm
from diagnosability.factors import TestFactor
from diagnosability.perception_system import System
from sklearn.metrics import confusion_matrix as compute_confusion_matrix
from math import sqrt
import numpy as np

TEMPORAL_EVALUATION_LAST_ONLY = True


def safestat(func):
    """Wrapper to make sure ZeroDivisionError is handled gracefully."""

    def wrapper(*args, **kwargs):
        try:
            r = float(func(*args, **kwargs))
        except ZeroDivisionError:
            r = float("nan")
        return r

    return wrapper


class ModelEvaluator:
    def __init__(self, system: Union[System, List[System]], metrics: List[Callable]):
        assert metrics, "At least one metric must be provided"
        self.system = system
        self.metrics = metrics

    def evaluate(
        self,
        model: DiagnosticModel,
        dataset: DiagnosabilityDataset,
        results: Optional[pd.DataFrame] = None,
        ignore_modules=False,
    ):
        """
        Evaluate the model on the dataset.
        """
        # Instantiate metrics
        if not ignore_modules:
            metrics_modules = [metric() for metric in self.metrics]
            assert all(
                isinstance(m, Metric) for m in metrics_modules
            ), "All metrics must be Metric"
        metrics_outputs = [metric() for metric in self.metrics]
        assert all(
            isinstance(m, Metric) for m in metrics_outputs
        ), "All metrics must be Metric"
        # Computet the metrics
        # fmt: off
        if isinstance(self.system, System):
            fm_output = sorted([f.varname for f in self.system.get_failure_modes(System.Filter.OUTPUT_ONLY)])
            fm_modules = sorted([f.varname for f in self.system.get_failure_modes(System.Filter.MODULE_ONLY)]) if not ignore_modules else None 
        else:
            if TEMPORAL_EVALUATION_LAST_ONLY:
                fm_output = sorted([f.varname for f in self.system[-1].get_failure_modes(System.Filter.OUTPUT_ONLY)])
                fm_modules = sorted([f.varname for f in self.system[-1].get_failure_modes(System.Filter.MODULE_ONLY)]) if not ignore_modules else None 
            else:
                fm_modules = sorted([f.varname for sys in self.system for f in sys.get_failure_modes(System.Filter.MODULE_ONLY)]) if not ignore_modules else None
                fm_output = sorted([f.varname for sys in self.system for f in sys.get_failure_modes(System.Filter.OUTPUT_ONLY)])
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            assert sample.ground_truth is not None, "The dataset must have ground truth"
            if results is None:
                fi = model.fault_identification(sample)
                if not ignore_modules:
                    y_hat_modules = torch.IntTensor([fi[v].value for v in fm_modules])
                y_hat_output = torch.IntTensor([fi[v].value for v in fm_output])
            else:
                if isinstance(results, pd.DataFrame):
                    if not ignore_modules:
                        y_hat_modules = torch.IntTensor([results.iloc[i][v] for v in fm_modules])
                    y_hat_output = torch.IntTensor([results.iloc[i][v] for v in fm_output])
                else:
                    if not ignore_modules:
                        y_hat_modules = torch.IntTensor([results[i][v].value for v in fm_modules])
                    y_hat_output = torch.IntTensor([results[i][v].value for v in fm_output])
            if not ignore_modules:
                y_true_modules = torch.IntTensor([sample.ground_truth[v].value for v in fm_modules])
            y_true_output = torch.IntTensor([sample.ground_truth[v].value for v in fm_output])
            if not ignore_modules:
                for metric in metrics_modules:
                    metric.update(y_hat_modules, y_true_modules)
            for metric in metrics_outputs:
                metric.update(y_hat_output, y_true_output)
        # fmt: on
        # Compute the results
        res = {"modules": dict(), "outputs": dict()}
        if not ignore_modules:
            for metric in metrics_modules:
                res["modules"][type(metric).__name__] = metric.compute()
        for metric in metrics_outputs:
            res["outputs"][type(metric).__name__] = metric.compute()
        return res


def FaultIdentificationBound(
    num_failures: int,
    num_samples: int,
    probability: float,
    empirical_hamming_distance: Union[float, torch.tensor],
):
    delta = 1 - probability
    if torch.is_tensor(empirical_hamming_distance):
        assert (
            empirical_hamming_distance.numel() == 1
        ), "empirical_hamming_distance must be a scalar"
        h = empirical_hamming_distance.flatten().cpu().detach().numpy()[0]
    else:
        h = empirical_hamming_distance
    return num_failures * (h + np.sqrt(np.log(2 / delta) / (2 * num_samples)))


class FaultDetectionEvaluator:
    """Compute the metrics for fault detection."""

    @staticmethod
    def evaluate(dataset: DiagnosabilityDataset):
        mdl = dataset.model
        if dataset.is_temporal and TEMPORAL_EVALUATION_LAST_ONLY:
            vars = set(f.varname for f in mdl.temporal_systems[-1].get_failure_modes())
            tests = [
                phi.test.varname
                for phi in mdl.factors
                if isinstance(phi, TestFactor) and bool(set(phi.scope()) & vars)
            ]
        else:
            vars = mdl.failure_modes
            tests = mdl.tests
        gt, pred = [], []
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            # Did the system experience at least one failure?
            # gt.append(bool(sample.ground_truth.active_failures()))
            gt.append(int(any(sample.ground_truth[v].value for v in vars)))
            # Did at least one test fail?
            # pred.append(bool(sample.syndrome.failed_tests()))
            pred.append(int(any(sample.syndrome[t].value for t in tests)))
        tn, fp, fn, tp = compute_confusion_matrix(gt, pred).ravel()
        return {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}


class DiagnosticTestsEvaluator:
    @staticmethod
    def evaluate(dataset: DiagnosabilityDataset):
        test_confusion_matrices = dict()
        for phi in tqdm(dataset.model.get_factors(lambda f: isinstance(f, TestFactor))):
            test = phi.test
            outcomes = dataset.samples[test.varname] > 0
            gt = dataset.samples[[f.varname for f in test.scope]].sum(axis=1) > 0
            df = pd.concat([outcomes, gt], axis=1)
            df.columns = ["Outcomes", "Actual"]
            confusion_matrix = pd.crosstab(
                df["Actual"],
                df["Outcomes"],
                rownames=["Actual"],
                colnames=["Outcomes"],
            )
            try:
                tp = confusion_matrix.loc[True, True]
            except:
                tp = 0
            try:
                fp = confusion_matrix.loc[False, True]
            except:
                fp = 0
            try:
                fn = confusion_matrix.loc[True, False]
            except:
                fn = 0
            try:
                tn = confusion_matrix.loc[False, False]
            except:
                tn = 0
            test_confusion_matrices[test.varname] = {
                "tp": float(tp),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
            }
        return test_confusion_matrices


class Stats:
    @staticmethod
    def confusion_matrix_to_dict(cm: torch.Tensor) -> Dict[str, float]:
        """
        Convert a confusion matrix to a dictionary.
        """
        tn, fp, fn, tp = tuple(cm.flatten().cpu().detach().numpy())
        return {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}

    @staticmethod
    def all(cm: Union[Dict[str, int], torch.Tensor]):
        """
        Compute all metrics from a confusion matrix.
        """
        if isinstance(cm, torch.Tensor):
            cm_dict = Stats.confusion_matrix_to_dict(cm)
        else:
            cm_dict = cm
        return {
            "confusion_matrix": cm_dict,
            "accuracy": Stats.accuracy(cm_dict),
            "f1": Stats.f1(cm_dict),
            "f2": Stats.f2(cm_dict),
            "precision": Stats.precision(cm_dict),
            "recall": Stats.recall(cm_dict),
            "specificity": Stats.specificity(cm_dict),
            "informedness": Stats.informedness(cm_dict),
            "odds_ratio": Stats.odds_ratio(cm_dict),
            "phi": Stats.phi(cm_dict),
        }

    @staticmethod
    @safestat
    def negative_predictive_value(cm: Dict[str, float]) -> float:
        return cm["tn"] / (cm["tn"] + cm["fn"])

    @staticmethod
    @safestat
    def miss_rate(cm: Dict[str, float]) -> float:
        return cm["fn"] / (cm["fn"] + cm["tp"])

    # aka selectivity or true negative rate (TNR)
    @staticmethod
    @safestat
    def specificity(confusion_matrix: Dict[str, float]) -> float:
        return confusion_matrix["tn"] / (
            confusion_matrix["tn"] + confusion_matrix["fp"]
        )

    @staticmethod
    @safestat
    def accuracy(confusion_matrix: Dict[str, float]):
        return (confusion_matrix["tp"] + confusion_matrix["tn"]) / (
            confusion_matrix["tp"]
            + confusion_matrix["tn"]
            + confusion_matrix["fp"]
            + confusion_matrix["fn"]
        )

    # aka positive predictive value (PPV)
    @staticmethod
    @safestat
    def precision(confusion_matrix: Dict[str, float]) -> float:
        return confusion_matrix["tp"] / (
            confusion_matrix["tp"] + confusion_matrix["fp"]
        )

    # aka sensitivity, hit rate, or true positive rate (TPR)
    @staticmethod
    @safestat
    def recall(confusion_matrix: Dict[str, float]) -> float:
        return confusion_matrix["tp"] / (
            confusion_matrix["tp"] + confusion_matrix["fn"]
        )

    @staticmethod
    @safestat
    def f1(confusion_matrix: Dict[str, float]) -> float:
        return (
            2
            * confusion_matrix["tp"]
            / (
                2 * confusion_matrix["tp"]
                + confusion_matrix["fp"]
                + confusion_matrix["fn"]
            )
        )

    @staticmethod
    @safestat
    def f2(confusion_matrix: Dict[str, float]) -> float:
        return (
            5
            * confusion_matrix["tp"]
            / (
                5 * confusion_matrix["tp"]
                + 4 * confusion_matrix["fp"]
                + confusion_matrix["fn"]
            )
        )

    @staticmethod
    @safestat
    def informedness(confusion_matrix: Dict[str, float]) -> float:
        tp_plus_fn = confusion_matrix["tp"] + confusion_matrix["fn"]
        tn_plus_fp = confusion_matrix["tn"] + confusion_matrix["fp"]
        return (
            confusion_matrix["tp"] / tp_plus_fn
            + confusion_matrix["tn"] / tn_plus_fp
            - 1
        )

    @staticmethod
    @safestat
    def odds_ratio(confusion_matrix: Dict[str, float]) -> float:
        fp_times_fn = confusion_matrix["fp"] * confusion_matrix["fn"]
        return (confusion_matrix["tp"] * confusion_matrix["tn"]) / fp_times_fn

    # https://en.wikipedia.org/wiki/Phi_coefficient
    @staticmethod
    @safestat
    def phi(confusion_matrix: Dict[str, float]) -> float:
        tp = confusion_matrix["tp"]
        tn = confusion_matrix["tn"]
        fp = confusion_matrix["fp"]
        fn = confusion_matrix["fn"]
        n = tp + fp + tn + fn
        s = (tp + fn) / n
        p = (tp + fp) / n
        return (tp / n - s * p) / sqrt(p * s * (1 - s) * (1 - p))
