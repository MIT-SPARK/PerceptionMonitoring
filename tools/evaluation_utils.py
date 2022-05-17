from torchmetrics import ConfusionMatrix, HammingDistance
from diagnosability.base import DiagnosticModel
from diagnosability.evaluator import ModelEvaluator, Stats
from diagnosability.perception_system import System
from dataclasses import dataclass
import pandas as pd


@dataclass
class ModelWithResults:
    model: DiagnosticModel
    results: pd.DataFrame


def hamming_distance(empirical_hamming_distance, num_failure_modes):
    return (
        num_failure_modes
        * empirical_hamming_distance.flatten().cpu().detach().numpy()[0]
    )

def evaluate_on_dataset(system, models, dataset, ignore_modules=False):
    evaluator = ModelEvaluator(
        system=system,
        metrics=[
            lambda: ConfusionMatrix(
                num_classes=2, normalize=None, compute_on_step=False
            ),
            HammingDistance,
        ],
    )
    # Run Evaluator
    eval = dict()
    for mdl_name, mdl in models.items():
        results = None
        model = mdl
        if isinstance(mdl, ModelWithResults):
            model = mdl.model
            results = mdl.results
        elif isinstance(mdl, DiagnosticModel):
            pass
        else:
            raise ValueError(
                "Model must be either DiagnosticModel or ModelWithResults"
            )
        eval[mdl_name] = evaluator.evaluate(
            model,
            dataset,
            results=results,
            ignore_modules=ignore_modules,
        )
    # Post process
    num_fm_modules = len(
        dataset.model.system.get_failure_modes(System.Filter.MODULE_ONLY)
    )
    num_fm_outputs = len(
        dataset.model.system.get_failure_modes(System.Filter.OUTPUT_ONLY)
    )
    stats = {k:{"modules": dict(), "outputs": dict()} for k in eval.keys()}
    if not ignore_modules:
        for mdl_name, mdl_stats in eval.items():
            stats[mdl_name]["modules"] = Stats.all(mdl_stats["modules"]["ConfusionMatrix"])
            stats[mdl_name]["modules"]["HammingDistance"] = hamming_distance(
                mdl_stats["modules"]["HammingDistance"], num_fm_modules
            )
    for mdl_name, mdl_stats in eval.items():
        stats[mdl_name]["outputs"] = Stats.all(mdl_stats["outputs"]["ConfusionMatrix"])
        stats[mdl_name]["outputs"]["HammingDistance"] = hamming_distance(
            mdl_stats["outputs"]["HammingDistance"], num_fm_outputs
        ) 
    return stats
