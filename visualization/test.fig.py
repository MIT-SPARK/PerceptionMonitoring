#!/usr/bin/env python3

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from adjustText import adjust_text
from diagnosability.dataset.diagnosability_dataset import DiagnosabilityDataset
from diagnosability.evaluator import DiagnosticTestsEvaluator, Stats
from rich import print
from diagnosability.perception_system import System
from diagnosability.factors import TestFactor


def main(dataset_path):
    dataset = DiagnosabilityDataset.load(dataset_path)
    cm = DiagnosticTestsEvaluator.evaluate(dataset)
    stats = {t: Stats.all(cm[t]) for t in cm}
    tests = {
        phi.test.varname: phi.test.name.replace("obstacles_", "").replace("_", " ")
        for phi in dataset.model.get_factors(lambda f: isinstance(f, TestFactor))
    }
    # PLOT
    with plt.style.context(["science"]):
        fig = plt.figure(figsize=(10, 3))
        ax = plt.gca()
        # Regular
        xx, yy, labels = [], [], []
        for varname, stat in stats.items():
            xx.append(stat["precision"] * 100)
            yy.append(stat["recall"] * 100)
            labels.append(tests[varname])
        texts = [
            ax.text(x, y, label, fontsize=12) for label, x, y in zip(labels, xx, yy)
        ]
        ax.scatter(xx, yy, marker="o", s=60)
        ax.set_xlabel("Precision (\%)")
        ax.set_ylabel("Recall (\%)")
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", lw=1),
            ax=ax,
        )
    fig.savefig(f"temporary/tests.pdf")


if __name__ == "__main__":
    dataset_path = "/home/antonap/sparklab/dataset/aij/regular/train.pkl"
    main(dataset_path)
