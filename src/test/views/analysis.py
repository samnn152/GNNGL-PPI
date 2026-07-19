# pyright: reportUnknownMemberType=false
"""Plot analysis data produced by evaluation models."""

import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray
import numpy as np


class EvaluationAnalysisView:
    """Render evaluation matrices without participating in model computation."""

    @staticmethod
    def plot_correlation(correlation: NDArray[np.float64], output_path: str | None = None) -> None:
        """Render and optionally save a correlation heatmap."""
        labels = [str(index) for index in range(correlation.shape[0])]
        sns.set_theme(rc={'figure.figsize': (10, 8), 'figure.dpi': 200})
        heatmap = sns.heatmap(correlation, cmap='viridis', xticklabels=labels, yticklabels=labels)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0, horizontalalignment='right')
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
        if output_path is not None:
            plt.savefig(output_path)
        plt.close()
