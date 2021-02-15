import pandas as pd
import numpy as np
import os
import seaborn as sns
import statistics as st
import itertools as it

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from scripts_plots.LRG.LRG_filter_metrics import LRG_M


"""LRG Summary metrics
    get the id of the model
    get performence metrics and plot results in a headmap
"""

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/SVM/"
}


def storage_info():
    # precision
    x1, precision, X1 = LRG_M().precision()
    # balanced acc
    x2, balanced_accuracy, X2 = LRG_M().balanced_acc()
    # f1
    x3, f1, X3 = LRG_M().f1()
    # recall
    x4, recall, X4 = LRG_M().recall()
    DF = pd.DataFrame.from_dict(
        {
            "id_model": X1,
            # "Model name": arr,
            "Precision": precision,
            "Balanced accuracy": balanced_accuracy,
            "F1": f1,
            "Recall": recall,
        }
    )
    # print(DF.head())
    DF = DF.sort_values("Precision", ascending=False)
    DF = DF.reset_index(drop=True)
    print(DF.head())
    DF.to_csv(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/summary_metrics/LRG_summary_metrics.csv"
    )
    # print(DF.dtypes)
    return DF


def plot(DF, output_figure):
    print(DF)
    DF = DF.set_index("id_model")
    # DF = DF.drop("Model name", axis=1)
    print(DF.head())
    plt.figure(figsize=[8, 10])
    ax = plt.subplot()
    ################3

    bounds = [
        0.50,
        0.59,
        0.70,
        0.74,
        0.76,
        0.81,
        0.85,
        0.89,
        0.93,
        0.95,
        0.97,
        0.98,
        1.0,
    ]
    colors = [
        "yellow",
        "gold",
        "orange",
        "peru",
        "pink",
        "deeppink",
        "mediumvioletred",
        "blueviolet",
        # "indigo",
        "yellowgreen",
        "forestgreen",
        # "cyan",
        "deepskyblue",
        "dodgerblue",
        "darkblue",
    ]

    norm = plt.Normalize(bounds[0], bounds[-1])
    cmap = LinearSegmentedColormap.from_list(
        "", [(norm(b), c) for b, c in zip(bounds, colors)], N=256
    )

    ###################
    sns.set_context("paper")
    sns.set_style("darkgrid")
    ax = sns.heatmap(
        data=DF,
        cmap=cmap,
        linewidth=1,
        annot=True,
        linecolor="ivory",
        vmin=0.50,
        vmax=1.0,
        cbar_kws={"shrink": 0.8},
    )
    ax.tick_params(axis="both", labelsize=8)

    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel("Metrics", fontsize=10)
    plt.ylabel("Trained models", fontsize=10)
    plt.title("Summary metrics", fontsize=12)
    ###modify bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout(h_pad=0.9)
    plt.savefig(
        f'{"/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/plot_metrics/"}{output_figure}',
        dpi=200,
    )
    plt.show()


DF = storage_info()
# print(DF.head())
plot(DF, "LRG_metrics.png")
