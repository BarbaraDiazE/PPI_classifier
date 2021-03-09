import pandas as pd
import numpy as np
import os
import seaborn as sns
import statistics as st
import itertools as it

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter  # useful for `logit` scale


"""RF Summary metrics
    get the id of the model
    get performence metrics and plot results in a headmap
"""

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/summary_metrics/"
}


def storage_info(file_name):
    _ = root["root"]
    DF = pd.read_csv(f"{_}{file_name}", index_col="Unnamed: 0")
    # DF = DF.sort_values("Precision", ascending=False)
    # DF = DF.reset_index(drop=True)
    print(DF.head())

    return DF


def plot(DF, output_figure):
    print(DF)
    DF = DF.set_index("id_model")
    # DF = DF.drop("Model name", axis=1)
    print(DF.head())
    plt.figure(figsize=[6.4, 4.8])
    ax = plt.subplot()
    ################3

    bounds = [x * 0.01 for x in range(2, 9)]
    bounds = [x + 0.9 for x in bounds]
    print(bounds)

    colors = [
        "yellow",
        "orange",
        "pink",
        "mediumvioletred",
        "yellowgreen",
        "deepskyblue",
        "dodgerblue",
    ]

    norm = plt.Normalize(bounds[0], bounds[-1])
    cmap = LinearSegmentedColormap.from_list(
        "", [(norm(b), c) for b, c in zip(bounds, colors)], N=256,
    )

    ###################
    sns.set_context("paper")
    sns.set_style("darkgrid")
    ax = sns.heatmap(
        data=DF,
        cmap=cmap,
        linewidth=2,
        annot=True,
        linecolor="ivory",
        cbar_kws={"shrink": 0.8},
        yticklabels=True,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=8)  #

    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel("Metrics", fontsize=14)
    plt.ylabel("Trained models", fontsize=14)
    # plt.title("Summary metrics", fontsize=14)
    ###modify bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout(h_pad=0.9)
    plt.savefig(
        f'{"/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/plot_metrics/"}{output_figure}',
        dpi=200,
    )
    plt.show()


DF = storage_info("metrics_best_models.csv")
plot(DF, "plot_metrics_best_models.png")
