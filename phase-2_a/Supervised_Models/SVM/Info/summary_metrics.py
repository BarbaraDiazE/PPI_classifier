import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
import itertools as it

"""Recall"""

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Databases/",
    "SVM": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/",
}
arr = list()
for file in os.listdir():
    if file.endswith(".csv"):
        arr.append(file)
DF = pd.DataFrame


def storage_info(arr):
    """storage recall"""
    balanced_accuracy = list()
    precision = list()
    f1 = list()
    recall = list()
    for i in range(len(arr)):
        print(arr[i])
        df = pd.read_csv(arr[i])
        a = df.loc[2][2]
        if a == "linear":
            balanced_accuracy.append(round(float(df.iloc[13][2]), 2))
            precision.append(round(float(df.iloc[14][2]), 2))
            f1.append(round(float(df.loc[15][2]), 2))
            recall.append(round(float(df.loc[19][2]), 2))
        else:
            balanced_accuracy.append(round(float(df.iloc[12][2]), 2))
            precision.append(round(float(df.iloc[13][2]), 2))
            f1.append(round(float(df.loc[14][2]), 2))
            recall.append(round(float(df.loc[18][2]), 2))
    DF = pd.DataFrame.from_dict(
        {
            "Model": arr,
            "Balanced accuracy": balanced_accuracy,
            "Precision": precision,
            "F1": f1,
            "Recall": recall,
        }
    )
    DF = DF.sort_values("Balanced accuracy", ascending=False)
    DF = DF.reset_index(drop=True)
    DF.to_csv(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/info_metrics/summary_metrics.csv"
    )
    print(DF.dtypes)
    return DF


def plot(DF):
    DF = DF.set_index("Model")
    print(DF.head())
    plt.figure(figsize=[8, 10])
    ax = plt.subplot()
    sns.set_context("paper")
    sns.set_style("darkgrid")
    ax = sns.heatmap(
        data=DF,
        cmap="tab20",
        linewidth=1,
        annot=False,
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
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/plot_metrics/headmap_summary.png",
        dpi=200,
    )
    plt.show()


DF = storage_info(arr)
# print(DF.head())
plot(DF)
