import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

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
    values = list()
    for i in range(len(arr)):
        print(arr[i])
        df = pd.read_csv(arr[i])
        a = df.loc[2][2]
        # print(a, arr[i])
        if a == "linear":
            b = df.loc[19][2]
            values.append(float(b))
        else:
            b = df.loc[18][2]
            values.append(float(b))
    DF = pd.DataFrame.from_dict({"Exp": arr, "Recall": values})
    DF.to_csv(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/info_metrics/recall.csv"
    )
    return DF


def plot_sim(DF):
    plt.figure(figsize=[10, 4.8], dpi=200)
    DF.to_csv(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/info_metrics/recall.csv"
    )
    DF.sort_values(by=["Exp"])
    X = list()
    for i in DF.Exp.to_list():
        i = i.replace(".csv", "")
        X.append(i.replace("SVM_", ""))
    x = [i for i in range(len(DF["Recall"]))]
    y = list(DF["Recall"])
    plt.plot(x, y, "ro", color="steelblue")
    plt.xticks(x, X, rotation="vertical")
    plt.ylabel("Recall")
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/plot_metrics/recall.png",
        dpi=150,
    )
    plt.show()


DF = storage_info(arr)
print(DF.head())
plot_sim(DF)
