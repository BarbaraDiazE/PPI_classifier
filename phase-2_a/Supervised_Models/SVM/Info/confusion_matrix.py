import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import statistics as st
import itertools as it

"""Confusion matrix"""

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
    """storage balanced accuracy"""
    values = list()
    for i in range(len(arr)):
        df = pd.read_csv(arr[i])
        a = df.loc[2][2]
        # print(a, arr[i])
        if a == "linear":
            b = df.loc[18][2]
            values.append(b)
        else:
            b = df.loc[17][2]
            values.append(b)
    DF = pd.DataFrame.from_dict({"Exp": arr, "Confusion matrix": values})
    DF.to_csv(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/info_metrics/confusion_matrix.csv"
    )
    m = [
        "SVM_p2_F2L6P3SVM2ABN.csv",
        "SVM_p2_F1L6P3SVM2ABN.csv",
        "SVM_p2_F1L6P5SVM2ABN.csv",
        "SVM_p2_F2L6P3SVM3ABN.csv",
        "SVM_p2_F2L6P5SVM2ABN.csv",
    ]
    for i in m:
        print(DF[DF["Exp"] == i])
    return DF


def plot_sim(DF):
    plt.figure(figsize=[10, 4.8], dpi=200)
    DF.to_csv(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/info_metrics/f1.csv"
    )
    DF.sort_values(by=["Exp"])
    X = list()
    for i in DF.Exp.to_list():
        i = i.replace(".csv", "")
        X.append(i.replace("SVM_", ""))
    x = [i for i in range(len(DF["F1"]))]
    y = list(DF["F1"])
    plt.plot(x, y, "ro", color="crimson")
    plt.xticks(x, X, rotation="vertical")
    plt.ylabel("F1")
    plt.subplots_adjust(bottom=0.35)
    plt.show()


DF = storage_info(arr)
# print(DF)
# print(DF.head())
