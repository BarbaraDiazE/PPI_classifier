import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import statistics as st
import itertools as it

"""Plot balanced accuracy"""

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
        print("#####", i, arr[i], "#####")
        df = pd.read_csv(arr[i])
        a = df.loc[2][2]
        print("kernel", a)
        if a == "linear":
            b = df.iloc[13][2]
            values.append(float(b))
        else:
            b = df.iloc[12][2]
            values.append(float(b))
    #         continue
    DF = pd.DataFrame.from_dict({"Exp": arr, "balanced accuracy": values})
    return DF


def plot_sim(DF):
    plt.figure(figsize=[10, 4.8], dpi=200)
    DF.to_csv(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/info_metrics/balanced_accuracy.csv"
    )
    DF = DF.sort_values(by=["Exp"])
    X = list()
    for i in DF.Exp.to_list():
        i = i.replace(".csv", "")
        X.append(i.replace("SVM_", ""))
    x = [i for i in range(len(DF["balanced accuracy"]))]
    y = list(DF["balanced accuracy"])
    plt.plot(x, y, "ro", color="navy")
    plt.xticks(x, X, rotation="vertical")
    plt.tick_params(axis="x", labelsize=10)
    plt.ylabel("Balanced Accuracy")
    # plt.title(str(ref_output))
    plt.subplots_adjust(bottom=0.15)
    plt.show()


DF = storage_info(arr)
print(DF.max())
print(DF.sort_values(by=["balanced accuracy"]))
plot_sim(DF)
