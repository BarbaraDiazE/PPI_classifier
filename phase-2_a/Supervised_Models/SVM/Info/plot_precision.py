import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import statistics as st
import itertools as it

"""Plot Precision"""

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Databases/",
    "SVM": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/",
}
arr = list()
for file in os.listdir():
    if file.endswith(".csv"):
        arr.append(file)
DF = pd.DataFrame

# read id models csv
id_dataframe = pd.read_csv(f'{root["SVM"]}{"p2_id_models.csv"}', index_col="Unnamed: 0")
print(id_dataframe.head())


def find_model_id(model_name):
    "find the model id acording to model name"
    model_name = model_name.replace(".csv", "")
    _ = id_dataframe.loc[id_dataframe["model name"] == model_name, "id_model"]
    id_model = _.iloc[0]
    if len(id_model) == 3:
        label = id_model + "  "
    else:
        label = id_model
    return label


def storage_info(arr):
    """storage Precision"""
    values = list()
    id_models = list()
    for i in range(len(arr)):
        print("#####", i, arr[i], "#####")
        id_models.append(find_model_id(arr[i]))  # storage id model
        df = pd.read_csv(arr[i])
        a = df.loc[2][2]
        if a == "linear":
            b = df.iloc[14][2]
            values.append(float(b))
        else:
            b = df.iloc[13][2]
            values.append(float(b))
    DF = pd.DataFrame.from_dict(
        {"id_models": id_models, "Exp": arr, "Precision": values}
    )
    return DF


def plot_sim(DF):
    plt.figure(figsize=[10, 4.8], dpi=200)
    DF.to_csv(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/info_metrics/Precision.csv"
    )
    x = [i for i in range(DF.shape[0])]
    X = DF.id_models.to_list()  # xlabels
    y = list(DF["Precision"])
    plt.plot(x, y, "ro", color="teal", alpha=0.7)
    plt.grid(color="lightgray", axis="y", linestyle="dotted", linewidth=2)
    plt.xticks(x, X, rotation="vertical")
    # plt.tick_params(axis="x", labelsize=10)
    plt.ylabel("Precision")
    plt.ylim([0.5, 1.0])
    plt.subplots_adjust()
    plt.savefig(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/plot_metrics/precision.png",
        dpi=200,
    )
    plt.show()


DF = storage_info(arr)
# print(DF.max())
print(DF.sort_values(by=["Precision"]))
plot_sim(DF)
