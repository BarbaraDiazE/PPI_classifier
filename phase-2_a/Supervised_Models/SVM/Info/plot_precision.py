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


def storage_info(arr):
    """storage Precision"""
    values = list()
    for i in range(len(arr)):
        print("#####", i, arr[i], "#####")
        df = pd.read_csv(arr[i])
        a = df.loc[2][2]
        print("kernel", a)
        if a == "linear":
            b = df.iloc[14][2]
            values.append(float(b))
        else:
            b = df.iloc[13][2]
            values.append(float(b))
    #         continue
    DF = pd.DataFrame.from_dict({"Exp": arr, "Precision": values})
    return DF


def rename_x(model_name):
    model_name = model_name.replace(".csv", "")
    _ = id_dataframe.loc[id_dataframe["model name"] == model_name, "id_model"]
    return _.iloc[0]


def unify_string_len(id_model):
    if len(id_model) == 3:
        label = id_model + "  "
    else:
        label = id_model
    return label


def plot_sim(DF):
    plt.figure(figsize=[10, 4.8], dpi=200)
    DF.to_csv(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/info_metrics/Precision.csv"
    )
    DF = DF.sort_values(by=["Exp"])
    # X = list()
    # for i in DF.Exp.to_list():
    #     X.append(i)

    x = [i for i in range(len(DF["Precision"]))]
    # xlabels
    X = DF.Exp.to_list()
    X = list(map(rename_x, X))
    X_label = list(map(unify_string_len, X))
    print("soy x", "\n", X_label)
    y = list(DF["Precision"])
    plt.plot(x, y, "ro", color="teal")
    plt.xticks(x, X_label, rotation="vertical")
    # plt.tick_params(axis="x", labelsize=10)
    plt.ylabel("Precision")
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(
        "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info/plot_metrics/precision.png",
        dpi=150,
    )
    plt.show()


DF = storage_info(arr)
print(DF.max())
print(DF.sort_values(by=["Precision"]))
plot_sim(DF)
