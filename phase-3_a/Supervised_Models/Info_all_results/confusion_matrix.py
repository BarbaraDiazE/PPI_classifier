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
    """storage confusion matric info"""
    id_models = list()
    values = list()
    for i in range(len(arr)):
        id_models.append(find_model_id(arr[i]))  # storage id model
        df = pd.read_csv(arr[i])
        a = df.loc[2][2]
        # print(a, arr[i])
        if a == "linear":
            b = df.loc[18][2]
            values.append(b)
        else:
            b = df.loc[17][2]
            values.append(b)
    DF = pd.DataFrame.from_dict(
        {"id_model": id_models, "Exp": arr, "Confusion matrix": values}
    )
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


DF = storage_info(arr)
# print(DF)
# print(DF.head())
