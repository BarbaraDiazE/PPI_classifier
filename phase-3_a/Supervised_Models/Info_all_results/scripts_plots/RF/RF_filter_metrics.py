import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import statistics as st
import itertools as it

"""Plot balanced accuracy"""

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/info_metrics/",
    "RF": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/RF/",
}
arr = list()
for file in os.listdir():
    if file.endswith(".csv"):
        arr.append(file)
DF = pd.DataFrame

# read id models csv


class RF_M:
    def __init__(self):
        self.id_dataframe = pd.read_csv(
            f'{root["root"]}{"RF_id_models.csv"}', index_col="Unnamed: 0"
        )
        print(self.id_dataframe.head())
        self.arr = [i + ".csv" for i in self.id_dataframe["Model name"]]
        # print(self.arr)

    def find_model_id(self, model_name):
        id_dataframe = self.id_dataframe
        "find the model id acording to model name"
        model_name = model_name.replace(".csv", "")
        _ = id_dataframe.loc[id_dataframe["Model name"] == model_name, "ID model"]
        id_model = _.iloc[0]
        if len(id_model) == 3:
            label = id_model + "  "
        else:
            label = id_model
        return label

    def precision(self):
        arr = self.arr
        values = list()
        id_models = list()
        for i in range(len(arr)):
            # print("#####", i, arr[i], "#####")
            id_models.append(self.find_model_id(arr[i]))
            df = pd.read_csv(f'{root["RF"]}{arr[i]}')
            # print(df)
            print(df.iloc[9])
            b = df.iloc[9][2]
            values.append(float(b))
        #         continue
        DF = pd.DataFrame.from_dict(
            {"id_models": id_models, "Exp": arr, "precision": values}
        )
        # plt.figure(figsize=[10, 4.8], dpi=200)
        x = [i for i in range(len(DF["precision"]))]
        X = DF.id_models.to_list()  # xlabels
        y = list(DF["precision"])

        return x, y, X

    def balanced_acc(self):
        arr = self.arr
        values = list()
        id_models = list()
        for i in range(len(arr)):
            # print("#####", i, arr[i], "#####")
            id_models.append(self.find_model_id(arr[i]))
            df = pd.read_csv(f'{root["RF"]}{arr[i]}')
            print(df.iloc[8])
            b = df.iloc[8][2]
            values.append(float(b))
        DF = pd.DataFrame.from_dict(
            {"id_models": id_models, "Exp": arr, "balanced accuracy": values}
        )
        x = [i for i in range(len(DF["balanced accuracy"]))]
        X = DF.id_models.to_list()  # xlabels
        y = list(DF["balanced accuracy"])
        return x, y, X

    def f1(self):
        arr = self.arr
        values = list()
        id_models = list()
        for i in range(len(arr)):
            # print("#####", i, arr[i], "#####")
            id_models.append(self.find_model_id(arr[i]))
            df = pd.read_csv(f'{root["RF"]}{arr[i]}')
            print(df.iloc[10])
            b = df.iloc[10][2]
            values.append(float(b))
        DF = pd.DataFrame.from_dict({"id_models": id_models, "Exp": arr, "F1": values})
        # plt.figure(figsize=[10, 4.8], dpi=200)
        x = [i for i in range(len(DF["F1"]))]
        X = DF.id_models.to_list()  # xlabels
        y = list(DF["F1"])

        return x, y, X

    def recall(self):
        arr = self.arr
        values = list()
        id_models = list()
        for i in range(len(arr)):
            # print("#####", i, arr[i], "#####")
            id_models.append(self.find_model_id(arr[i]))
            df = pd.read_csv(f'{root["RF"]}{arr[i]}')
            print(df.iloc[13])
            b = df.iloc[13][2]
            values.append(round(float(b), 2))
        #         continue
        DF = pd.DataFrame.from_dict(
            {"id_models": id_models, "Exp": arr, "recall": values}
        )
        # plt.figure(figsize=[10, 4.8], dpi=200)
        x = [i for i in range(len(DF["recall"]))]
        X = DF.id_models.to_list()  # xlabels
        y = list(DF["recall"])

        return x, y, X


a = RF_M()
# x, y, X = a.precision()

# x, y, X = a.balanced_acc()
# x, y, X = a.f1()
x, y, X = a.recall()
print(y)
