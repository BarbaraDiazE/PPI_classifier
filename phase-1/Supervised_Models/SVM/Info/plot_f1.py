import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import statistics as st
import itertools as it

"""Plot F1"""

root = {"root": "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/",
        "SVM":"/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Supervised_Models/SVM"
        }
arr = os.listdir()
arr.remove("plot_balanced_accuracy.py")
arr.remove( 'Coeff')
arr.remove("plot_metrics")
arr.remove("plot_f1.py")
arr.remove( "readme.md")
print(arr)
DF = pd.DataFrame
def storage_info(arr):
    #ROC_info = dict()
    """storage balanced accuracy"""
    values = list()
    for i in range(len(arr)):
        df = pd.read_csv(arr[i])
        #print(df.index.tolist())#funciona
        #print(df.columns.tolist())#funciona
        #print(df)
        a = df.iloc[[1]]["SVM"].to_list()
        if str(a[0]) == "linear":
            #print(df.iloc[[14]]["SVM"])
            b = df.iloc[[14]]["SVM"].to_list()
            values.append(float(b[0]))
        else:
            #print(df.iloc[[13]]["SVM"])
            b = df.iloc[[13]]["SVM"].to_list()
            values.append(float(b[0]))
    DF = pd.DataFrame.from_dict({"Exp":arr, "F1": values})
    return DF
      
def plot_sim(DF):
    plt.figure(figsize = [10, 4.8], dpi = 200)
    DF.sort_values(by=["Exp"])
    X = list()
    for i in DF.Exp.to_list():
        i = i.replace(".csv", "")
        X.append(i.replace("SVM_", ""))
    x = [i for i in range(len(DF["F1"]))]
    y = list(DF["F1"])
    plt.plot(x, y, 'ro', color = "crimson")
    plt.xticks(x, X, rotation=  "vertical")
    plt.ylabel('F1')
    #plt.title(str(ref_output))
    plt.subplots_adjust(bottom=0.35)
    plt.show()

DF = storage_info(arr)
#print(DF)
plot_sim(DF)