import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statistics as st
import itertools as it
"""Plot ROC by differents Kernels"""

root = {"root":"/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Supervised_Models/SVM/ROC/"
        }
class PlotROC:
    
    def storage_data(self, input_file):
        data = dict()
        for i in range(len(input_file)):
            df = pd.read_csv(str(root["root"]) + str(input_file[i]))
            data[input_file[i]] = df
        return data    
      
    def plot_sim(self,input_file, dict, ref_output):
        plt.figure()
        lw = 2
        plt.plot(dict[input_file[0]]["fpr"], dict[input_file[0]]["tpr"], color= "darkslategray", lw=lw, linestyle='-', label= "Linear")
        plt.plot(dict[input_file[1]]["fpr"], dict[input_file[1]]["tpr"], color= "yellowgreen", lw=lw, linestyle='-', label = "Poly")
        plt.plot(dict[input_file[2]]["fpr"], dict[input_file[2]]["tpr"], color= "darkturquoise", lw=lw, linestyle='-', label = "RBF")
        plt.plot(dict[input_file[3]]["fpr"], dict[input_file[3]]["tpr"], color= "crimson", lw=lw, linestyle='-', label = "Sigmoid")
        #plt.plot(dict[input_file[4]]["fpr"], dict[input_file[4]]["tpr"], color= "crimson", lw=lw, linestyle='-', label = input_file[4])
        #plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(ref_output)
        plt.legend(loc="lower right")
        plt.savefig(str(root["root"]) + "ROC_General"+ str(ref_output) + ".png")
        plt.show()

Files = ["ROC_data_F2L6P5SVM1A.csv", "ROC_data_F2L6P5SVM2A.csv", "ROC_data_F2L6P5SVM3A.csv", "ROC_data_F2L6P5SVM4A.csv"]
a = PlotROC()
data = a.storage_data(Files)
a.plot_sim(Files, data, "ECFP6_P5")
