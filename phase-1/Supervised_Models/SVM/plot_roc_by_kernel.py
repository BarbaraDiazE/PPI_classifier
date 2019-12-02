import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statistics as st
import itertools as it

"""Plot ROC by KERNEL"""

root = {"root": "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/",
        "morgan2":"/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/morgan2/",
        "morgan3": "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/morgan3/",
        "maccskeys":"/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/maccskeys/",
        "atom": "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/atom"}

class PlotROC:
    
    def __init__(self, df1, df2, df3, df4):
        Data  = pd.read_csv(str(root["root"]) + str(input_file))
        self.Data = Data
        self.df1 = pd.DataFrame.sample(self.Data, frac=0.5, replace=True,  random_state=1992, axis=None) 
        self.df2 = pd.DataFrame.sample(self.Data, frac=0.5, replace=True,  random_state=1992, axis=None) 
        self.df3 = pd.DataFrame.sample(self.Data, frac=0.5, replace=True,  random_state=1992, axis=None) 
        self.df4 = pd.DataFrame.sample(self.Data, frac=0.5, replace=True,  random_state=1992, axis=None) 
        
    def storage_ROC_info(self, compute_fp):
        Library = self.Library
        ROC_info = dict()
        for i in range(len(Library)):
            smiles = list(self.Data[self.Data["Library"] == Library[i]].SMILES)
            #ROC_info[Library[i]] = morgan3_fp(smiles, Library[i])
            ROC_info[Library[i]] = compute_fp(smiles, Library[i])
        return ROC_info    
      
    def plot_sim(self, dict, ref_output):
        plt.figure()
        lw = 2
        plt.plot(dict[Library[0]]["sim"], dict[Library[0]]["y"], color= "darkslategray", lw=lw, linestyle='-', label= "Linear")
        plt.plot(dict[Library[1]]["sim"], dict[Library[1]]["y"], color= "yellowgreen", lw=lw, linestyle='-', label = "Poly")
        plt.plot(dict[Library[2]]["sim"], dict[Library[2]]["y"], color= "darkturquoise", lw=lw, linestyle='-', label = "rgb")
        plt.plot(dict[Library[3]]["sim"], dict[Library[3]]["y"], color= "lightcoral", lw=lw, linestyle='-', label = "Sigmoid")
        #plt.plot(dict[Library[4]]["sim"], dict[Library[4]]["y"], color= "crimson", lw=lw, linestyle='-', label = Library[4]
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.xlabel('Similarity')
        plt.ylabel('CDF')
        plt.title(str(ref_output))
        plt.legend(loc = "lower right", ncol=1, shadow=False, fancybox=False)
        plt.show()

    def stats(self, dict, ref_output):
        Library = self.Library
        frames = [ dict[Library[0]]["df"], dict[Library[1]]["df"], dict[Library[2]]["df"], dict[Library[3]]["df"], dict[Library[4]]["df"]]
        DF = pd.concat(frames, axis = 0)
        print(DF)
        DF.to_csv("stats_" + str(ref_output) +  ".csv", sep = "," )    

a = PlotSimPlt("Dataset.csv")
ROC_info = a.storage_ROC_info(maccskeys_fp)
a.plot_sim(ROC_info, "MACCS KEYS")
a.stats(ROC_info, "MACCS KEYS")