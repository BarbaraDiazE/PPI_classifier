import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statistics as st
import itertools as it
"""Plot ROC by differents Kernels"""

root = {"root":"/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/morgan2/"
        }
class PlotROC:
    
    def storage_data(self, input_file):
        sim_data = dict()
        for i in range(len(input_file)):
            df = pd.read_csv(str(root["root"]) + str(input_file[i]))
            print(df.columns)
            #sim_data[Library[i]] = compute_fp(smiles, Library[i])
        #return sim_data    
      
    def plot_sim(self, dict, ref_output):
        plt.figure()
        lw = 2
        plt.plot(dict[Library[0]]["sim"], dict[Library[0]]["y"], color= "darkslategray", lw=lw, linestyle='-', label= Library[0])
        plt.plot(dict[Library[1]]["sim"], dict[Library[1]]["y"], color= "yellowgreen", lw=lw, linestyle='-', label = Library[1])
        plt.plot(dict[Library[2]]["sim"], dict[Library[2]]["y"], color= "darkturquoise", lw=lw, linestyle='-', label = Library[2])
        plt.plot(dict[Library[3]]["sim"], dict[Library[3]]["y"], color= "lightcoral", lw=lw, linestyle='-', label = Library[3])
        plt.plot(dict[Library[4]]["sim"], dict[Library[4]]["y"], color= "crimson", lw=lw, linestyle='-', label = Library[4])
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.xlabel('Similarity')
        plt.ylabel('CDF')
        plt.title(str(ref_output))
        plt.legend(loc = "lower right", ncol=1, shadow=False, fancybox=False)
        plt.show()

    def stats(self, dict, ref_output):
        #Library = self.Library
        frames = [ dict[Library[0]]["df"], dict[Library[1]]["df"], dict[Library[2]]["df"], dict[Library[3]]["df"], dict[Library[4]]["df"]]
        DF = pd.concat(frames, axis = 0)
        print(DF)
        DF.to_csv("stats_" + str(ref_output) +  ".csv", sep = "," )    

Files = ["ROC_data_F3L6P5SVM1A.csv", "ROC_data_F3L6P5SVM2A.csv"]
a = PlotROC()
a.storage_data(Files)
print("storage data")
#sim_data = a.storage_data(maccskeys_fp)
#a.plot_sim(sim_data, "MACCS KEYS")
#a.stats(sim_data, "MACCS KEYS")