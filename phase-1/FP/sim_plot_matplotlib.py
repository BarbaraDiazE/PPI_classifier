import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statistics as st
import itertools as it
#import random

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

root = {"root": "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/",
        "morgan2":"/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/morgan2/",
        "morgan3": "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/morgan3/",
        "maccskeys":"/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/maccskeys/",
        "atom": "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/atom"}

def morgan2_fp(SMILES, Library):
        ms = list()
        sim = list()
        y = list()
        ms=[Chem.MolFromSmiles(i) for i in SMILES]
        fps_Morgan = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in ms]
        Morgan = [DataStructs.FingerprintSimilarity(y,x) for x,y in it.combinations(fps_Morgan,2)]
        Morgan.sort()
        sim = Morgan
        #estatistical values
        stat = {"MIN": [round(min(sim),2)],
                "1Q": [round(np.percentile(sim, 25))],
                "MEDIAN": [round(st.median(sim))],
                "MEAN": [round(st.mean(sim),2)],
                "3Q": [round(np.percentile(sim, 75),2)],
                "MAX": [max(sim)],
                "STD": [round(st.stdev(sim),2)],
                "Library": [str(Library)] }
        df = pd.DataFrame.from_dict(stat)
        fp_result = {"sim" : sim,   
                    "y"  : np.arange(1, len(sim) + 1)/ len(sim),
                    "df" : df}
        return fp_result

def morgan3_fp(SMILES, Library):
        ms=[Chem.MolFromSmiles(i) for i in SMILES]
        fps_Morgan = [AllChem.GetMorganFingerprintAsBitVect(x,3) for x in ms]
        Morgan = [DataStructs.FingerprintSimilarity(y,x) for x,y in it.combinations(fps_Morgan,2)]
        Morgan.sort()
        sim = Morgan
        #estatistical values
        stat = {"MIN": [round(min(sim),2)],
                "1Q": [round(np.percentile(sim, 25))],
                "MEDIAN": [round(st.median(sim))],
                "MEAN": [round(st.mean(sim),2)],
                "3Q": [round(np.percentile(sim, 75),2)],
                "MAX": [max(sim)],
                "STD": [round(st.stdev(sim),2)],
                "Library": [str(Library)] }
        df = pd.DataFrame.from_dict(stat)
        fp_result = {"sim" : sim,   
                    "y"  : np.arange(1, len(sim) + 1)/ len(sim),
                    "df" : df}
        return fp_result
def maccskeys_fp(SMILES, Library):
        ms = list()
        sim = list()
        y = list()
        ms=[Chem.MolFromSmiles(i) for i in SMILES]
        fps_MACCKeys = [MACCSkeys.GenMACCSKeys(x) for x in ms]
        MACCKeys = [DataStructs.FingerprintSimilarity(y,x) for x,y in it.combinations(fps_MACCKeys,2)]
        MACCKeys.sort()
        sim = MACCKeys
        y= np.arange(1, len(sim) + 1)/ len(sim) #eje y#estatistical values
        stat = {"MIN": [round(min(sim),2)],
                "1Q": [round(np.percentile(sim, 25))],
                "MEDIAN": [round(st.median(sim))],
                "MEAN": [round(st.mean(sim),2)],
                "3Q": [round(np.percentile(sim, 75),2)],
                "MAX": [max(sim)],
                "STD": [round(st.stdev(sim),2)],
                "Library": [str(Library)] }
        df = pd.DataFrame.from_dict(stat)
        fp_result = {"sim" : sim,   
                    "y"  : np.arange(1, len(sim) + 1)/ len(sim),
                    "df" : df}
        return fp_result

class PlotSimPlt:
    
    def __init__(self, input_file):
        Data  = pd.read_csv(str(root["root"]) + str(input_file))
        self.Data = Data
        self.Data = pd.DataFrame.sample(self.Data, frac=0.5, replace=True,  random_state=1992, axis=None) 
        print(self.Data.columns)
        self.Library = self.Data.Library.unique()
        print(self.Library)

    def storage_sim_data(self):
        Library = self.Library
        sim_data = dict()
        for i in range(len(Library)):
            smiles = list(self.Data[self.Data["Library"] == Library[i]].SMILES)
            sim_data[Library[i]] = morgan3_fp(smiles, Library[i])
        return sim_data    
      
    def plot_sim(self, dict, ref_output):
        Library = self.Library
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
        Library = self.Library
        frames = [ dict[Library[0]]["df"], dict[Library[1]]["df"], dict[Library[2]]["df"], dict[Library[3]]["df"], dict[Library[4]]["df"]]
        DF = pd.concat(frames, axis = 0)
        print(DF)
        DF.to_csv("stats_" + str(ref_output) +  ".csv", sep = "," )    

a = PlotSimPlt("Dataset.csv")
sim_data = a.storage_sim_data()
a.plot_sim(sim_data, "ECFP6")
a.stats(sim_data, "ECFP6")