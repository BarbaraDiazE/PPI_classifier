"""Compute Morgan2 FP"""

import numpy as np
import pandas as pd
import statistics as st
import itertools as it
import random

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
    
def morgan2_fp(SMILES, Library):
        ms = list()
        sim = list()
        y = list()
        random.seed(43)
        SMILES =round(len(SMILES)*.1)
        SMILES = random.sample(SMILES,N)
        ms=[Chem.MolFromSmiles(i) for i in SMILES]
        fps_Morgan = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in ms]
        Morgan = [DataStructs.FingerprintSimilarity(y,x) for x,y in it.combinations(fps_Morgan,2)]
        Morgan.sort()
        sim = Morgan    
        y = np.arange(1, len(sim) + 1)/ len(sim)
        #estadisticas
        stat = {"MIN": [round(min(sim),2)],
                "1Q": [round(np.percentile(sim, 25))],
                "MEDIAN": [round(st.median(sim))],
                "MEAN": [round(st.mean(sim),2)],
                "3Q": [round(np.percentile(sim, 75),2)],
                "MAX": [max(sim)],
                "STD": [round(st.stdev(sim),2)],
                "Library": [str(Library)] }
        df = pd.DataFrame.from_dict(stat)
        return sim, y, df

def morgan2(SMILES):
        ms = [Chem.MolFromSmiles(i) for i in SMILES]
        fp = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in ms]
        return fp
