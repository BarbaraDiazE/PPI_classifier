"""Compute Topological FP similarity"""

import numpy as np
import itertools as it
import random
import itertools as it

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
    
def topological_fp(Library):
        ms = list()
        sim = list()
        y = list()
        random.seed(43)
        N =round(len(Library)*.2)
        X = random.sample(Library,N)
        ms =[Chem.MolFromSmiles(i) for i in X]
        fps_Topological = [FingerprintMols.FingerprintMol(x) for x in ms]
        Topological = [DataStructs.FingerprintSimilarity(y,x) for x,y in it.combinations(fps_Topological,2)]
        Topological.sort()
        sim = Topological    
        y = np.arange(1, len(sim) + 1)/len(sim) 
        return sim, y

def topological(SMILES):
        ms =[Chem.MolFromSmiles(i) for i in SMILES]
        fp = [FingerprintMols.FingerprintMol(x) for x in ms]
        return fp
