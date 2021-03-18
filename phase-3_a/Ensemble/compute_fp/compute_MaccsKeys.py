"""Compute MACCS Keys FP"""

import numpy as np
import itertools as it
import random

import rdkit    
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols

def maccskeys_fp(Library):
        ms = list()
        sim = list()
        y = list()
        random.seed(43)
        N=round(len(Library)*.2)
        X = random.sample(Library,N)
        ms=[Chem.MolFromSmiles(i) for i in X]
        fps_MACCKeys = [MACCSkeys.GenMACCSKeys(x) for x in ms]
        MACCKeys = [DataStructs.FingerprintSimilarity(y,x) for x,y in it.combinations(fps_MACCKeys,2)]
        MACCKeys.sort()
        sim = MACCKeys    #similitud
        y= np.arange(1, len(sim) + 1)/ len(sim) #eje y
        return sim, y

def maccskeys(SMILES):
        ms=[Chem.MolFromSmiles(i) for i in SMILES]
        fp = [MACCSkeys.GenMACCSKeys(x) for x in ms]
        return fp
