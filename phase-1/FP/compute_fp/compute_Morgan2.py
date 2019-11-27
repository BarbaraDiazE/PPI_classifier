"""Compute Morgan2 FP"""

import numpy as np
import itertools as it
import random

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
    
def morgan2_fp(Library):
        ms = list()
        sim = list()
        y = list()
        random.seed(43)
        N=round(len(Library)*.2)
        X = random.sample(Library,N)
        ms=[Chem.MolFromSmiles(i) for i in X]
        fps_Morgan = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in ms]
        Morgan = [DataStructs.FingerprintSimilarity(y,x) for x,y in it.combinations(fps_Morgan,2)]
        Morgan.sort()
        sim = Morgan    
        y= np.arange(1, len(sim) + 1)/ len(sim)
        return sim, y

def morgan2(SMILES):
        ms = [Chem.MolFromSmiles(i) for i in SMILES]
        fp = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in ms]
        return fp
