"""Compute AtomPair FP"""

import numpy as np
import itertools as it
import random

import rdkit    
from rdkit import Chem, DataStructs
from rdkit.Chem.AtomPairs import Pairs  
    
def atom_fp(Library):
        ms = list()
        sim = list()
        y = list()
        random.seed(43)
        N=round(len(Library)*.2)
        X = random.sample(Library,N)
        ms=[Chem.MolFromSmiles(i) for i in X]
        fps_atom = [Pairs.GetAtomPairFingerprintAsBitVect(x) for x in ms]
        Atom = [DataStructs.FingerprintSimilarity(y,x) for x,y in it.combinations(fps_atom,2)]
        Atom.sort()
        sim = Atom    
        y= np.arange(1, len(sim) + 1)/ len(sim)
        return sim, y
    
def atom(SMILES):
        ms = [Chem.MolFromSmiles(i) for i in SMILES]
        fp = [Pairs.GetAtomPairFingerprintAsIntVect(x) for x in ms]
        return fp
