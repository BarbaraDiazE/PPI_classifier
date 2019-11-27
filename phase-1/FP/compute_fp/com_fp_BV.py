"""Compute FPs as BitVect"""

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem,  MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs  
    
def morgan2(SMILES):
        ms = [Chem.MolFromSmiles(i) for i in SMILES]
        fp = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in ms]
        return fp

def morgan3(SMILES):
        ms=[Chem.MolFromSmiles(i) for i in SMILES]
        fp = [AllChem.GetMorganFingerprintAsBitVect(x,3) for x in ms]
        return fp

def topological(SMILES):
        ms =[Chem.MolFromSmiles(i) for i in SMILES]
        fp = [FingerprintMols.FingerprintMol(x) for x in ms]
        return fp

def maccskeys(SMILES):
        ms=[Chem.MolFromSmiles(i) for i in SMILES]
        fp = [MACCSkeys.GenMACCSKeys(x) for x in ms]
        return fp

from rdkit.Chem.AtomPairs import Pairs  
    
def atom(SMILES):
        ms = [Chem.MolFromSmiles(i) for i in SMILES]
        fp = [Pairs.GetAtomPairFingerprintAsIntVect(x) for x in ms]
        return fp

