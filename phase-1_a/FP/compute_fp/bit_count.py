"""
Generate bit count matrix for a library
"""
import pandas as pd
import numpy as np

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem.AtomPairs import Pairs


def fp_matrix(fp):
    matrix_fp = []
    for f in fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        matrix_fp.append(arr)
    return matrix_fp


class Bit_Count_AtomPair:
    def __init__(self, route, csv_name):
        data = pd.read_csv(route + csv_name, index_col="Unnamed: 0")
        # print(data.columns)
        print(data.head())
        self.data = data

    def atom_pairs(self):
        ms = np.array([Chem.MolFromSmiles(i) for i in self.data.SMILES])
        # compute Atom Pair
        fp = [
            Pairs.GetAtomPairFingerprint(Chem.RemoveHs(x)).GetNonzeroElements()
            for x in ms
        ]
        # obtain all bits present
        bits_ap = set()
        for i in fp:
            bits_ap.update([*i])  # add bits for each molecule
        bits_ap = sorted(bits_ap)
        feature_matrix = list()
        # convert fp to bits
        for item in fp:
            vect_rep = np.isin(
                bits_ap, [*item]
            )  # vect_rep, var that indicates bits presents
            # identify axis to replace
            ids_to_update = np.where(vect_rep == True)
            vect_rep = 1 * vect_rep
            vect_rep = np.array(vect_rep).astype(int)
            # replace indices with bict values
            vect_rep[ids_to_update] = list(item.values())
            feature_matrix.append(vect_rep)
        return feature_matrix

    def ecfp6(self):
        ms = np.array([Chem.MolFromSmiles(i) for i in self.data.SMILES])
        fp = [Chem.AllChem.GetMorganFingerprintAsBitVect(x, 3) for x in ms]
        feature_matrix = fp_matrix(fp)
        return feature_matrix

    def feature_matrix(self, output_name):
        fp_matrix = self.atom_pairs()
        print(fp_matrix)
        fp = pd.DataFrame(fp_matrix)
        fp = fp.astype(int)
        print(fp.head())
        features = [
            "ID Database",
            "Name",
            "SMILES",
            "subLibrary",
            "Library",
            "PPI",
            "Epigenetic",
        ]
        ref_id = self.data[features].reset_index()
        frames = [ref_id, fp]
        print(ref_id.shape, fp.shape)
        result = pd.concat(frames, axis=1)
        result = result.drop("index", axis=1)
        result.to_csv(
            "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/"
            + output_name,
            sep=",",
            index=True,
        )
        print(result.info())
        print(result.head())
        # return feature_matrix, ref_id


route = "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/"
csv_name = "ECFP6_L6.csv"

a = Bit_Count_AtomPair(route, csv_name)
a.feature_matrix("AtomPairs_L6.csv")
