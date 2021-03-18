"""
Compute bit count matrix
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


class Bit_Count_AtomPairs:
    def __init__(self):
        csv_name = "dataset_p2.csv"
        self.data = pd.read_csv(
            f'{"/home/babs/Desktop/Data_Phase_2/final_files/"}{csv_name}',
            index_col="Unnamed: 0",
        )
        self.data = self.data.sample(frac=0.3)
        # print(self.data.head())

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
        result_matrix = list()
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
            result_matrix.append(vect_rep)
            print(result_matrix)
            # self.result_matrix = result_matrix
            array_length = len(result_matrix)
            last_element = result_matrix[array_length - 1]
            input_arr = np.reshape(last_element, (1, len(last_element)))
            return input_arr

    # def write_output(self, output_file):
    #     explicit_vect = pd.DataFrame(
    #         data=self.result_matrix,
    #         # columns=[str(i) for i in range(self.result_matrix[1])],
    #         # index=[i for i in range(self.result_matrix[0])],
    #     )
    #     print(explicit_vect.head(3))
    #     print("column names", explicit_vect.columns)
    #     print(self.data.shape, explicit_vect.shape)
    #     frames = [self.data, explicit_vect]
    #     Final = pd.concat([self.data, explicit_vect], axis=1)
    #     print(Final.head(5))


# a = Bit_Count_AtomPairs()
# a.atom_pairs()
# a.write_output("dataset_atompairs_p2.csv")
