import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from compute_fp.com_fp_BV import morgan2, morgan3, topological, maccskeys, atom

# import atompairs
from compute_fp.compute_atompairs_explicict_bits import Bit_Count_AtomPairs

"""
make prediction usig trained models"""
import pandas as pd
import numpy as np
from sklearn.externals import joblib


def test_ipp_compound(Data, ipp_id):

    test = Data[Data["ipp_id"] == ipp_id]
    test = test.drop(
        ["ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"], axis=1
    )
    return np.asarray(test)


def test_fda_compound(Data, chembl_id):
    test = Data[Data["chembl_id"] == chembl_id]
    test = test.drop(
        ["ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"], axis=1
    )

    return np.asarray(test)


class Ensemble:
    def __init__(self, smiles_molecule):
        self.smiles_molecule = [smiles_molecule]
        self.route = "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/models/"

    def fp_array(self, fp_func):
        fp = fp_func(self.smiles_molecule)
        output = []
        for f in fp:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(f, arr)
            output.append(arr)
        return np.asarray(output)

    def model1(self, fp_func):
        """return RF prediction"""
        # generate input descriptors
        x_test = self.fp_array(fp_func)
        # model1 RF
        model1 = joblib.load(f'{self.route}{"RF_p2_F1L6P3GRF3A.pkl"}')
        pred1 = model1.predict(x_test)
        return pred1[0]

    def model2(self, fp_func):
        """return SVM9 prediction"""
        # generate input descriptors
        x_test = self.fp_array(fp_func)
        # model2
        model2 = joblib.load(f'{self.route}{"SVM_p2_F1L6P5SVM1A.pkl"}')
        pred2 = model2.predict(x_test)
        # print("prediction atom", )
        return pred2[0]

    def model3(self, fp_func):
        """return SVM25 prediction"""
        # generate input descriptors
        x_test = self.fp_array(fp_func)
        # model3 SVM
        model1 = joblib.load(f'{self.route}{"SVM_p2_F2L6P5SVM1B.pkl"}')
        pred1 = model1.predict(x_test)
        return pred1[0]

    def score_ensemble(self):
        pred1 = self.model1(morgan2)
        pred2 = self.model2(morgan2)
        pred3 = self.model3(morgan3)
        score = pred1 + pred2 + pred3
        if score >= 2:
            return "Yes"
        else:
            return "No"


molecule_ppi_1 = (
    "Cc1cc2CN(CCc2c(C)c1C(=O)N[C@@H](CNC(=O)c1cccs1)C(O)=O)C(=O)c1ccc(Cl)cc1"
)
pred1 = Ensemble(molecule_ppi_1).model1(morgan2)
pred2 = Ensemble(molecule_ppi_1).model2(morgan2)
pred3 = Ensemble(molecule_ppi_1).model3(morgan3)

print(
    "Molecule: ",
    molecule_ppi_1,
    "\n" "Prediction 1: ",
    pred1,
    "\n" "Prediction 2: ",
    pred2,
    "\n" "Prediction 3: ",
    pred3,
    "\n",
)

print(Ensemble(molecule_ppi_1).score_ensemble())
# paracetamol = "CC(=O)NC1=CC=C(C=C1)O"
# pred1b = Ensemble(paracetamol).model1(morgan2)
# pred2b = Ensemble(paracetamol).model2(morgan2)
# pred3b = Ensemble(paracetamol).model3(morgan3)
# print(
#     "paracetamol: ",
#     paracetamol,
#     "\n" "Prediction 1: ",
#     pred1b,
#     "\n" "Prediction 2: ",
#     pred2b,
#     "\n" "Prediction 3: ",
#     pred3b,
#     "\n",
# )
# print()
# final_pred = np.array([])
# for i in range(0, len(x_test)):
#     final_pred = np.append(final_pred, [pred1[i], pred2[i], pred3[i]])

# etelcalcetide = "CC(C(=O)NC(CCCN=C(N)N)C(=O)N)NC(=O)C(CCCN=C(N)N)NC(=O)C(CCCN=C(N)N)NC(=O)C(CCCN=C(N)N)NC(=O)C(C)NC(=O)C(CSSCC(C(=O)O)N)NC(=O)C"
# pred1c = Ensemble(etelcalcetide).model1(morgan2)
# pred2c = Ensemble(etelcalcetide).model2(morgan2)
# pred3c = Ensemble(etelcalcetide).model3(morgan3)
# print(
#     "etelcalcetide: ",
#     etelcalcetide,
#     "\n" "Prediction 1: ",
#     pred1c,
#     "\n" "Prediction 2: ",
#     pred2c,
#     "\n" "Prediction 3: ",
#     pred3c,
#     "\n",
# )

# venetoclax = "CC1(CCC(=C(C1)C2=CC=C(C=C2)Cl)CN3CCN(CC3)C4=CC(=C(C=C4)C(=O)NS(=O)(=O)C5=CC(=C(C=C5)NCC6CCOCC6)[N+](=O)[O-])OC7=CN=C8C(=C7)C=CN8)C"
# pred1d = Ensemble(venetoclax).model1(morgan2)
# pred2d = Ensemble(venetoclax).model2(morgan2)
# pred3d = Ensemble(venetoclax).model3(morgan3)
# print(
#     "venetoclax: ",
#     venetoclax,
#     "\n" "Prediction 1: ",
#     pred1d,
#     "\n" "Prediction 2: ",
#     pred2d,
#     "\n" "Prediction 3: ",
#     pred3d,
#     "\n",
# )
