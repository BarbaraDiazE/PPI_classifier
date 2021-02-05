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
    print(type(test))
    return np.asarray(test)


def test_fda_compound(Data, chembl_id):
    test = Data[Data["chembl_id"] == chembl_id]
    test = test.drop(
        ["ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"], axis=1
    )

    return np.asarray(test)


Data = pd.read_csv(
    "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Databases/dataset_maccskeys_p2.csv",
    index_col="Unnamed: 0",
)
# print(Data.head(2))

test = test_ipp_compound(Data, 282)
test_2 = test_fda_compound(Data, "Ospemifene")

model = joblib.load(
    "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/trained_models/p2_F3L6P3SVM1A.pkl"
)
print(model.predict(test))
print(model.predict(test_2))
