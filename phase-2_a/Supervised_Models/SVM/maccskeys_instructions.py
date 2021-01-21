import os
import pandas as pd
from SVM_FP import SVM

# root = {"root": "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Databases/",
#        "root_Info" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/SVM/Info",
#        "root_ROC" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/SVM/ROC"}

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info",
    "root_ROC": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/ROC",
    "trained_models": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/trained_models",
}

"""MACCS"""
print(str(root["root"]))
Data = pd.read_csv(str(root["root"]) + str("dataset_maccskeys_p2.csv"))
ids = ["Unnamed: 0", "ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"]
numerical_data = Data.drop(ids, axis=1)
descriptors = numerical_data.columns.to_list()
print(len(descriptors))


def execute(
    root, input_file, target, descriptors, fraction, kernel, balanced, ref_output
):
    a = SVM(root, input_file, target, descriptors, fraction)
    a.train_model(kernel, balanced)
    a.report(ref_output)
    print("report ", str(ref_output), "is done")


execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "linear",
    "balanced",
    "p2_F3L6P3SVM1A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "poly",
    "balanced",
    "p2_F3L6P3SVM2A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "rbf",
    "balanced",
    "p2_F3L6P3SVM3A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "sigmoid",
    "balanced",
    "p2_F3L6P3SVM4A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "linear",
    "balanced",
    "p2_F3L6P5SVM1A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "poly",
    "balanced",
    "p2_F3L6P5SVM2A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "rbf",
    "balanced",
    "p2_F3L6P5SVM3A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "sigmoid",
    "balanced",
    "p2_F3L6P5SVM4A",
)
