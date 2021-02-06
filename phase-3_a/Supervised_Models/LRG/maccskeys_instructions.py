"Instructions to train LRG from MACCSKeys representation"

import os
import pandas as pd
from LRG import LRG

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results",
    "root_ROC": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/ROC",
    "trained_models": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/models",
}

"""MACCS"""
print(str(root["root"]))
Data = pd.read_csv(str(root["root"]) + str("dataset_maccskeys_p2.csv"))
ids = ["Unnamed: 0", "ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"]
numerical_data = Data.drop(ids, axis=1)
descriptors = numerical_data.columns.to_list()
print(len(descriptors))


def execute(
    root, input_file, target, descriptors, fraction, solver, balanced, ref_output
):
    a = LRG(root, input_file, target, descriptors, fraction)
    a.train_model(solver, balanced)
    a.report(ref_output)
    print("report ", str(ref_output), "is done")


execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "newton-cg",
    "balanced",
    "p2_F3L6P3LRG1A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "lbfgs",
    "balanced",
    "p2_F3L6P3LRG2A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "liblinear",
    "balanced",
    "p2_F3L6P3LRG3A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "sag",
    "balanced",
    "p2_F3L6P3LRG4A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "saga",
    "balanced",
    "p2_F3L6P3LRG5A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "newton-cg",
    "balanced",
    "p2_F3L6P5LRG1A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "lbfgs",
    "balanced",
    "p2_F3L6P5LRG2A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "liblinear",
    "balanced",
    "p2_F3L6P5LRG3A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "sag",
    "balanced",
    "p2_F3L6P5LRG4A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "saga",
    "balanced",
    "p2_F3L6P5LRG5A",
)
