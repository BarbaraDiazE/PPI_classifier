"Instructions to train Random  forest from MACCSKeys representation"

import os
import pandas as pd
from RF import RF

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results",
    "root_ROC": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/ROC",
    "trained_models": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/models",
}

"""MACCS"""
print(str(root["root"]))
Data = pd.read_csv(
    str(root["root"]) + str("dataset_maccskeys_p2.csv"), index_col="Unnamed: 0"
)
ids = ["ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"]
numerical_data = Data.drop(ids, axis=1)
descriptors = numerical_data.columns.to_list()
print(len(descriptors))


def execute(
    root,
    input_file,
    target,
    descriptors,
    fraction,
    n_estimators,
    criterion,
    max_depth,
    balanced,
    ref_output,
):
    a = RF(root, input_file, target, descriptors, fraction)
    a.train_model(n_estimators, criterion, max_depth, balanced)
    a.report(ref_output)
    print("report ", str(ref_output), "is done")


# test set 0.2
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    100,
    "gini",
    None,
    "balanced",
    "p2_F3L6P3GRF1A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    100,
    "entropy",
    None,
    "balanced",
    "p2_F3L6P3GRF1A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    100,
    "gini",
    None,
    None,
    "p2_F3L6P3GRF1B",
)
# nstimators 500
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    500,
    "gini",
    None,
    "balanced",
    "p2_F3L6P3GRF2A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    500,
    "entropy",
    None,
    "balanced",
    "p2_F3L6P3ERF2A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    500,
    "gini",
    None,
    None,
    "p2_F3L6P3GRF2B",
)
# nstimators 1000
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    1000,
    "gini",
    None,
    "balanced",
    "p2_F3L6P3GRF3A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    1000,
    "entropy",
    None,
    "balanced",
    "p2_F3L6P3ERF3A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.2,
    1000,
    "gini",
    None,
    None,
    "p2_F3L6P3GRF3B",
)  # test set 0.3
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    100,
    "gini",
    None,
    "balanced",
    "p2_F3L6P5GRF1A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    100,
    "entropy",
    None,
    "balanced",
    "p2_F3L6P5GRF1A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    100,
    "gini",
    None,
    None,
    "p2_F3L6P5GRF1B",
)
# nstimators 500
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    500,
    "gini",
    None,
    "balanced",
    "p2_F3L6P5GRF2A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    500,
    "entropy",
    None,
    "balanced",
    "p2_F3L6P5ERF2A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    500,
    "gini",
    None,
    None,
    "p2_F3L6P5GRF2B",
)
# nstimators 1000
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    1000,
    "gini",
    None,
    "balanced",
    "p2_F3L6P5GRF3A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    1000,
    "entropy",
    None,
    "balanced",
    "p2_F3L6P5ERF3A",
)
execute(
    root,
    "dataset_maccskeys_p2.csv",
    "PPI",
    descriptors,
    0.3,
    1000,
    "gini",
    None,
    None,
    "p2_F3L6P5GRF3B",
)
