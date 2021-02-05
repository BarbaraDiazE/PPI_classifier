"""Perform SVM with descriptors"""

import os
import pandas as pd
from Kmeans import Kmeans

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info",
    "root_ROC": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/ROC",
    "trained_models": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/trained_models",
}


print(str(root["root"]))
data = pd.read_csv(str(root["root"]) + str("dataset_descriptors_p2.csv"))

numerical_data = data.drop(
    ["Unnamed: 0", "ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"],
    axis=1,
)
descriptors = numerical_data.columns.to_list()
print("number of descriptors: ", len(descriptors))


def execute(
    root, input_file, target, descriptors, fraction, kernel, balanced, ref_output
):
    a = Kmeans(root, input_file, target, descriptors, fraction)
    a.eda()
    a.train_model()
    # a.report(ref_output)
    # print("report ", str(ref_output), "is done")


execute(
    root,
    "dataset_descriptors_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "linear",
    "balanced",
    "p2_D11L6P3SVM1A",
)
# execute(
#     root,
#     "dataset_descriptors_p2.csv",
#     "PPI",
#     descriptors,
#     0.2,
#     "poly",
#     "balanced",
#     "p2_D11L6P3SVM2A",
# )
# execute(
#     root,
#     "dataset_descriptors_p2.csv",
#     "PPI",
#     descriptors,
#     0.2,
#     "rbf",
#     "balanced",
#     "p2_D11L6P3SVM3A",
# )
# execute(
#     root,
#     "dataset_descriptors_p2.csv",
#     "PPI",
#     descriptors,
#     0.2,
#     "sigmoid",
#     "balanced",
#     "p2_D11L6P3SVM4A",
# )
# execute(
#     root,
#     "dataset_descriptors_p2.csv",
#     "PPI",
#     descriptors,
#     0.3,
#     "linear",
#     "balanced",
#     "p2_D11L6P5SVM1A",
# )
# execute(
#     root,
#     "dataset_descriptors_p2.csv",
#     "PPI",
#     descriptors,
#     0.3,
#     "poly",
#     "balanced",
#     "p2_D11L6P5SVM2A",
# )
# execute(
#     root,
#     "dataset_descriptors_p2.csv",
#     "PPI",
#     descriptors,
#     0.3,
#     "rbf",
#     "balanced",
#     "p2_D11L6P5SVM3A",
# )
# execute(
#     root,
#     "dataset_descriptors_p2.csv",
#     "PPI",
#     descriptors,
#     0.3,
#     "sigmoid",
#     "balanced",
#     "p2_D11L6P5SVM4A",
# )

