"""Perform SVM with descriptors"""

import os
import pandas as pd
from tSNE_FP import TSNE_FP

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info",
    "root_chem_space": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/chemical_space",
    "tsne_results": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/tSNE/tSNE_results",
    "tsne_info_params": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/tSNE/tSNE_results/info_params",
}


print(str(root["root"]))
data = pd.read_csv(str(root["root"]) + str("dataset_ecfp4_p2.csv"))

numerical_data = data.drop(
    ["Unnamed: 0", "ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"],
    axis=1,
)
descriptors = numerical_data.columns.to_list()
print("number of descriptors: ", len(descriptors))


def execute(root, input_file, target, descriptors, ref_output):
    a = TSNE_FP(root, input_file, target, descriptors)
    # a.eda()
    a.plot_matplotlib(ref_output)
    print("results ", str(ref_output), "are saved")


execute(
    root, "ecfp4_tsne.csv", "PPI", descriptors, "p2_ecfp4_tsne_p30a5",
)
