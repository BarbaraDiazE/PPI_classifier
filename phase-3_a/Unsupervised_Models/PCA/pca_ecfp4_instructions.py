"""Instructions to perform pca based on ecfp4"""

import os
import pandas as pd
from PCA_FP import PCA_FP

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info",
    "root_chem_space": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/chemical_space/pca",
    "pca_results": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/PCA/PCA_results",
    "pca_info_params": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/PCA/PCA_results/info_params",
    "pca_loadings": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/PCA/PCA_results/loadings",
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
    a = PCA_FP(root, input_file, target, descriptors)
    # a.eda()
    a.plot_matplotlib(ref_output)
    print("results ", str(ref_output), "are saved")


execute(
    root, "ecfp4_tsne.csv", "PPI", descriptors, "p2_ecfp4_pca",
)
