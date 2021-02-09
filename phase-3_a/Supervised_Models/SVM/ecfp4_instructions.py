"Instructions to train SVM from ECFP4 representation"
import os
import pandas as pd
from SVM_FP import SVM


root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results",
    "root_ROC": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/ROC",
    "trained_models": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/models",
}

# """ECFP4 models"""
print("ecfp4")
print(str(root["root"]))
Data = pd.read_csv(str(root["root"]) + str("dataset_ecfp4_p2.csv"), low_memory=False)
print(Data.columns)
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
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "linear",
    "balanced",
    "p2_F1L6P3SVM1A",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "poly",
    "balanced",
    "p2_F1L6P3SVM2A",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "rbf",
    "balanced",
    "p2_F1L6P3SVM3A",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "sigmoid",
    "balanced",
    "p2_F1L6P3SVM4A",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "linear",
    "balanced",
    "p2_F1L6P5SVM1A",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "poly",
    "balanced",
    "p2_F1L6P5SVM2A",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "rbf",
    "balanced",
    "p2_F1L6P5SVM3A",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "sigmoid",
    "balanced",
    "p2_F1L6P5SVM4A",
)
# Not balanced
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "linear",
    None,
    "p2_F1L6P3SVM1B",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "poly",
    None,
    "p2_F1L6P3SVM2B",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "rbf",
    None,
    "p2_F1L6P3SVM3B",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "sigmoid",
    None,
    "p2_F1L6P3SVM4B",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "linear",
    None,
    "p2_F1L6P5SVM1B",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "poly",
    None,
    "p2_F1L6P5SVM2B",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "rbf",
    None,
    "p2_F1L6P5SVM3B",
)
execute(
    root,
    "dataset_ecfp4_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "sigmoid",
    None,
    "p2_F1L6P5SVM4B",
)
