import pandas as pd
from SVM_FP import SVM


root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info",
    "root_ROC": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/ROC",
    "trained_models": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/trained_models",
}

# """ECFP4 models"""
print("ecfp4")
print(str(root["root"]))
Data = pd.read_csv(str(root["root"]) + str("dataset_ecfp6_p2.csv"), low_memory=False)
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
    "dataset_ecfp6_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "linear",
    None,
    "p2_F1L6P3SVM1ABN",
)
execute(
    root,
    "dataset_ecfp6_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "poly",
    None,
    "p2_F2L6P3SVM2ABN",
)
execute(
    root,
    "dataset_ecfp6_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "rbf",
    None,
    "p2_F2L6P3SVM3ABN",
)
execute(
    root,
    "dataset_ecfp6_p2.csv",
    "PPI",
    descriptors,
    0.2,
    "sigmoid",
    None,
    "p2_F2L6P3SVM4ABN",
)
execute(
    root,
    "dataset_ecfp6_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "linear",
    None,
    "p2_F2L6P5SVM1ABN",
)
execute(
    root,
    "dataset_ecfp6_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "poly",
    None,
    "p2_F2L6P5SVM2ABN",
)
execute(
    root,
    "dataset_ecfp6_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "rbf",
    None,
    "p2_F2L6P5SVM3ABN",
)
execute(
    root,
    "dataset_ecfp6_p2.csv",
    "PPI",
    descriptors,
    0.3,
    "sigmoid",
    None,
    "p2_F2L6P5SVM4ABN",
)
