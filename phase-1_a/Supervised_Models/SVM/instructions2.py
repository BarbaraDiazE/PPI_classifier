import pandas as pd
from SVM_FP import SVM

# root = {"root": "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Databases/",
#        "root_Info" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/SVM/Info",
#        "root_ROC" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/SVM/ROC"}

root = {
    "root": "/tmpu/jlmf_g/bide_a/PPI_classifier/phase-1/Databases/",
    "root_Info": "/tmpu/jlmf_g/bide_a/PPI_classifier/phase-1/Supervised_Models/SVM/Info",
    "root_ROC": "/tmpu/jlmf_g/bide_a/PPI_classifier/phase-1/Supervised_Models/SVM/ROC",
}

# descriptors = ['HBA', 'HBD', 'RB', 'LogP', 'TPSA', 'MW', 'Heavy Atom', 'Ring Count', 'Fraction CSP3']
"""AtomPairs"""
Data = pd.read_csv(str(root["root"]) + str("AtomPairs_L6.csv"))
ids = [
    "Unnamed: 0",
    "ID Database",
    "Name",
    "SMILES",
    "subLibrary",
    "Library",
    "Epigenetic",
    "PPI",
]
numerical_data = Data.drop(ids, axis=1)
descriptors = numerical_data.columns.to_list()
print(descriptors)


def execute(
    root, input_file, target, descriptors, fraction, kernel, balanced, ref_output
):
    a = SVM(root, input_file, target, descriptors, fraction)
    a.train_model(kernel, balanced)
    a.report(ref_output)
    print("report ", str(ref_output), "is done")


# balanceados
execute(
    root, "AtomPairs_L6.csv", "PPI", descriptors, 0.2, "rbf", "balanced", "F4L6P3SVM3A"
)
execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.2,
    "sigmoid",
    "balanced",
    "F4L6P3SVM4A",
)
execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.3,
    "linear",
    "balanced",
    "F4L6P5SVM1A",
)
execute(
    root, "AtomPairs_L6.csv", "PPI", descriptors, 0.3, "poly", "balanced", "F4L6P5SVM2A"
)
execute(
    root, "AtomPairs_L6.csv", "PPI", descriptors, 0.3, "rbf", "balanced", "F4L6P5SVM3A"
)
execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.3,
    "sigmoid",
    "balanced",
    "F4L6P5SVM4A",
)
# no balanceados
execute(root, "AtomPairs_L6.csv", "PPI", descriptors, 0.2, "rbf", None, "F4L6P3SVM3ABN")
execute(
    root, "AtomPairs_L6.csv", "PPI", descriptors, 0.2, "sigmoid", None, "F4L6P3SVM4ABN"
)
execute(
    root, "AtomPairs_L6.csv", "PPI", descriptors, 0.3, "linear", None, "F4L6P5SVM1ABN"
)
execute(
    root, "AtomPairs_L6.csv", "PPI", descriptors, 0.3, "poly", None, "F4L6P5SVM2ABN"
)
execute(root, "AtomPairs_L6.csv", "PPI", descriptors, 0.3, "rbf", None, "F4L6P5SVM3ABN")
execute(
    root, "AtomPairs_L6.csv", "PPI", descriptors, 0.3, "sigmoid", None, "F4L6P5SVM4ABN"
)
