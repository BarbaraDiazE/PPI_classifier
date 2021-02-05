import pandas as pd
from LRG import LRG

root = {
    "root": "/tmpu/jlmf_g/bide_a/PPI_classifier/phase-1/Databases/",
    "root_Info": "/tmpu/jlmf_g/bide_a/PPI_classifier/phase-1/Supervised_Models/LRG/Info",
    "root_ROC": "/tmpu/jlmf_g/bide_a/PPI_classifier/phase-1/Supervised_Models/LRG/ROC",
}

# descriptors = [
#     "HBA",
#     "HBD",
#     "RB",
#     "LogP",
#     "TPSA",
#     "MW",
#     "Heavy Atom",
#     "Ring Count",
#     "Fraction CSP3",
# ]
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
    root, input_file, target, descriptors, fraction, class_weight, solver, ref_output
):
    a = LRG(root, input_file, target, descriptors, 0.3)
    a.train_model(class_weight, solver)
    print("model is trained")
    a.report(ref_output)
    print("report is done", str(ref_output))


execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.2,
    "balanced",
    "newton-cg",
    "F4L6P3LRG1A",
)
execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.2,
    "balanced",
    "lbfgs",
    "F4L6P3LRG2A",
)
execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.2,
    "balanced",
    "liblinear",
    "F4L6P3LRG3A",
)
execute(
    root, "AtomPairs_L6.csv", "PPI", descriptors, 0.2, "balanced", "sag", "F4L6P3LRG4A"
)
execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.2,
    "balanced",
    "saga",
    "F4L6P3LRG5A",
)
execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.3,
    "balanced",
    "newton-cg",
    "F4L6P5LRG1A",
)
execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.3,
    "balanced",
    "lbfgs",
    "F4L6P5LRG2A",
)
execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.3,
    "balanced",
    "liblinear",
    "F4L6P5LRG3A",
)
execute(
    root, "AtomPairs_L6.csv", "PPI", descriptors, 0.3, "balanced", "sag", "F4L6P5LRG4A"
)
execute(
    root,
    "AtomPairs_L6.csv",
    "PPI",
    descriptors,
    0.3,
    "balanced",
    "saga",
    "F4L6P5LRG5A",
)
