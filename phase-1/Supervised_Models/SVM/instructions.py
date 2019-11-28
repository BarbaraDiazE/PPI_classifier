import pandas as pd
from SVM_FP import SVM

root = {"root": "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Databases/",
        "root_Info" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/SVM/Info",
        "root_ROC" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/SVM/ROC"}

#root = {"root": "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/",
#        "root_Info" : "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Supervised_Models/SVM/Info",
#        "root_ROC" : "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Supervised_Models/SVM/ROC"}

#descriptors = ['HBA', 'HBD', 'RB', 'LogP', 'TPSA', 'MW', 'Heavy Atom', 'Ring Count', 'Fraction CSP3']
"""ECFP4"""
Data = pd.read_csv(str(root["root"]) + str("ECFP4_L5.csv")) 
ids = ['Unnamed: 0', 'ID Database', 'Name', 'SMILES', 'subLibrary', 'Library', 'Epigenetic', "PPI"]
numerical_data = Data.drop(ids, axis = 1)
descriptors = numerical_data.columns.to_list()
print(len(descriptors))


def execute(root, input_file, target, descriptors, fraction, kernel, balanced, ref_output):
    a = SVM(root, input_file, target, descriptors, fraction)
    a.train_model(kernel,balanced)
    a.report(ref_output)
    print("report ", str(ref_output), "is done")

execute(root, "ECFP4_L5.csv", "PPI", descriptors, 0.2, "rbf", "balanced", "D10L6P3SVM1A")
execute(root, "ECFP4_L5.csv", "PPI", descriptors, 0.2, "linear", "balanced", "D10L6P3SVM2A")
execute(root, "ECFP4_L5.csv", "PPI", descriptors, 0.2, "poly", "balanced", "D10L6P3SVM2A")
execute(root, "ECFP4_L5.csv", "PPI", descriptors, 0.2, "rbf", "balanced", "D10L6P3SVM3A")
execute(root, "ECFP4_L5.csv", "PPI", descriptors, 0.2, "sigmoid", "balanced", "D10L6P3SVM4A")
execute(root, "ECFP4_L5.csv", "PPI", descriptors, 0.3, "linear", "balanced", "D10L6P5SVM1A")
execute(root, "ECFP4_L5.csv", "PPI", descriptors, 0.3, "poly", "balanced", "D10L6P5SVM2A")
execute(root, "ECFP4_L5.csv", "PPI", descriptors, 0.3, "rbf", "balanced", "D10L6P5SVM3A")
execute(root, "ECFP4_L5.csv", "PPI", descriptors, 0.3, "sigmoid", "balanced", "D10L6P5SVM4A")
