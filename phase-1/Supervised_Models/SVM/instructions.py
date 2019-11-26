from SVM import SVM
print("Configurar ubicacion")
root = {"root": "/home/jlmf_g/jlmf/PPI_classifier/phase-1/Databases/",
        "root_Info" : "/home/jlmf_g/jlmf/PPI_classifier/phase-1/Supervised_Models/SVM/Info",
        "root_ROC" : "/home/jlmf_g/jlmf/PPI_classifier/phase-1/Supervised_Models/SVM/ROC"}

descriptors = ['HBA', 'HBD', 'RB', 'LogP', 'TPSA', 'MW', 'Heavy Atom', 'Ring Count', 'Fraction CSP3']

a = SVM(root, "Dataset.csv", "PPI", descriptors, 0.2)
a.train_model('rbf', "balanced")
a.report("D10L5P3SVM3A")