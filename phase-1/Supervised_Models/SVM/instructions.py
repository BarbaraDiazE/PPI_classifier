from SVM import SVM
print("Configurar ubicacion")

#root = {"root": "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Databases/",
#        "root_Info" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/SVM/Info",
#        "root_ROC" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/SVM/ROC"}

root = {"root": "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/",
        "root_Info" : "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Supervised_Models/SVM/Info",
        "root_ROC" : "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Supervised_Models/SVM/ROC"}

descriptors = ['HBA', 'HBD', 'RB', 'LogP', 'TPSA', 'MW', 'Heavy Atom', 'Ring Count', 'Fraction CSP3']

def execute(root, input_file, target, descriptors, fraction, kernel, balanced, ref_output):
    a = SVM(root, input_file, target, descriptors, fraction)
    a.train_model(kernel,balanced)
    a.report(ref_output)
    print("report ", str(ref_output), "is done")

#execute(root, "Dataset.csv", "PPI", descriptors, 0.2, "linear", "balanced", "prueba")
#execute(root, "Dataset.csv", "PPI", descriptors, 0.3, "poly", "balanced", "poly")
#execute(root, "Dataset.csv", "PPI", descriptors, 0.2, "linear", "balanced", "prueba2")
execute(root, "Dataset.csv", "PPI", descriptors, 0.2, "sigmoid", "balanced", "sigmoid")