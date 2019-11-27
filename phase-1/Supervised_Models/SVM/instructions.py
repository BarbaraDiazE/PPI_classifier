from SVM import SVM
print("Configurar ubicacion")

root = {"root": "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Databases/",
        "root_Info" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/SVM/Info",
        "root_ROC" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/SVM/ROC"}

descriptors = ['HBA', 'HBD', 'RB', 'LogP', 'TPSA', 'MW', 'Heavy Atom', 'Ring Count', 'Fraction CSP3']

def execute(root, input_file, target, descriptors, fraction, kernel, balanced, ref_output):
    a = SVM(root, input_file, target, descriptors, fraction)
    a.train_model(kernel,balanced)
    a.report(ref_output)
    print("termine ", str(ref_output))

execute(root, "Dataset.csv", "PPI", descriptors, 0.3, "rbf", "balanced", "D10L5P5SVM3A")
execute(root, "Dataset.csv", "PPI", descriptors, 0.3, "linear", "balanced", "D10L5P5SVM3A")
