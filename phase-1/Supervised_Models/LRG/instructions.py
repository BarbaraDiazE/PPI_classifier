from LRG import LRG

root = {"root": "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Databases/",
        "root_Info" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/LRG/Info",
        "root_ROC" : "/tmpu/jlmf_g/jlmf/B/PPI_classifier/phase-1/Supervised_Models/LRG/ROC"}

descriptors = ['HBA', 'HBD', 'RB', 'LogP', 'TPSA', 'MW', 'Heavy Atom', 'Ring Count', 'Fraction CSP3']

def execute(root, input_file, target, descriptors, fraction, class_weight, solver, ref_output):
    a = LRG(root,input_file,  target, descriptors, 0.3)
    a.train_model(class_weight, solver)
    print("model is trained")
    a.report(ref_output)
    print("report is done", str(ref_output))

execute(root,"Dataset.csv",  "PPI", descriptors, 0.2, "balanced","newton-cg", "D10L5P3LRG1A" )
execute(root,"Dataset.csv",  "PPI", descriptors, 0.2, "balanced","lbfgs", "D10L5P3LRG2A" )
execute(root,"Dataset.csv",  "PPI", descriptors, 0.2, "balanced","liblinear", "D10L5P3LRG3A" )
execute(root,"Dataset.csv",  "PPI", descriptors, 0.2, "balanced","sag", "D10L5P3LRG4A" )
execute(root,"Dataset.csv",  "PPI", descriptors, 0.2, "balanced","saga", "D10L5P3LRG5A" )
execute(root,"Dataset.csv",  "PPI", descriptors, 0.3, "balanced","newton-cg", "D10L5P5LRG1A" )
execute(root,"Dataset.csv",  "PPI", descriptors, 0.3, "balanced","lbfgs", "D10L5P5LRG2A" )
execute(root,"Dataset.csv",  "PPI", descriptors, 0.3, "balanced","liblinear", "D10L5P5LRG3A" )
execute(root,"Dataset.csv",  "PPI", descriptors, 0.3, "balanced","sag", "D10L5P5LRG4A" )
execute(root,"Dataset.csv",  "PPI", descriptors, 0.3, "balanced","saga", "D10L5P5LRG5A" )
