""" Compute FP """
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from compute_fp.com_fp_BV import *

root = {"root": "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Databases/",
        "root_Info" : "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Supervised_Models/SVM/Info",
        "root_ROC" : "/home/barbara/Documents/DIFACQUIM/PPI_classifier/phase-1/Supervised_Models/SVM/ROC"}

#### Reading molecules and activity from SDF
class CompFP:
    
    def __init__(self, input_file):
        self.Data  = pd.read_csv(str(root["root"]) + str(input_file))
        #Muestreo
        self.Data = pd.DataFrame.sample(self.Data, frac=0.01, replace=True,  random_state=1992, axis=None) 
        
    def fp_array(self, fp_func):
        smiles = self.Data.SMILES.to_list()
        fp = fp_func(smiles)
        output = []
        for f in fp:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(f, arr)
            output.append(arr)
        y = np.asarray(output)
        print(y.shape)
        print(y)
        return y

a = CompFP("BIOFACQUIM_DB.csv")
a.fp_array(morgan2)

