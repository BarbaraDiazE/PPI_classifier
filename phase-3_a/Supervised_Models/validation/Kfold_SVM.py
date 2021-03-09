import pandas as pd
import numpy as np
import sklearn

# from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results",
    "root_ROC": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/ROC",
    "trained_models": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/models/",
    "validation_results": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/validation_results/",
}


class Kfold:
    def __init__(self, root, input_file, input_model, target, descriptors, fraction):
        # read de database
        self.Data = pd.read_csv(
            str(root["root"]) + str(input_file), index_col="Unnamed: 0"
        )
        self.root = root
        # preprocesing
        ids = [
            "ipp_id",
            "chembl_id",
            "SMILES",
            "library",
            "PPI family",
            "PPI",
        ]
        self.numerical_data = self.Data.drop(ids, axis=1)
        self.target = target
        self.fraction = fraction
        self.input_model = input_model

    def evaluate_model(self):
        """compute accuracy with kfold"""
        y = np.array(self.Data[self.target])
        y = label_binarize(y, classes=["No", "Yes"])
        y = np.reshape(y, int(y.shape[0]))
        X_train, X_test, y_train, y_test = train_test_split(
            self.numerical_data, y, test_size=self.fraction, random_state=1992
        )
        model = joblib.load(str(root["trained_models"]) + str(self.input_model))
        accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)

        print(
            "accuracy mean: ",
            accuracies.mean(),
            "\n",
            "accuracy std: ",
            accuracies.std(),
        )
        output_kfold = self.input_model.replace(".pkl", "")
        data = {
            "model": output_kfold,
            "accuracies": accuracies,
            "accuracies mean": accuracies.mean(),
            "accuracy std": accuracies.std(),
        }
        data = pd.DataFrame.from_dict(data)
        data = data.T
        # save results
        data.to_csv(f'{root["validation_results"]}{output_kfold}{".csv"}')


# drivercode
print(str(root["root"]))
Data = pd.read_csv(str(root["root"]) + str("dataset_maccskeys_p2.csv"))
ids = ["Unnamed: 0", "ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"]
numerical_data = Data.drop(ids, axis=1)
descriptors = numerical_data.columns.to_list()
del Data
print(len(descriptors))

# SVM9
print("###SVM###")
A = Kfold(
    root, "dataset_ecfp4_p2.csv", "SVM_p2_F1L6P5SVM1A.pkl", "PPI", descriptors, 0.3
)
A.evaluate_model()

# SVM10
print("###SVM###")
A = Kfold(
    root, "dataset_ecfp4_p2.csv", "SVM_p2_F1L6P5SVM1B.pkl", "PPI", descriptors, 0.3
)
A.evaluate_model()

# SVM24
print("###SVM###")
A = Kfold(
    root, "dataset_ecfp6_p2.csv", "SVM_p2_F2L6P5SVM1A.pkl", "PPI", descriptors, 0.3
)
A.evaluate_model()

# SVM25
print("###SVM2###")
A = Kfold(
    root, "dataset_ecfp6_p2.csv", "SVM_p2_F2L6P5SVM1B.pkl", "PPI", descriptors, 0.3
)
A.evaluate_model()

