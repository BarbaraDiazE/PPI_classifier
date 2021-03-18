"""Virtual screaning of ppi dataset (including training and test set)"""
import pandas as pd

from Ensemble_1 import Ensemble

# read csv
route = "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Databases/"
data = pd.read_csv(f'{route}{"dataset_maccskeys_p2.csv"}', index_col="Unnamed: 0")
""""
sample
"""
print("initial data shape", data.shape)
features = ["SMILES", "library"]
data = data[features]
data = data[data["library"] == "PPI"]
print(data.head())
print("finl data shape", data.shape)
n_ppi = data.shape[0]
# filter smiles
smiles = data.SMILES.to_list()


# execute Ensamble
ensemble_predictions = list()
for i in smiles:
    ensemble_predictions.append(Ensemble(i).score_ensemble())

# save results
data["ensemble predictions"] = ensemble_predictions
data.to_csv("ensemble_predictions_DF.csv")
# compute the number of positives
positive_predictions = data[data["ensemble predictions"] == "Yes"].shape
print("number of positive predictions:", positive_predictions)
# compute compute the percentage of positives
print("percentage positive predictions: ", (positive_predictions[0] / n_ppi) * 100)

