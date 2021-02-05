"""Exploratory Analisys by library, once at the time"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/SVM/Info",
    "root_ROC": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/SVM/ROC",
    "trained_models": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/trained_models",
}


class Exploratory_Analysis:
    def __init__(self, input_file, Library):
        Data = pd.read_csv(
            f'{root["root"]}{input_file}', index_col="Unnamed: 0", low_memory=False
        )
        self.Library = Library
        # self.features = features
        # filter by library
        Data = Data[Data["library"] == Library].reset_index()
        self.Data = Data.drop(
            ["index", "ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"],
            axis=1,
        )
        print(self.Data.head())
        print(self.Data.columns)
        print(str(Library), "Library contains : ", str(self.Data.shape[0]), "elements")

    def stats(self):
        stats = self.Data.describe()
        stats = stats.round(3)
        stats = stats.T  # transpose
        stats.to_csv((str(self.Library) + "_statistics.csv"), sep=",")
        print(stats)

    def correlation(self):
        corr = self.Data.corr()
        f, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = sns.heatmap(corr, annot=False)
        figure = corr_matrix.get_figure()
        # figure.savefig((str(self.Library)+'_correlation.png'), dpi=800)
        figure.savefig("corr_tutoral.png")
        h = self.Data.hist(figsize=(10, 12))

    def boxplot(self, feature):
        Data = self.Data
        sns.boxplot(
            x="Library", y=Data[feature], data=Data,
        )
        plt.savefig("boxplot_feature.png")


# features = [
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

a = Exploratory_Analysis("dataset_descriptors_p2.csv", "FDA",)
a.stats()
a.correlation()
# a.boxplot("HBD")

