"""K-means code with descriptors"""
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.externals import joblib


class TSNE_FP:
    def __init__(self, root, input_file, target, descriptors):
        self.data = pd.read_csv(
            str(root["root"]) + str(input_file), low_memory=True, index_col="Unnamed: 0"
        )
        self.data = self.data[self.data.library == "PPI"]
        print(self.data.head())
        print("Libraries are: ", self.data.library.unique())
        print("Total compounds: ", self.data.shape[0])
        self.descriptors = descriptors
        self.target = target
        self.root = root
        ids = [
            "ipp_id",
            "chembl_id",
            "SMILES",
            "library",
            "PPI family",
            "PPI",
        ]
        self.numerical_data = self.data.drop(ids, axis=1)
        # print(self.numerical_data)

    def eda(self):
        # numerical_data = self.data[self.descriptors]
        fig = plt.subplots()
        sns.heatmap(self.numerical_data, cmap="Set2")
        plt.show()

    def train_model(self):
        b_targets = self.data["PPI family"].unique()
        N = len(b_targets)
        print(N)
        model = TSNE(
            n_components=2, init="pca", random_state=1992, angle=0.5, perplexity=30
        ).fit_transform(self.numerical_data)
        result = pd.DataFrame(data=model, columns=["PC 1", "PC 2"])
        result["ipp_id"] = self.data["ipp_id"]
        result["PPI family"] = self.data["PPI family"]
        print(result.head())
        return result

    def plot_matplotlib(self, ref_output):
        data = self.train_model()
        data.to_csv(f'{self.root["tsne_results"]}{"/"}{ref_output}{".csv"}')
        sns.set_context("paper", font_scale=0.7)
        sns.set_style("darkgrid")
        plt.figure(figsize=(10, 6))
        sns.scatterplot("PC 1", "PC 2", data=data, hue="PPI family", palette="Set2")
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.title("chemical space PPI modulators")
        plt.tight_layout()
        plt.savefig(
            f'{self.root["root_chem_space"]}{"/"}{ref_output} {".png"}', dpi=150
        )
        plt.show()

    def plot_bokeh():
        result = self.train_model()
