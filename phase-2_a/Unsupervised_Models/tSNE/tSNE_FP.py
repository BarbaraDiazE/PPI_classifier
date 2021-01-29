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
        # self.data = self.data[self.data.library == "PPI"]
        print(self.data.head())
        print("Libraries are: ", self.data.library.unique())
        print("PPI family: ", self.data["PPI family"].unique())
        print("Number of PPI family: ", len(self.data["PPI family"].unique()))
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

    def train_model(self, ref_output):
        b_targets = self.data["PPI family"].unique()
        N = len(b_targets)
        print(N)
        model = TSNE(
            n_components=2,
            init="pca",
            random_state=1992,
            angle=0.5,
            perplexity=30,
            n_iter=1000,
        )
        result = model.fit_transform(self.numerical_data)
        print(result)
        result = pd.DataFrame(data=result, columns=["PC 1", "PC 2"])
        result["ipp_id"] = self.data["ipp_id"]
        result["PPI family"] = self.data["PPI family"]
        params = model.get_params()
        params = pd.DataFrame(
            data=np.array([list(params.keys()), list(params.values())])
        )
        params.to_csv(
            f'{self.root["tsne_info_params"]}{"/"}{"info_"}{ref_output}{".csv"}'
        )
        return result

    def plot_matplotlib(self, ref_output):
        data = self.train_model(ref_output)
        data.to_csv(f'{self.root["tsne_results"]}{"/"}{ref_output}{".csv"}')
        sns.set_context("paper", font_scale=0.6)
        sns.set_style("darkgrid")
        plt.figure(figsize=(10, 6))
        colors = [
            # amarillos
            # "yellow",
            "gold",
            "orange",
            # rosas
            "pink",
            "hotpink",
            "deeppink",
            # morados
            "mediumvioletred",
            # "blueviolet",
            "indigo",
            # verde
            "yellowgreen",
            "limegreen",
            "forestgreen",
            # azules
            "cyan",
            "dodgerblue",
            "darkblue",
            # gris
            # "gray",
            # "darkgray",
            # "teal",
        ]

        sns.scatterplot("PC 1", "PC 2", data=data, hue="PPI family", palette=colors)
        plt.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=7, prop={"size": 8}
        )
        plt.tick_params(axis="both", labelsize=10)
        plt.xlabel("PC 1", fontsize=12)
        plt.ylabel("PC 2", fontsize=12)
        # plt.title("chemical space PPI modulators")
        plt.tight_layout()
        plt.savefig(
            f'{self.root["root_chem_space"]}{"/"}{ref_output} {".png"}', dpi=200
        )
        plt.show()

    def plot_bokeh():
        result = self.train_model()
