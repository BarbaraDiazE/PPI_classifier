"""K-means code with descriptors"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.externals import joblib


class Kmeans:
    def __init__(self, root, input_file, target, descriptors, fraction):
        self.data = pd.read_csv(
            str(root["root"]) + str(input_file), low_memory=True, index_col="Unnamed: 0"
        )
        self.data = self.data[self.data.library == "PPI"]
        print(self.data.head())
        # Muestreo
        # self.Data = pd.DataFrame.sample(self.Data, frac=0.1, replace=True,  random_state=1992, axis=None)
        self.fraction = fraction
        print(self.data.PPI.unique())
        print("Libraries are: ", self.data.library.unique())
        print("Total compounds ", self.data.shape[0])
        self.descriptors = descriptors
        # print(descriptors)
        self.target = target
        self.root = root

    def eda(self):
        numerical_data = self.data[self.descriptors]
        numerical_data = numerical_data.fillna(numerical_data.mean())
        sns.heatmap(numerical_data.corr())
        self.numerical_data = numerical_data

        fig = plt.subplots()

        sns.heatmap(numerical_data.corr())
        plt.show()

    def train_model(self):
        b_targets = self.data["PPI family"].unique()
        N = len(b_targets)
        print(N)
        model = KMeans(n_clusters=N)
        model.fit(self.numerical_data)
        # Add predictions to Data
        self.data["Cluster"] = model.labels_
        print("descriptors", self.descriptors[0], self.descriptors[1])
        fig, (ax1, ax2) = plt.subplots(2)
        # sns.regplot(x, y, ax=ax1)
        # sns.kdeplot(x, ax=ax2)
        sns.scatterplot(
            self.descriptors[0],
            self.descriptors[100],
            data=self.data,
            hue="PPI family",
            palette="cool",
            # fit_reg=False,
            ax=ax1,
        )
        sns.scatterplot(
            self.descriptors[0],
            self.descriptors[100],
            data=self.data,
            hue="Cluster",
            palette="cool",
            # fit_reg=False,
            ax=ax2,
        )

        fig.tight_layout()
        plt.show()

