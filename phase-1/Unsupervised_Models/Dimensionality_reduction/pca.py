import pandas as pd
import numpy as np
from decimal import Decimal

import sklearn
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


def pca_loadings(self, descriptors):
    model = self.model 
    loadings = model.components_
    loadings = np.transpose(loadings)
    PCA_loadings = pd.DataFrame(
                    data =loadings,
                    index = descriptors,
                    columns = ["PC 1", "PC 2", "PC 3","PC 4","PC 5", "PC 6" ]
                    )
    return PCA_loadings

def pca_summary(self):
	model = self.model
	variance = model.explained_variance_ratio_
	cumulative_variance = np.cumsum(variance)
	data = [variance, cumulative_variance]
	PCA_summary = pd.DataFrame (
					data = data,
					index = ["Percentage of variance", "Cummulative percentage of variance"],
					columns = ["PC 1", "PC 2", "PC 3","PC 4","PC 5", "PC 6" ]
		)
	return PCA_summary
