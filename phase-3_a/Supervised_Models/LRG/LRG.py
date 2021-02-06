import pandas as pd
import numpy as np

import os

from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from Functions_LRG import (
    test_compound,
    test_compound_real_category,
    lrg_report,
    plot_roc,
)


class LRG:
    def __init__(self, root, input_file, target, descriptors, fraction):
        self.Data = pd.read_csv(str(root["root"]) + str(input_file))
        # Muestreo
        # self.Data = pd.DataFrame.sample(self.Data, frac=0.05, replace=True,  random_state=1992, axis=None)
        self.fraction = fraction
        print(self.Data.PPI.unique())
        print("Libraries are: ", self.Data.library.unique())
        # print("PPI modulator: ", self.Data[target].unique())
        print("Total compounds ", self.Data.shape[0])
        self.descriptors = descriptors
        self.target = target
        self.root = root

    def train_model(self, solver, class_weight):
        """
        class_weight: dict or ‘balanced’, optional (default=None)
        solver: str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’).
        """
        y = np.array(self.Data[self.target])
        y = label_binarize(y, classes=["No", "Yes"])
        numerical_data = pd.DataFrame(
            StandardScaler().fit_transform(self.Data[self.descriptors])
        )
        X_train, X_test, y_train, y_test = train_test_split(
            numerical_data, y, test_size=self.fraction, random_state=1992
        )
        model = LogisticRegression(
            fit_intercept=True,
            class_weight=class_weight,
            random_state=1992,
            n_jobs=2,
            solver=solver,
        )
        model.fit(X_train, y_train)
        self.atributes = {
            "classes": model.classes_,
            "coeff": list(model.coef_[0]),
            "inter": model.intercept_,
            "iter": model.n_iter_,
        }
        self.parameters = {
            "Method": "Linear Regression",
            "class weight": class_weight,
            "solver": solver,
            "fraction": self.fraction * 100,
        }
        self.predictions = {
            "predictions": model.predict(X_test),
            "y_score": model.decision_function(X_test),
            "X_text": X_test,
            "y_test": y_test,
        }
        self.model = model

    def single_prediction(self, Library, Name, target):
        compound = test_compound(self.Data, Library, Name, self.descriptors)
        result = test_compound_real_category(self.Data, Name, target)
        print("Evaluation of ", str(Name))
        print("Predicted activity value: ", str(self.model.predict(compound)))
        print("Real activity value", result)

    def report(self, ref_output):
        roc_auc = plot_roc(
            ref_output,
            self.predictions["y_test"],
            self.predictions["y_score"],
            self.root["root_ROC"],
        )
        lrg_report(
            ref_output,
            self.Data,
            self.parameters,
            self.predictions["y_test"],
            self.predictions["predictions"],
            self.descriptors,
            self.atributes,
            roc_auc,
            self.root["root_Info"],
        )
        print("report ", str(ref_output), "is Done")
