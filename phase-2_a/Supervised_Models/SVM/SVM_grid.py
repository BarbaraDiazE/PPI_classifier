import pandas as pd
import numpy as np

import os

from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from Functions_grid import (
    get_atributes,
    test_compound,
    test_compound_real_category,
    svm_report,
    plot_roc,
)


class SVM:
    def __init__(self, root, input_file, target, descriptors, fraction):
        self.Data = pd.read_csv(str(root["root"]) + str(input_file), low_memory=True,)
        print(self.Data.head())
        # Muestreo
        # self.Data = pd.DataFrame.sample(self.Data, frac=0.1, replace=True,  random_state=1992, axis=None)
        self.fraction = fraction
        print(self.Data.PPI.unique())
        print("Libraries are: ", self.Data.library.unique())
        print("Total compounds ", self.Data.shape[0])
        self.descriptors = descriptors
        print(self.descriptors)
        self.target = target
        self.root = root

    def train_model(self, kernel, class_weight):
        """
        kernel: str, ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        class_weight : ‘balanced’
        """
        ###grid###
        param_grid = {
            # "C": [0.1, 1, 10, 100, 1000],
            # "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
            "C": [0.1, 100],
            "gamma": [1, 0.001],
            "kernel": [kernel],
        }

        ####
        y = np.array(self.Data[self.target])
        y = label_binarize(y, classes=["No", "Yes"])
        y = np.reshape(y, int(y.shape[0]))
        numerical_data = pd.DataFrame(
            StandardScaler().fit_transform(self.Data[self.descriptors])
        )
        print(numerical_data.head())
        # numerical_data = numerical_data[numerical_data.columns[0:10]].round(3)
        numerical_data = numerical_data.fillna(numerical_data.mean())
        numerical_data.to_csv("borrar.csv")
        print(numerical_data.head())
        X_train, X_test, y_train, y_test = train_test_split(
            numerical_data, y, test_size=self.fraction, random_state=1992
        )
        # model = SVC(
        #     kernel=kernel,
        #     probability=True,
        #     class_weight=class_weight,
        #     random_state=1992,
        # )
        model = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
        print(model)
        model.fit(X_train, y_train)
        # self.atributes = get_atributes(kernel, model)
        # self.parameters = {
        #     "Method": "Linear Regression",
        #     "Class weight": class_weight,
        #     "kernel": kernel,
        #     "fraction": self.fraction * 100,
        # }
        # self.predictions = {
        #     "predictions": model.predict(X_test),
        #     "y_score": model.decision_function(X_test),
        #     "X_text": X_test,
        #     "y_test": y_test,
        # }
        self.model = model

    def single_prediction(self, Library, Name, target):
        compound = test_compound(self.Data, Library, Name, self.descriptors)
        result = test_compound_real_category(self.Data, Name, target)
        print("Evaluation of ", str(Name))
        print("Predicted activity value: ", str(self.model.predict(compound)))
        print("Real activity value", result)

    def report(self, ref_output):
        # save model
        output = ref_output

        # Save the model as a pickle in a file
        joblib.dump(self.model, f'{self.root["trained_models"]}{"/"}{output}{".pkl"}')
        print("model_saved")
        # roc_auc = plot_roc(
        #     ref_output,
        #     self.predictions["y_test"],
        #     self.predictions["y_score"],
        #     self.root["root_ROC"],
        # )

        # svm_report(
        #     ref_output,
        #     self.Data,
        #     self.parameters,
        #     self.predictions["y_test"],
        #     self.predictions["predictions"],
        #     self.descriptors,
        #     self.atributes,
        #     roc_auc,
        #     self.root["root_Info"],
        # )
        print("model.best_params_", self.model.best_params_)
        print("best stimators", self.model.best_estimator_)
        print("final descriptors", self.model.cv_results_.keys())