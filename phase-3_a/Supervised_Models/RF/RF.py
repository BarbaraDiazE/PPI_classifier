import pandas as pd
import numpy as np

import os

from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import plot_roc_curve
from Functions_RF import (
    test_compound,
    test_compound_real_category,
    rf_report,
    plot_roc,
)


class RF:
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

    def train_model(self, n_estimators, criterion, max_depth, class_weight):
        """
        n_estimators: int, {100,500,1000}.
        criterion: str {"gini", "entropy”}
        max_depth, int, default=None 
        class_weight: str, {‘balanced’ or None}
        """
        y = np.array(self.Data[self.target])
        y = label_binarize(y, classes=["No", "Yes"])
        numerical_data = pd.DataFrame(
            StandardScaler().fit_transform(self.Data[self.descriptors])
        )
        X_train, X_test, y_train, y_test = train_test_split(
            numerical_data, y, test_size=self.fraction, random_state=1992
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            # min_samples_split=2,
            # min_samples_leaf=1,
            # min_weight_fraction_leaf=0.0,
            # max_features="auto",
            # max_leaf_nodes=None,
            random_state=2020,
            class_weight=class_weight,
            # ccp_alpha=0.0,
        )
        model.fit(X_train, y_train)
        self.atributes = {
            "classes": model.classes_,
            "base_estimator": model.base_estimator_,
            "stimators": model.estimators_,
            "feature_importances": model.feature_importances_,
        }
        print("base stimator", model.base_estimator_)
        print("stimators", model.estimators_)
        self.parameters = {
            "Method": "Linear Regression",
            "class weight": class_weight,
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "fraction": self.fraction * 100,
        }
        self.predictions = {
            "predictions": model.predict(X_test),
            # "y_score": model.decision_function(X_test),
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
            self.model,
            self.predictions["X_text"],
            self.predictions["y_test"],
            self.root["root_ROC"],
        )
        rf_report(
            ref_output,
            self.Data,
            self.parameters,
            self.predictions["y_test"],
            self.predictions["predictions"],
            self.descriptors,
            self.atributes,
            # roc_auc,
            self.root["root_Info"],
        )
        # print("report ", str(ref_output), "is Done")
        # save model
        output = ref_output.replace(".csv", "")

        # Save the model as a pickle in a file
        joblib.dump(
            self.model, f'{self.root["trained_models"]}{"/"}{"RF_"}{output}{".pkl"}'
        )

