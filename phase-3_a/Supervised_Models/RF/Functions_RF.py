import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import sklearn

from sklearn.metrics import plot_roc_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    auc,
    roc_curve,
    confusion_matrix,
    recall_score,
)


def test_compound(Data, Library, Name, descriptors):
    """
    Data, DataFrame with Dataset
    Library, test compound library
    Name, str compound name
    descriptors, list that contains desired descriptors
    """
    DF = Data[Data["Library"] == Library]
    compound = DF[DF["Name"] == Name]
    compound = compound[descriptors]
    return compound


def test_compound_real_category(Data, Name, target):
    """
    Data, DataFrame with Dataset
    Name, str compound name
    target, str target category
    """
    test = Data[Data["Name"] == Name]
    result = list(test[target])
    return result


def plot_roc(ref_output, model, x, y, root_ROC):
    roc_auc = plot_roc_curve(model, x, y, color="red",)
    lw = 2
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.savefig(str(root_ROC) + "/RF_ROC_" + str(ref_output) + ".png")
    plt.show()

    return roc_auc


def rf_report(
    ref_output,
    Data,
    parameters,
    y_test,
    predictions,
    descriptors,
    atributes,
    root_Info,
):
    y = np.array([1, 1, 2, 2])
    pred = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=2)
    print(fpr)

    data = {
        ("Info", "Method"): parameters["Method"],
        ("Info", "Class weight"): parameters["class weight"],
        ("Info", "Max depth"): parameters["max_depth"],
        ("Info", "Libraries"): " ".join(map(str, list(Data.library.unique()))),
        ("Info", "Test fraction"): parameters["fraction"],
        ("Info", "Descriptors"): " ".join(map(str, descriptors)),
        ("Results", "Clases"): " ".join(map(str, list(atributes["classes"]))),
        ("Metrics", "Accuracy"): round(accuracy_score(y_test, predictions), 2),
        ("Metrics", "Balanced Accuracy"): round(
            balanced_accuracy_score(y_test, predictions), 2
        ),
        ("Metrics", "Precision"): round(precision_score(y_test, predictions), 2),
        ("Metrics", "F1"): round(f1_score(y_test, predictions), 2),
        ("Metrics", "ROC AUC score"): round(roc_auc_score(y_test, predictions), 2),
        ("Metrics", "AUC"): round(auc(fpr, tpr), 2),
        ("Metrics", "Confusion matrix"): confusion_matrix(y_test, predictions),
        ("Metrics", "recall"): recall_score(y_test, predictions),
    }
    Report = pd.Series(data)
    Report.to_csv(str(root_Info) + "/RF_info_" + str(ref_output) + ".csv", sep=",")

