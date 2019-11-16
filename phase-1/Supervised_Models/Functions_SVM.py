import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, auc, roc_curve, confusion_matrix

def test_compound(Data,Library, Name, descriptors):
    """
    Data, DataFrame with Dataset
    Name, str compound name
    target,str target category
    descriptors, list that contains desired descriptors
    """
    DF = Data[Data["Library"]== Library]
    compound = DF[DF["Name"]== Name]
    compound = compound[descriptors]
    return compound

def test_compound_real_category(Data, Name, target):
    """
    Data, DataFrame with Dataset
    Name, str compound name
    target, str target category
    """
    test = Data[Data["Name"]== Name]
    result = list(test[target])
    return result

def svm_report(ref_output, Data,  kernel,fraction, y_test, predictions, descriptors, roc_auc):
    data = {
                ('Info', "Method"): "SVM",
                ('Info', "Kernel"): str(kernel),
                ('Info', "Libraries"): " ".join(map(str, list(Data.Library.unique()))),
                ('Info', "Test fraction"): fraction * 100,
                ('Info', "Descriptors"): ' '.join(map(str, descriptors)),               
                ('Metrics',"Accuracy"):accuracy_score(y_test, predictions),            
                ('Metrics',"Precision"):precision_score(y_test, predictions),
                ('Metrics',"F1"):f1_score(y_test, predictions),
                ('Metrics',"ROC AUC score"):roc_auc_score(y_test, predictions),   
                ('Metrics',"AUC"):roc_auc,       
                ('Metrics', "Confusion matrix"): confusion_matrix(y_test, predictions)
            }
    Report = pd.Series(data)
    Report.to_csv("SVM_"+str(ref_output)+".csv", sep = ",")

def plot_roc(y_test, y_score):
    # predict probabilities
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print("AUC", roc_auc)
    fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='green', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc, roc_auc_micro
