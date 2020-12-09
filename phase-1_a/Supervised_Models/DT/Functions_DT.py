import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import sklearn
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score, roc_auc_score, auc, roc_curve, confusion_matrix

def test_compound(Data,Library, Name, descriptors):
    """
    Data, DataFrame with Dataset
    Library, test compound library
    Name, str compound name
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

def plot_roc(ref_output, y_test, y_score, root_ROC):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    pd.DataFrame.from_dict({"fpr": fpr , "tpr": tpr}).to_csv(str(root_ROC) + "/LF_ROC_data_"+str(ref_output)+".csv", sep = ",")
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(str(root_ROC) + "/DT_ROC" + str(ref_output) + ".png")
    plt.show()
    
    return roc_auc

def dt_report(ref_output, Data, parameters, y_test, predictions, descriptors, atributes,roc_auc, root_Info):
    #
    #df = pd.DataFrame({ 'Descriptors': descriptors,
    #         'Coeff': np.around(atributes["coeff"],2) })
    #df = df.sort_values(by='Coeff', ascending=False)
    #df.to_csv(str(root_Info) + "/DT_coeff_"+str(ref_output)+".csv", sep = ",")
    data = {
                ('Info', "Method"): parameters["Method"],
                ('Info', "Class weight"): parameters["class weight"],
                ('Info', "Max depth"): parameters["max_depth"],
                ('Info', "Libraries"): " ".join(map(str, list(Data.Library.unique()))),
                ('Info', "Test fraction"): parameters["fraction"],
                ('Info', "Descriptors"): ' '.join(map(str, descriptors)),    
                #('Results', "Coeff"):  " ".join(map(str, list(np.around(np.array(df["Coeff"]),2)))),         
                ('Results', "Clases"): " ".join(map(str, list(atributes["classes"]))),
                ('Results', "tree_"): atributes["tree_"],
                ('Results', "max_features_"): atributes["max_features_"],
                ('Metrics',"Accuracy"):round(accuracy_score(y_test, predictions),2),   
                ('Metrics',"Balanced Accuracy"): round(balanced_accuracy_score(y_test, predictions),2),
                ('Metrics',"Precision"): round(precision_score(y_test, predictions),2),
                ('Metrics',"F1"): round(f1_score(y_test, predictions),2),
                ('Metrics',"ROC AUC score"): round(roc_auc_score(y_test, predictions),2),   
                ('Metrics',"AUC"): round(roc_auc,2),       
                ('Metrics', "Confusion matrix"): confusion_matrix(y_test, predictions)
            }
    Report = pd.Series(data)
    Report.to_csv(str(root_Info) + "/DT_info_"+str(ref_output)+".csv", sep = ",")


