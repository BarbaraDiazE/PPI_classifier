import pandas as pd
import numpy as np

def test_compound(Data, Name, descriptors):
    """
    Data, DataFrame with Dataset
    Name, str compound name
    target,str target category
    descriptors, list that contains desired descriptors
    """
    DF = Data[Data["Library"]== Library]
    compound = DF[DF["Name"]== Name]
    compound = test[descriptors]
    return compound

def test_compound_real_category(Data, Name, target):
    """
    Data, DataFrame with Dataset
    Name, str compound name
    target, str target category
    """
    test = Data[Data["Name"]== Name]
    result = test[target]
    return result