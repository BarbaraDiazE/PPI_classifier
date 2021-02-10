import os
import re
import pandas as pd

"create a dictionary to generate and storage RF model information"
# define dunctions
csv_files = list()
for file in os.listdir():
    if file.endswith(".csv"):
        csv_files.append(file)


def remove_csv(s):
    return s.replace(".csv", "")


def population(string):
    if ["P3"] == re.findall("P3", string):
        p = "80/20"
    elif ["P5"] == re.findall("P5", string):
        p = "70/30"
    else:
        p = "No information"
    return p


def libraries(string):
    "identify libraries"
    if ["L6"] == re.findall("L6", string):
        l = "PPI and FDA"
    else:
        l = "Libraries not known"
    return l


def stimators(string):
    "identify employed kernel to train the model"
    if ["RF1"] == re.findall("RF1", string):
        n_stimators = "100"
    elif ["RF2"] == re.findall("RF2", string):
        n_stimators = "500"
    elif ["RF3"] == re.findall("RF3", string):
        n_stimators = "1000"

    else:
        n_stimators = "No n stimator information"
    return n_stimators


def get_descriptor(string):
    "identify desscriptor to train the model"
    if ["F1L"] == re.findall("F1L", string):
        descriptor = "ECFP4"
    elif ["F2L"] == re.findall("F2L", string):
        descriptor = "ECFP6"
    elif ["F3L"] == re.findall("F3L", string):
        descriptor = "MACCS Keys"
    elif ["F4L"] == re.findall("F4L", string):
        descriptor = "AtomPairs"
    elif ["D12"] == re.findall("D12", string):
        descriptor = "Physicochemical descriptors"
    else:
        descriptor = "No descriptor information"
    return descriptor


def class_weight(string):
    if ["A"] == re.findall("A", string):
        c = "True"
    elif ["B"] == re.findall("B", string):
        c = "False"
    else:
        c = "No information"
    return c


# define empty dictionary
dict1 = dict()
# populate the dictionary with model information
model_names = list(map(remove_csv, csv_files))
model_names.sort()
dict1["ID model"] = ["RF" + str(num + 1) for num in range(len(model_names))]
dict1["Model name"] = model_names
dict1["Representations"] = list(map(get_descriptor, model_names))
dict1["Data"] = list(map(population, model_names))
dict1["Libraries"] = list(map(libraries, model_names))
dict1["Algorithm"] = ["RF" for i in range(len(model_names))]
dict1["N stimator"] = list(map(stimators, model_names))
dict1["Class weight"] = list(map(class_weight, model_names))


DF = pd.DataFrame.from_dict(data=dict1,)
DF.to_csv(
    "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/info_metrics/RF_id_models.csv"
)
print(DF.head())
