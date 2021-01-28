import os
import re
import pandas as pd

"create a dictionary to generate to storage model information"
# define dunctions
csv_files = list()
for file in os.listdir():
    if file.endswith(".csv"):
        csv_files.append(file)


def remove_csv(s):
    return s.replace(".csv", "")


model_names = list(map(remove_csv, csv_files))


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
    x = re.findall("L6", string)
    if x[0] == "L6":
        l = "PPI and FDA"
    else:
        l = "Libraries not known"
    return l


def kernel(string):
    "identify employed kernel to train the model"
    if ["SVM1"] == re.findall("SVM1", string):
        id_kernel = "linear"
    elif ["SVM2"] == re.findall("SVM2", string):
        id_kernel = "poly"
    elif ["SVM3"] == re.findall("SVM3", string):
        id_kernel = "rbf"
    elif ["SVM4"] == re.findall("SVM4", string):
        id_kernel = "sigmoid"
    else:
        id_kernel = "No kernel information"
    return id_kernel


def class_weight(string):
    if ["ABN"] == re.findall("ABN", string):
        c = "True"
    else:
        c = "False"
    return c


# define empty dictionary
dict1 = dict()
print(dict1)
# populate the dictionary with model information
for i in range(len(model_names)):
    dict1["model name"] = model_names
    dict1["Data"] = list(map(population, model_names))
    dict1["Libraries"] = list(map(libraries, model_names))
    dict1["Algorithm"] = ["SVM" for i in range(len(model_names))]
    dict1["Kernel"] = list(map(kernel, model_names))
    dict1["class_weight"] = list(map(class_weight, model_names))

# iter throught a dictionary
# for i in range(len(model_names)):
# dict1[i] = {
#     "model name": model_names[i],
#     "Data": population(model_names[i]),
#     "Libraries": libraries(model_names[i]),
#     "Algorithm": "SVM",
#     "Kernel": kernel(model_names[i]),
#     "class_weight": class_weight(model_names[i]),
# }

# for i in range(len(model_names)):
#     print(dict1[i])

DF = pd.DataFrame.from_dict(
    data=dict1,
    # orient=index,
    # columns=["model name", "Data", "Libraries", "Algorithm", "Kernel", "class_weight"],
)
print(DF.head())
