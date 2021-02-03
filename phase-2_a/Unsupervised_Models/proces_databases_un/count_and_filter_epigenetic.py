"""remove not desired ppi family entries and keep just the epigentic ones"""

import os
import pandas as pd


root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info",
    "root_chem_space": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/chemical_space",
    "tsne_results": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/tSNE/tSNE_results",
    "tsne_info_params": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/tSNE/tSNE_results/info_params",
}


def filter(targets, file, output):
    # target, list that contain variables we want to keep
    data = pd.read_csv(f'{root["root"]}{"/"}{file}', index_col="Unnamed: 0")
    print(data.head())
    ppi = data[data["library"] == "PPI"]
    ppi_family = list(ppi["PPI family"].unique())
    # borrar las targets de la lista de PPI family
    for target in targets:
        ppi_family.remove(target)
    # PPI tiene las targets que se desean borrar
    # print(ppi_family)
    for i in ppi_family:
        print("borrar la diana ", i)
        # print(ppi[ppi["PPI family"] == i])
        indexNames = ppi[ppi["PPI family"] == i].index
        ppi.drop(indexNames, inplace=True)
    print(targets)
    print(ppi["PPI family"].unique())
    print(ppi.head())
    # pegar resultado con FDA
    fda = data[data["library"] == "FDA"]
    fda["PPI family"] = ["FDA" for i in range(fda.shape[0])]
    print(fda.head())
    final = pd.concat([fda, ppi], axis=0).reset_index()
    final = final.drop("index", axis=1)
    print(final.head())
    print("file", file, "has been procesed")
    final.to_csv(f'{root["root"]}{"/"}{output}')


targets = [
    # "MDM2-Like / P53",
    # "BCL2-Like / BAX",
    "Bromodomain / Histone",
    # "LFA / ICAM",
    # "XIAP / Smac",
    # "CD4 / gp120",
    # "LEDGF / IN",
    # "CD80 / CD28",
    # "TTR",
    "WDR5/MLL",
    "SPIN1 / H3",
    "MENIN / MLL",
]
file1 = "dataset_maccskeys_p2.csv"
file2 = "dataset_ecfp4_p2.csv"
file3 = "dataset_ecfp6_p2.csv"

filter(targets, file1, "epigenetic_maccskeys_tsne.csv")
filter(targets, file2, "epigenetic_ecfp4_tsne.csv")
filter(targets, file3, "epigenetic_ecfp6_tsne.csv")
