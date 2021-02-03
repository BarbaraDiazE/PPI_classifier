"""chemical space of epigenetic modulators"""

import os
import pandas as pd
from tSNE_FP_epigenetic import TSNE_FP

root = {
    "root": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Databases/",
    "root_Info": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Supervised_Models/SVM/Info",
    "root_chem_space": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/epigenetic_chemical_space/tsne",
    "tsne_results": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/tSNE/tSNE_results",
    "tsne_info_params": "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-2_a/Unsupervised_Models/tSNE/tSNE_results/info_params",
}


print(str(root["root"]))
data = pd.read_csv(str(root["root"]) + str("dataset_ecfp4_p2.csv"))

numerical_data = data.drop(
    ["Unnamed: 0", "ipp_id", "chembl_id", "SMILES", "library", "PPI family", "PPI"],
    axis=1,
)
descriptors = numerical_data.columns.to_list()
print("number of descriptors: ", len(descriptors))


def execute(
    root, input_file, target, descriptors, ref_output, perplexity, n_iter, random
):
    a = TSNE_FP(root, input_file, target, descriptors, perplexity, n_iter, random)
    # a.eda()
    a.plot_matplotlib(ref_output)
    print("results ", str(ref_output), "are saved")


# execute(
#     root,
#     "ecfp4_tsne.csv",
#     "PPI",
#     descriptors,
#     "p2_ecfp4_tsne_p10_n1000_r1992",
#     10,
#     1000,
#     1992,
# )
# execute(
#     root,
#     "ecfp4_tsne.csv",
#     "PPI",
#     descriptors,
#     "p2_ecfp4_tsne_p10_n1500_r1992",
#     10,
#     1500,
#     1992,
# )
# execute(
#     root,
#     "ecfp4_tsne.csv",
#     "PPI",
#     descriptors,
#     "p2_ecfp4_tsne_p10_n2000_r1992",
#     10,
#     2000,
#     1992,
# )
execute(
    root,
    "epigenetic_ecfp4_tsne.csv",
    "PPI",
    descriptors,
    "epigenetic_p2_ecfp4_tsne_p30_n1000_r1992",
    30,
    1000,
    1992,
)
execute(
    root,
    "epigenetic_ecfp4_tsne.csv",
    "PPI",
    descriptors,
    "epigenetic_p2_ecfp4_tsne_p30_n1500_r1992",
    30,
    1500,
    1992,
)
execute(
    root,
    "epigenetic_ecfp4_tsne.csv",
    "PPI",
    descriptors,
    "epigenetic_p2_ecfp4_tsne_p30_n2000_r1992",
    30,
    2000,
    1992,
)
execute(
    root,
    "epigenetic_ecfp4_tsne.csv",
    "PPI",
    descriptors,
    "epigenetic_p2_ecfp4_tsne_p50_n1000_r1992",
    50,
    1000,
    1992,
)
execute(
    root,
    "epigenetic_ecfp4_tsne.csv",
    "PPI",
    descriptors,
    "epigenetic_p2_ecfp4_tsne_p50_n1000_r1992",
    50,
    1000,
    1992,
)
execute(
    root,
    "epigenetic_ecfp4_tsne.csv",
    "PPI",
    descriptors,
    "epigenetic_p2_ecfp4_tsne_p50_n1500_r1992",
    50,
    1500,
    1992,
)
execute(
    root,
    "epigenetic_ecfp4_tsne.csv",
    "PPI",
    descriptors,
    "epigenetic_p2_ecfp4_tsne_p50_n2000_r1992",
    50,
    2000,
    1992,
)
execute(
    root,
    "epigenetic_ecfp4_tsne.csv",
    "PPI",
    descriptors,
    "epigenetic_p2_ecfp4_tsne_p50_n2000_r2020",
    50,
    2000,
    2020,
)

