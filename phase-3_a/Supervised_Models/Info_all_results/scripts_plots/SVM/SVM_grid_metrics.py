route = "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/SVM/"

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from SVM_filter_metrics import AC


# plot with
fig = plt.figure(figsize=(20, 12))

# precision
ax1 = fig.add_subplot(2, 2, 1)
x1, y1, X1 = AC().precision()
ax1.plot(x1, y1, "ro", color="magenta", alpha=0.7)
ax1.grid(color="lightgray", axis="y", linestyle="dotted", linewidth=2)
ax1.set_xticks(x1)
ax1.set_xticklabels(X1, rotation="vertical", fontsize=8)
ax1.set_ylabel("Precision")
ax1.set_ylim([0.5, 1.0])
ax1.set_title("A)", loc="left")

# balanced acc
ax2 = fig.add_subplot(2, 2, 2)
x2, y2, X2 = AC().balanced_acc()
ax2.plot(x2, y2, "ro", color="darkviolet", alpha=0.7)
ax2.grid(color="lightgray", axis="y", linestyle="dotted", linewidth=2)
ax2.set_xticks(x2)
ax2.set_xticklabels(X2, rotation="vertical", fontsize=8)
ax2.set_ylabel("Balanced Accuracy")
ax2.set_ylim([0.5, 1.0])
ax2.set_title("B)", loc="left")


# f1
ax3 = fig.add_subplot(2, 2, 3)
x3, y3, X3 = AC().f1()
ax3.plot(x3, y3, "ro", color="yellowgreen", alpha=0.7)
ax3.grid(color="lightgray", axis="y", linestyle="dotted", linewidth=2)
ax3.set_xticks(x3)
ax3.set_xticklabels(X3, rotation="vertical", fontsize=8)
ax3.set_ylabel("f1")
ax3.set_ylim([0.5, 1.0])
ax3.set_title("C)", loc="left")

# recall
ax4 = fig.add_subplot(2, 2, 4)
x4, y4, X4 = AC().recall()
ax4.plot(x4, y4, "ro", color="deepskyblue", alpha=0.7)
ax4.grid(color="lightgray", axis="y", linestyle="dotted", linewidth=2)
ax4.set_xticks(x4)
ax4.set_xticklabels(X4, rotation="vertical", fontsize=8)
ax4.set_ylabel("Recall")
ax4.set_ylim([0.5, 1.0])
ax4.set_title("D)", loc="left")

# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
fig.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
fig.subplots_adjust(
    top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=0.25, wspace=0.15
)
plt.savefig(
    "/home/babs/Documents/DIFACQUIM/PPI_classifier/phase-3_a/Supervised_Models/Info_all_results/plot_metrics/SVM_p3.png",
    dpi=200,
)
plt.show()
