import matplotlib.pyplot as plt
import numpy as np

def average_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must be of equal length")
    return [(a + b) / 2 for a, b in zip(list1, list2)]


CNN_mat = [98, 81, 86, 74, 71]
CNN_tex = [92, 92, 77, 85, 83]
CNN_all = [91, 78, 69, 65, 60]

CAE_mat = [95, 88, 78, 84, 73]
CAE_tex = [96, 82, 84, 73, 85]
CAE_all = [90, 76, 69, 62, 62]

mid_lst = average_lists(CNN_mat, CNN_tex)

runs = ['2 Mat, 2 Tex', '3 Mat, 3 Tex', '4 Mat, 4 Tex', '5 Mat, 5 Tex', '5 Mat, 6 Tex']

plt.figure(figsize=(8, 5))
plt.plot(runs, CNN_mat, marker='o', label='Material Accuracies', alpha=0.5)
plt.plot(runs, CNN_tex, marker='s', label='Texture Accuracies', alpha=0.5)
plt.plot(runs, CNN_all, marker='^', label='All Accuracies')
# plt.plot(runs, mid_lst, marker='x', label='Average Accuracies')

# Labels & Title
plt.xlabel("Class Structure (Materials & Textures)")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracies Across Class Structures")
plt.ylim(50, 100)  # accuracy is % scale
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

ymin, ymax = plt.gca().get_ylim()
plt.vlines(2, ymin=ymin, ymax=ymax, color='red', linestyle='--')
plt.text(2, plt.ylim()[0] - 1, f"{int(2)}", ha="center", va="top", color="red", bbox=dict(facecolor='white', edgecolor='none', pad=5.0))

# Show plot
plt.tight_layout()
plt.show()
