import matplotlib.pyplot as plt
import numpy as np

# ---------------------- V1 Data ----------------------
# CNN:
OLD_CNN_mat_accs = [98, 80, 86, 76, 73]
OLD_CNN_tex_accs = [99, 88, 84, 80, 85]
OLD_CNN_all_accs = [97, 70, 75, 60, 63]

# AE:
OLD_AE_mat_accs = [96, 79, 84, 73, 69]
OLD_AE_tex_accs = [99, 92, 77, 79, 82]
OLD_AE_all_accs = [95, 76, 65, 59, 56]

# ---------------------- V2 Data ----------------------

CNN_mat_accs = [94, 81, 85, 71, 72]
CNN_tex_accs = [100, 87, 85, 84, 84]
CNN_all_accs = [94, 75, 74, 61, 59]

CNN_grouped_mat_accs = [96, 68, 87, 66, 66]
CNN_grouped_tex_accs = [97, 88, 87, 79, 79]
CNN_grouped_all_accs = [93, 64, 78, 52, 51]

CAE_mat_accs = [97, 82, 84, 72, 73]
CAE_tex_accs = [98, 89, 83, 82, 86]
CAE_all_accs = [95, 77, 72, 59, 62]

# ---------------------- V2 Data ----------------------

CNN_mat = [98, 81, 86, 74, 71]
CNN_tex = [92, 92, 77, 85, 83]
CNN_all = [91, 78, 69, 65, 60]

CAE_mat = [95, 88, 78, 84, 73]
CAE_tex = [96, 82, 84, 73, 85]
CAE_all = [90, 76, 69, 62, 62]

runs = ['2 Mat, 2 Tex', '3 Mat, 3 Tex', '4 Mat, 4 Tex', '5 Mat, 5 Tex', '5 Mat, 6 Tex']

# Single Plot
plt.figure(figsize=(8, 5))
plt.plot(runs, CAE_mat, marker='o', label='Material Accuracies')
plt.plot(runs, CAE_tex, marker='s', label='Texture Accuracies')
plt.plot(runs, CAE_all, marker='^', label='All Accuracies')

# Labels & Title
plt.xlabel("Class Structure (Materials & Textures)")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracies Across Class Structures")
plt.ylim(50, 100)  # accuracy is % scale
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

# Plot
# plt.figure(figsize=(10, 6))

# CNN curves
# plt.plot(runs, CNN_mat_accs, marker='o', label='CNN - Material')
# plt.plot(runs, CNN_tex_accs, marker='s', label='CNN - Texture')
# plt.plot(runs, CNN_all_accs, marker='^', label='CNN - Overall')

# # AE curves
# plt.plot(runs, AE_mat_accs, marker='o', linestyle='--', label='AE - Material')
# plt.plot(runs, AE_tex_accs, marker='s', linestyle='--', label='AE - Texture')
# plt.plot(runs, AE_all_accs, marker='^', linestyle='--', label='AE - Overall')

# # Formatting
# plt.xlabel("Class Structure (Materials & Textures)")
# plt.ylabel("Accuracy (%)")
# plt.title("CNN vs AE Accuracy Across Class Structures")
# plt.ylim(0, 100)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()

# plt.tight_layout()
# plt.show()
