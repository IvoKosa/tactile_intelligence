import matplotlib.pyplot as plt
import numpy as np

# CNN:
CNN_mat_accs = [98, 80, 86, 76, 73]
CNN_tex_accs = [99, 88, 84, 80, 85]
CNN_all_accs = [97, 70, 75, 60, 63]

# AE:
AE_mat_accs = [96, 79, 84, 73, 69]
AE_tex_accs = [99, 92, 77, 79, 82]
AE_all_accs = [95, 76, 65, 59, 56]

runs = ['2 Mat, 2 Tex', '3 Mat, 3 Tex', '4 Mat, 4 Tex', '5 Mat, 5 Tex', '5 Mat, 6 Tex']

# Single Plot
# plt.figure(figsize=(8, 5))
# plt.plot(runs, mat_accs, marker='o', label='Material Accuracies')
# plt.plot(runs, tex_accs, marker='s', label='Texture Accuracies')
# plt.plot(runs, all_accs, marker='^', label='All Accuracies')

# # Labels & Title
# plt.xlabel("Class Structure (Materials & Textures)")
# plt.ylabel("Accuracy (%)")
# plt.title("Model Accuracies Across Class Structures")
# plt.ylim(0, 100)  # accuracy is % scale
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()

# # Show plot
# plt.tight_layout()
# plt.show()

# Plot
plt.figure(figsize=(10, 6))

# CNN curves
plt.plot(runs, CNN_mat_accs, marker='o', label='CNN - Material')
plt.plot(runs, CNN_tex_accs, marker='s', label='CNN - Texture')
plt.plot(runs, CNN_all_accs, marker='^', label='CNN - Overall')

# AE curves
plt.plot(runs, AE_mat_accs, marker='o', linestyle='--', label='AE - Material')
plt.plot(runs, AE_tex_accs, marker='s', linestyle='--', label='AE - Texture')
plt.plot(runs, AE_all_accs, marker='^', linestyle='--', label='AE - Overall')

# Formatting
plt.xlabel("Class Structure (Materials & Textures)")
plt.ylabel("Accuracy (%)")
plt.title("CNN vs AE Accuracy Across Class Structures")
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
