import os, re, torch, random
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import butter, filtfilt
from typing import List, Dict, Tuple

# ------------------------------------------------------------------------ Dataset Formating Functions

def collect_files_old(root_dir, split_str='train', split_distribution=[0.7, 0.2, 0.1]):
    """
    Return a list of [full_file_path_s0, full_file_path_s2, subfolder_1, subfolder_2] for every file
    found two levels down from root_dir.
    """

    # validate args
    assert split_str in ('train', 'test', 'val'), \
        "Split String must be one of: 'train', 'test' or 'val'"
    assert abs(sum(split_distribution) - 1.0) < 1e-6, \
        'split_distribution must sum to 1.0'

    entries = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname[0] == 'g': # Filtering out gipper data files
                continue
            if fname[6] == '0':
                full_path_s0 = os.path.join(dirpath, fname)
                fname_s1 = fname[:6] + '1' + fname[7:]
                full_path_s1 = os.path.join(dirpath, fname_s1)
                rel_parts = os.path.relpath(full_path_s0, root_dir).split(os.sep)
                if len(rel_parts) == 3:
                    sub1, sub2, _ = rel_parts
                    entries.append([full_path_s0, full_path_s1, sub1, sub2])

    return entries

def collect_files(root_dir, split_str, split_distribution):
    """
    Crawl two levels deep under root_dir and collect all pairs of files
    (s0, s1) along with their sub1 and sub2 labels. Then stratify by
    (sub1, sub2), sort deterministically within each group, and split into
    train/test/val according to split_distribution. Finally, return only the
    list for split_str.

    Returns a list of [full_path_s0, full_path_s1, sub1, sub2].
    """
    # validate args
    assert split_str in ('train', 'test', 'val'), \
        "Split String must be one of: 'train', 'test' or 'val'"
    assert abs(sum(split_distribution) - 1.0) < 1e-6, \
        'split_distribution must sum to 1.0'

    # 1) collect all entries
    all_entries = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.startswith('g'):  # skip gipper data files
                continue
            # only look at files where 7th char == '0'
            if len(fname) > 6 and fname[6] == '0':
                full_s0 = os.path.join(dirpath, fname)
                # build the paired filename
                fname_s1 = fname[:6] + '1' + fname[7:]
                full_s1 = os.path.join(dirpath, fname_s1)
                # get sub1/sub2 from relative path
                rel = os.path.relpath(full_s0, root_dir).split(os.sep)
                if len(rel) == 3:
                    sub1, sub2, _ = rel
                    all_entries.append([full_s0, full_s1, sub1, sub2])

    # 2) group by (sub1, sub2)
    groups = defaultdict(list)
    for entry in all_entries:
        key = (entry[2], entry[3])
        groups[key].append(entry)

    # mapping for split index
    idx_map = {'train': 0, 'test': 1, 'val': 2}
    out = []

    # 3) for each group, sort deterministically and split
    for grp_entries in groups.values():
        # sort by s0 path for reproducibility
        grp_sorted = sorted(grp_entries, key=lambda x: x[0])
        n = len(grp_sorted)
        n_train = int(split_distribution[0] * n)
        n_test  = int(split_distribution[1] * n)

        train_slice = grp_sorted[:n_train]
        test_slice  = grp_sorted[n_train:n_train + n_test]
        val_slice   = grp_sorted[n_train + n_test:]

        splits = {
            'train': train_slice,
            'test':  test_slice,
            'val':   val_slice
        }
            
        out.extend(splits[split_str])

    return out

def num_dir_classes(root_dir):
    unique_names = set()
    for _, dirs, _ in os.walk(root_dir):
        for d in dirs:
            unique_names.add(d)
    return len(unique_names)

def confusion_plotter_dual(mat_cm, tex_cm,
                           mat_file_name, tex_file_name,
                           plotting=False, normalize='true'):
    """
    Plot and save two confusion‐matrix heatmaps: one for material (6×6),
    one for texture (6×6).

    Args:
        mat_cm (ndarray):    material confusion matrix (shape [6,6])
        tex_cm (ndarray):    texture confusion matrix (shape [6,6])
        mat_file_name (str): where to save the material plot
        tex_file_name (str): where to save the texture plot
        plotting (bool):     whether to plt.show() each
        normalize (str):     'true' (row‐wise), 'all', or anything else (counts)
    """
    material_list = sorted(['ds20', 'ds30', 'ef10', 'ef30', 'ef50'])     
    texture_list  = sorted(['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry'])

    # texture_list, material_list = get_cls_lists('data')

    def _plot_cm(cm, classes, file_name):
        print(cm)
        print(classes)
        # Normalize
        if normalize.lower() == 'true':
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_pct = (cm.astype(float) / row_sums) * 100
        elif normalize.lower() == 'all':
            total = cm.sum() or 1
            cm_pct = (cm.astype(float) / total) * 100
        else:
            cm_pct = cm.astype(float)

        # Mask zeros
        cm_mask = np.ma.masked_where(cm_pct == 0, cm_pct)
        # cmap = plt.cm.viridis.copy() # type: ignore
        cmap = plt.cm.Blues.copy() # type: ignore                    
        cmap.set_bad(color='white')

        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(cm_mask, cmap=cmap, vmin=0, vmax=100)
        cbar = fig.colorbar(cax)
        cbar.ax.set_ylabel('Percentage', rotation=270, labelpad=15)
        cbar.ax.yaxis.set_major_formatter(lambda x, pos: f'{x:.0f}%')

        n = len(classes)
        # gridlines
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle=':', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)

        # axis labels
        ax.set_xticks(range(n))
        ax.set_xticklabels(classes, rotation=90, fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(classes, fontsize=8)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.set_xlabel('Predicted label', labelpad=10)
        ax.set_ylabel('True label')

        # annotate
        for i in range(n):
            for j in range(n):
                if cm[i, j] != 0:
                    val = round(cm_pct[i, j], 1)
                    if str(val)[-1] == '0':
                        val = int(val)
                    if val <= 50:
                        ax.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=10)
                    else:
                        ax.text(j, i, f'{val}', ha='center', va='center', color='white', fontsize=10)

        plt.tight_layout()
        if plotting:
            plt.show()
        plt.savefig(file_name, dpi=300)
        plt.close(fig)

    # Plot material (5 classes)
    _plot_cm(mat_cm, material_list, mat_file_name)
    # Plot texture (6 classes)
    _plot_cm(tex_cm,  texture_list,  tex_file_name)

def confusion_plotter(cm, file_name, plotting=False, normalize='true'):
    # Class labels
    class_names = [
        "ds20 bigberry", "ds20 citrus", "ds20 rough", "ds20 smallberry", "ds20 smooth", "ds20 strawberry",
        "ds30 bigberry", "ds30 citrus", "ds30 rough", "ds30 smallberry", "ds30 smooth", "ds30 strawberry",
        "ef10 bigberry", "ef10 citrus", "ef10 rough", "ef10 smallberry", "ef10 smooth", "ef10 strawberry",
        "ef30 bigberry", "ef30 citrus", "ef30 rough", "ef30 smallberry", "ef30 smooth", "ef30 strawberry",
        "ef50 bigberry", "ef50 citrus", "ef50 rough", "ef50 smallberry", "ef50 smooth", "ef50 strawberry"
    ]

    # Compute percentages
    if normalize == 'true':
        # Normalize per true label (row-wise)
        sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        sums[sums == 0] = 1
        cm_percent = (cm.astype(float) / sums) * 100
    elif normalize == 'all':
        # Normalize over entire matrix
        total = cm.sum()
        cm_percent = (cm.astype(float) / total) * 100
    else:
        # No normalization, treat values as-is
        cm_percent = cm.astype(float)

    # Mask zeros for display
    cm_masked = np.ma.masked_where(cm_percent == 0, cm_percent)
    cmap = plt.cm.viridis.copy()                        # type: ignore
    cmap.set_bad(color='white')

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(cm_masked, cmap=cmap)
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel('Percentage', rotation=270, labelpad=15)
    # Format colorbar ticks as percentages
    cbar.ax.yaxis.set_major_formatter(lambda x, pos: f'{x:.0f}%')

    # Add white dotted gridlines separating each cell
    n = cm.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle=':', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names, fontsize=6)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('Predicted label', labelpad=10)
    ax.set_ylabel('True label')

    # Annotate each cell with percentage
    # Use contrasting text color based on background intensity
    # max_val = cm_percent.max()
    for i in range(n):
        for j in range(n):
            val = cm_percent[i, j]
            if cm[i, j] != 0:
                # text_color = 'white' if val > max_val / 2 else 'black'
                text_color = 'black'
                # val = int(val)
                ax.text(j, i, f'{val}', ha='center', va='center', color=text_color, fontsize=4)

    plt.tight_layout()
    if plotting:
        plt.show()
    plt.savefig(file_name, dpi=300)


# def run_plotter():
#     pth         = 'data/ds20/bigberry/sensor0_data_20250722_161000.csv'
#     calibrated  = True
#     df          = data_loader(pth, cropping=False, filtering=False, calibrated=calibrated)
#     # filt_df     = data_loader(pth, cropping=False, filtering=True, calibrated=calibrated)
#     sensor_plotter(df, calibrated=calibrated)