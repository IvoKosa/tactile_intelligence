import os, re, torch, random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import butter, filtfilt
from typing import List, Dict, Tuple

# ------------------------------------------------------------------------ Loading Functions

# Load Data from Single CSV path
def data_loader(csv_path, cropping, filtering, calibrated=True):
    df = pd.read_csv(csv_path) 
    df.set_index('timestamp', inplace=True)
    df = normalise_df_time(df)
    if filtering:
        df = butterworth_filter(df)
    if cropping:
        df = crop_df(df, center_pct=50, window_size=200) # Cropped to the sensor contact 
    else:
        df = crop_df(df, center_pct=50, window_size=425) # Cropped approx 50 datapoints off each end. Effectively full signal (win=700 for old data)
    if calibrated:
        df = df.drop(columns=df.columns[0 : 12])
    else:
        df = df.drop(columns=df.columns[12 : ])

    return df

# Normalise the time starting at 0 and dateTime -> seconds
def normalise_df_time(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        # df.index = pd.to_datetime(df.index)
        df.index = df.index = pd.to_datetime(df.index, format='mixed')
    t0 = df.index[0]
    secs = (df.index - t0).total_seconds()
    df.index = secs
    df.index.name = 't_seconds'
    return df

# Crop data around centre percentage and with given window size on either size
def crop_df(df, center_pct, window_size):
    n = len(df)
    pct = np.clip(center_pct, 0, 100)
    center_idx = int(n * pct / 100)
    start = max(center_idx - window_size, 0)
    end   = min(center_idx + window_size, n)
    return df.iloc[start:end]

def df_convert_unix_ts(csv_path, timestamp_col='timestamp_us'):
    df = pd.read_csv(csv_path)
    df[timestamp_col] = df[timestamp_col].apply(
        lambda x: (datetime(1970, 1, 1) + timedelta(microseconds=int(x))).isoformat()
    )
    return df
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

def dfs_to_tensor_nearest(df1: pd.DataFrame,
                          df2: pd.DataFrame):

    left  = df1.reset_index().rename(columns={'t_seconds':'t'}).sort_values('t')
    right = df2.reset_index().rename(columns={'t_seconds':'t'}).sort_values('t')
    merged = pd.merge_asof(
        left, right,
        on='t',
        direction='nearest'
    ).dropna()
    return torch.from_numpy(merged.drop(columns='t').T.values)

def get_class(material, texture):

    texture_list    = ['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry']
    material_list   = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50', 'rigid']
    # texture_list, material_list = get_cls_lists('data')

    material_index  = {s: i for i, s in enumerate(material_list)}
    texture_index   = {s: j for j, s in enumerate(texture_list)}

    i = material_index[material]
    j = texture_index[texture]
    return i * len(texture_index) + j

def get_class_dual(material, texture):

    texture_list    = ['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry']
    material_list   = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50', 'rigid']
    # texture_list, material_list = get_cls_lists('data')

    material_index  = {s: i for i, s in enumerate(material_list)}
    texture_index   = {s: j for j, s in enumerate(texture_list)}

    return material_index[material], texture_index[texture]

def get_cls_lists(root_dir):
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"Provided path '{root}' is not a directory.")
    
    texture = sorted([p.name for p in root.iterdir() if p.is_dir()])

    if texture:
        first_sub = root / texture[0]
        material = sorted([p.name for p in first_sub.iterdir() if p.is_dir()])
    else:
        material = []

    return texture, material

# ------------------------------------------------------------------------ Filtering

# Moving Average Filter
def sliding_window_avg_filter(raw_df, win_size=35):
    return raw_df.rolling(window=win_size, center=True).mean()

# First Derivative Calc
def first_deriv_filter(raw_df):
    dt = np.median(np.diff(raw_df.index.values))
    return raw_df.diff().divide(dt, axis=0)

# Butterworth Filter
def butterworth_filter(signal_df, order=4, cutoff_hz=20.0):
    dt = np.median(np.diff(signal_df.index.values))
    fs = 1.0 / dt
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False) # type: ignore
    filt_df = signal_df.copy()
    for col in signal_df.columns:
        filt_df[col] = filtfilt(b, a, signal_df[col].values)
    return filt_df

# ------------------------------------------------------------------------ Plotting

def sensor_plotter(signal1_df, 
                    signal2_df=None,
                    signal2_alpha=0.5,
                    plot_mode='taxel',
                    calibrated=True):
    
    has_filt = signal2_df is not None

    if has_filt:
        signal2_df = signal2_df.copy()
    else:
        signal2_df = None

    all_vals = signal1_df.values.flatten()
    if has_filt:
        all_vals = np.concatenate([all_vals, signal2_df.values.flatten()])                       # type: ignore
    y_min, y_max = np.nanmin(all_vals), np.nanmax(all_vals)

    def _make_subplots(nrows):
        fig, axes = plt.subplots(nrows=nrows, ncols=1,
                                 sharex=True, figsize=(10, 3*nrows))
        return axes if nrows>1 else [axes]

    taxel_range, data_dim = range(1,5), ['x','y','z']
    raw_alpha = signal2_alpha if has_filt else 1.0

    match plot_mode:
        case 'taxel':
            axes = _make_subplots(4)
            for ax, sensor in zip(axes, taxel_range):                                           # type: ignore
                for dim in data_dim:
                    if calibrated:
                        col = f'{dim}{sensor}_calib'
                    else:
                        col = f'{dim}{sensor}'
                    ax.plot(signal1_df.index, signal1_df[col], alpha=raw_alpha, lw=1, label=f'{col} (raw)')
                    if has_filt:
                        ax.plot(signal2_df.index, signal2_df[col], lw=2, label=f'{col} (filt)')             # type: ignore
                ax.set_ylim(y_min, y_max)
                ax.set_ylabel('Cilia Deformation')
                ax.set_title(f'Taxel {sensor}')
                ax.legend(loc='upper right', fontsize='small')
            axes[-1].set_xlabel('Time (s)')                                                     # type: ignore
            plt.tight_layout()
            plt.show()

        case 'xyz':
            axes = _make_subplots(3)
            for ax, dim in zip(axes, data_dim):                                                 # type: ignore
                for sensor in taxel_range:
                    if calibrated:
                        col = f'{dim}{sensor}_calib'
                    else:
                        col = f'{dim}{sensor}'
                    ax.plot(signal1_df.index, signal1_df[col], alpha=raw_alpha, lw=1, label=f'{col} (raw)')
                    if has_filt:
                        ax.plot(signal2_df.index, signal2_df[col], lw=2, label=f'{col} (filt)')             # type: ignore
                ax.set_ylim(y_min, y_max)
                ax.set_ylabel('Cilia Deformation')
                ax.set_title(f'Dimension: {dim.upper()}')
                ax.legend(loc='upper right', fontsize='small')
            axes[-1].set_xlabel('Time (s)')                                                     # type: ignore
            plt.tight_layout()
            plt.show()

        case _:
            raise ValueError("Choose plot_mode='taxel' or 'xyz'")
        
def run_plotter():
    pth         = 'data/ds20/bigberry/sensor0_data_20250722_161000.csv'
    calibrated  = True
    df          = data_loader(pth, cropping=False, filtering=False, calibrated=calibrated)
    # filt_df     = data_loader(pth, cropping=False, filtering=True, calibrated=calibrated)
    sensor_plotter(df, calibrated=calibrated)

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
    material_list = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50', 'rigid']
    texture_list  = ['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry']
    # texture_list, material_list = get_cls_lists('data')

    def _plot_cm(cm, classes, file_name):
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
        cmap = plt.cm.viridis.copy() # type: ignore                     
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
                    ax.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=6)

        plt.tight_layout()
        if plotting:
            plt.show()
        plt.savefig(file_name, dpi=300)
        plt.close(fig)

    # Plot material (5 classes)
    _plot_cm(mat_cm, material_list, mat_file_name)
    # Plot texture (6 classes)
    _plot_cm(tex_cm,  texture_list,  tex_file_name)

def loss_plots(save_pth,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            gap_loss,
            gap_acc,
            plotting=False):
    """
    Plot training/validation loss and accuracy, plus generalisation gaps.

    Args:
        train_loss (list of float): per‑epoch training loss
        val_loss   (list of float): per‑epoch validation loss
        train_acc  (list of float): per‑epoch training accuracy
        val_acc    (list of float): per‑epoch validation accuracy
        gap_loss   (list of float): per‑epoch loss gap (val_loss - train_loss)
        gap_acc    (list of float): per‑epoch accuracy gap (train_acc - val_acc)
    """

    # ——— Loss curves ———
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    if plotting:
        plt.show()
    plt.savefig(f'{save_pth}/loss_plot.png')

    # ——— Accuracy curves ———
    plt.figure()
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc,   label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    if plotting:
        plt.show()
    plt.savefig(f'{save_pth}/accuracy_plot.png')

    # ——— Generalisation gap ———
    plt.figure()
    plt.plot(gap_loss, label='Loss Gap (Val − Train)')
    plt.plot(gap_acc,  label='Acc Gap  (Train − Val)')
    plt.xlabel('Epoch')
    plt.title('Generalisation Gap Over Time')
    plt.legend()
    plt.grid(True)
    if plotting:
        plt.show()
    plt.savefig(f'{save_pth}/generalisation_plot.png')

# ------------------------------------------------------------------------ New Dataset Management 

def collect_file_info(root_dir, tex_classes, mat_classes):

    # Should return a list of dicts in the following form:
    # info_dict = {
    #     's0_file_pth'   : 'data/...',
    #     's1_file_pth'   : 'data/...',
    #     'tex_cls_str'   : 'bigberry',
    #     'tex_cls_int'   : '0',
    #     'mat_cls_str'   : 'ef10',
    #     'mat_cls_int'   : '1',
    #     'multigrasp'    : True
    # }

    dict_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            full_pth = os.path.join(dirpath, fname)
            if file_contains_str(full_pth, 'gripper_positions') or file_contains_str(full_pth, 'sensor1_data'):
                continue

            m = re.search(r"sensor(0)", full_pth)
            s1_path = full_pth[:m.start(1)] + '1' + full_pth[m.start(1)+1:] # type: ignore

            tex_cls, tex_idx = file_cls_finder(full_pth, tex_classes)
            mat_cls, mat_idx = file_cls_finder(full_pth, mat_classes)
            mult_bool        = file_contains_str(full_pth, 'multigrasp')
            gripper_idx      = get_file_index(dirpath, fname)
            gripper_pth      = rf'{dirpath}/gripper_positions_trial_{gripper_idx}.csv'

            data_dict = {
                's0_file_pth'   : full_pth,
                's1_file_pth'   : s1_path,
                'gripper_pos'   : gripper_pth,
                'tex_cls_str'   : tex_cls,
                'tex_cls_int'   : tex_idx,
                'mat_cls_str'   : mat_cls,
                'mat_cls_int'   : mat_idx,
                'multigrasp'    : mult_bool
            }
            dict_list.append(data_dict)
    # sort_dict = [item for item in dict_list if item.get('multigrasp') is True]
    # print(len(sort_dict))
    # print([item for item in dict_list if item.get('multigrasp') is True])
    return(dict_list)

def file_contains_str(file_str, search_str):
    return re.search(search_str, file_str, re.IGNORECASE) is not None

def file_cls_finder(file_str, cls_list):
    for cls in cls_list:
        if cls in file_str:
            sorted_cls = sorted(cls_list)
            position = sorted_cls.index(cls)
            return cls, position
    raise ValueError("None of the candidate strings were found in the path.")

def get_file_index(directory_path: str, filename: str) -> int:
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path!r}")
    ignore_prefixes = ('gripper', 'sensor1')
    files = []
    for entry in os.listdir(directory_path):
        if entry.startswith(ignore_prefixes):
            continue
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            files.append(entry)
    files.sort()
    try:
        return files.index(filename) + 1
    except ValueError:
        raise ValueError(f"File {filename!r} not found in directory (after filtering).")

def split_dataset(
    dataset,
    ratios,
    split,
    seed: int = 42
    ):

    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("`ratios` must be [test, train, val] and sum to 1.")
    if seed is not None:
        random.seed(seed)

    key = f"tex_cls_int"

    groups: Dict[int, List[Dict]] = {}
    for item in dataset:
        label = item.get(key)
        if label is None:
            raise KeyError(f"Item missing expected key {key!r}: {item}")
        groups.setdefault(label, []).append(item)

    test_set, train_set, val_set = [], [], []

    for items in groups.values():
        random.shuffle(items)
        n = len(items)
        n_test  = int(n * ratios[0])
        n_train = int(n * ratios[1])
        n_val   = n - n_test - n_train

        test_set.extend( items[:n_test] )
        train_set.extend(items[n_test:n_test + n_train])
        val_set.extend(  items[n_test + n_train:] )

    random.shuffle(test_set)
    random.shuffle(train_set)
    random.shuffle(val_set)

    if split == 'test':
        return test_set
    elif split == 'val':
        return val_set
    elif split == 'train':
        return train_set
    else:
        raise ValueError('Split must be either: test/train/val')

if __name__ == '__main__':

    root_dir = 'data'
    mat_classes    = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50', 'rigid']
    tex_classes    = ['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry']
    collect_file_info(root_dir, mat_classes, tex_classes)

    # tex, mat = get_cls_lists(root_dir)
    # print(tex)
    # print(mat)

    # pth         = 'data/ds20/bigberry/sensor0_data_20250722_161000.csv'
    # calibrated  = False
    # df          = data_loader(pth, cropping=False, filtering=False, calibrated=calibrated)
    # run_plotter()
    # confusion_plotter()
    # print(collect_files('data'))