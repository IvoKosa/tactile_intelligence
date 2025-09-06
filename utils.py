import os, re, torch
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
from scipy.signal import butter, filtfilt

# ****************************** Utils.py ******************************
#
# > Contains various helper functions for all model testing/ training 
# > Additionally contains some plotting functions for loss curves/ confusion matrices

# ------------------------------------------------------------------------ Loading Functions

# Load Data from Single CSV path
def data_loader(csv_path, cropping, filtering, calibrated=True, euc_norm=False):
    df = pd.read_csv(csv_path) 
    df.set_index('timestamp', inplace=True)
    df = normalise_df_time(df)

    if filtering:
        df = butterworth_filter(df)

    if cropping:
        df = crop_signal(df, centre_sec=2.7, window_size=120) # Cropped to the sensor contact 
    else:
        df = crop_signal(df, centre_sec=2.7, window_size=420) # Cropped approx 50 datapoints off each end. Effectively full signal
    
    if calibrated:
        df = df.drop(columns=df.columns[0 : 12])
    else:
        df = df.drop(columns=df.columns[12 : ])

    if euc_norm:
        df = collapse_xyz_to_norm(df)

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

# Crop signal to specified window around centre point specified in seconds
def crop_signal(df, centre_sec, window_size):
    idx_vals = df.index.to_numpy()
    pos = int(np.argmin(np.abs(idx_vals - centre_sec)))
    start = max(pos - window_size, 0)
    end   = min(pos + window_size, len(df))
    return df.iloc[start:end]

# Used for testing XYZ concatenation methods
def collapse_xyz_to_norm(df):
    if df.shape[1] % 3 != 0:
        raise ValueError("Number of columns must be divisible by 3 (x,y,z groups).")
    
    n_groups = df.shape[1] // 3
    normed_data = {}
    
    for i in range(n_groups):
        cols = df.iloc[:, 3*i : 3*i + 3]
        normed_data[f'norm_{i}'] = np.linalg.norm(cols.values, axis=1)
    
    return pd.DataFrame(normed_data, index=df.index)

# ------------------------------------------------------------------------ New Dataset Management 

# Central file collection system for pytorch dataset takes in root directory and desired texture/ material classes
# Generates a list of dictionaries, each one containing all file path and class information 
def collect_file_info(root_dir, tex_classes, mat_classes, singlegrasp_limit=20):
    tex_classes = sorted(tex_classes)
    mat_classes = sorted(mat_classes)
    tex_max     = [0] * len(tex_classes)
    mat_max     = [0] * len(mat_classes)

    cls_max     = [0] * len(tex_classes)  * len(mat_classes)
    counter = 0

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

            if tex_idx is None or mat_idx is None:
                continue

            if file_contains_str(dirpath, '200') and singlegrasp_limit is not None:
                cls_idx = mat_idx * len(tex_classes) + tex_idx
                cls_max[cls_idx] += 1
                if cls_max[cls_idx] > singlegrasp_limit:
                    continue
                else:
                    counter += 1

            mult_bool        = file_contains_str(full_pth, 'multigrasp')
            gripper_idx      = get_file_index(dirpath, fname)
            gripper_pth      = rf'{dirpath}/gripper_positions_trial_{gripper_idx}.csv'

            if mult_bool:
                pos = None
                for char in ['1', '2', 'l', 'm', 'r']:
                    if char == dirpath[-1]:
                        pos = char
                        break
                if pos == None:
                    pos = None
                if pos == '1' or pos == '2':
                    pos = 'h' + pos
                multi_grasp_pos     = pos
            else:
                multi_grasp_pos     = None

            data_dict = {
                's0_file_pth'   : full_pth,
                's1_file_pth'   : s1_path,
                'gripper_pos'   : gripper_pth,
                'tex_cls_str'   : tex_cls,
                'tex_cls_int'   : tex_idx,
                'mat_cls_str'   : mat_cls,
                'mat_cls_int'   : mat_idx,
                'multigrasp'    : mult_bool,
                'grasp_pos'     : multi_grasp_pos
            }
            dict_list.append(data_dict)
    # print(counter)
    return(dict_list)

# File collection helper functions
def file_contains_str(file_str, search_str):
    return re.search(search_str, file_str, re.IGNORECASE) is not None

# File collection helper functions
def file_cls_finder(file_str, cls_list):
    for cls in cls_list:
        if cls in file_str:
            sorted_cls = sorted(cls_list)
            position = sorted_cls.index(cls)
            return cls, position
        
    return (None, None)

# File collection helper functions
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
    
# ------------------------------------------------------------------------ Dataset Normalisation 

# Central function for computing dataset mean and standard deviation
# Takes in a pytorch dataset and returns dataset mean and std
def compute_dataset_mean_std(dataset, batch_size=32, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True)

    n_channels = None
    s1 = 0.0  
    s2 = 0.0  
    n  = 0    

    for x, *_ in loader:
        B, C = x.shape[:2]
        if n_channels is None:
            n_channels = C
        x_flat = x.view(B, C, -1)
        s1 += x_flat.sum(dim=(0,2))         
        s2 += (x_flat ** 2).sum(dim=(0,2)) 
        n  += B * x_flat.size(2)

    mean = s1 / n
    var  = (s2 / n) - mean**2
    std  = torch.sqrt(var) # type: ignore
    return mean, std

# Normalisation helper function
def zscore_denormalize(x, mean, std):
    C = x.shape[1]
    mean = mean.view(1, C, 1).to(x.device)
    std  = std.view(1, C, 1).to(x.device)
    return x * std + mean

# ------------------------------------------------------------------------ Dataset management helper functions 

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
def butterworth_filter(signal_df, order=4, cutoff_hz=50.0):
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
                    calibrated=True,
                    save_pth=None):
    
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
            def _make_subplots_2x2():
                fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(20, 8))
                return fig, axes.ravel()
            
            def shift_xaxis(x, pos):
                return f"{x + 315:.0f}" 

            taxel_range, data_dim = range(1, 5), ['x', 'y', 'z']
            raw_alpha = signal2_alpha if has_filt else 1.0

            # --- main plotting: now uses 2x2 grid ---
            fig, axes = _make_subplots_2x2()
            for ax, sensor in zip(axes, taxel_range):
                for dim in data_dim:
                    col = f'{dim}{sensor}_calib' if calibrated else f'{dim}{sensor}'
                    ax.plot(signal1_df.index, signal1_df[col], alpha=raw_alpha, lw=1, label=f'{col[:-6]}')
                    if has_filt:
                        ax.plot(signal2_df.index, signal2_df[col], lw=2, label=f'{col[:-6]} Reconstructed') # type: ignore
                ax.set_ylim(y_min, y_max)
                ax.set_ylabel('Cilia Deformation')
                ax.set_title(f'Taxel {sensor}')
                ax.legend(loc='upper right', fontsize='small')

            # Label x-axis only on the bottom row (axes[2] and axes[3])
            axes[2].set_xlabel('Time steps')
            axes[2].xaxis.set_major_formatter(FuncFormatter(shift_xaxis))
            axes[3].set_xlabel('Time steps')

            plt.tight_layout()
            if save_pth is not None:
                plt.savefig(save_pth)
                plt.close(fig)
            else:
                plt.show()

        case 'xyz':
            axes = _make_subplots(3)
            for ax, dim in zip(axes, data_dim):                                                 # type: ignore
                for sensor in taxel_range:
                    if calibrated:
                        col = f'{dim}{sensor}_calib'
                    else:
                        col = f'{dim}{sensor}'
                    ax.plot(signal1_df.index, signal1_df[col], alpha=raw_alpha, lw=1, label=f'{col[:-6]}')
                    if has_filt:
                        ax.plot(signal2_df.index, signal2_df[col], lw=2, label=f'{col[:-6]} Reconstructed')             # type: ignore
                ax.set_ylim(y_min, y_max)
                ax.set_ylabel('Cilia Deformation')
                ax.set_title(f'Dimension: {dim.upper()}')
                ax.legend(loc='upper right', fontsize='small')
            axes[-1].set_xlabel('Time (s)')                                                     # type: ignore
            plt.tight_layout()
            if save_pth is not None:
                plt.savefig(save_pth)
            else:
                plt.show()

        case _:
            raise ValueError("Choose plot_mode='taxel' or 'xyz'")

def plot_confusion_matrix(cm, classes, file_name, plotting=False, normalize='true'):
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
                val = round(cm_pct[i, j])
                if str(val)[-1] == '0':
                    val = int(val)
                ax.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=7)
                # if val <= 50:
                #     ax.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=10)
                # else:
                #     ax.text(j, i, f'{val}', ha='center', va='center', color='white', fontsize=10)

    plt.tight_layout()
    if plotting:
        plt.show()
    plt.savefig(file_name, dpi=400)
    plt.close(fig)

def loss_plots(save_pth,
            train_loss,
            val_loss,
            train_acc=None,
            val_acc=None,
            gap_loss=None,
            gap_acc=None,
            plotting=False,
            stopping_epoch=None):

    # ——— Loss curves ———
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    if stopping_epoch is not None:
        ymin, ymax = plt.gca().get_ylim()
        plt.vlines(stopping_epoch, ymin=ymin, ymax=ymax, color='red', linestyle='--')
        plt.text(stopping_epoch, plt.ylim()[0] - 1, f"{int(stopping_epoch)}", ha="center", va="top", color="red", bbox=dict(facecolor='white', edgecolor='none', pad=5.0))
    if plotting:
        plt.show()
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.savefig(f'{save_pth}/loss_plot.png')

    # ——— Accuracy curves ———
    if train_acc is not None and val_acc is not None:
        plt.figure()
        plt.plot(train_acc, label='Train Acc')
        plt.plot(val_acc,   label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.legend()
        plt.grid(True)
        if stopping_epoch is not None:
            ymin, ymax = plt.gca().get_ylim()
            plt.vlines(stopping_epoch, ymin=ymin, ymax=ymax, color='red', linestyle='--')
            plt.text(stopping_epoch, plt.ylim()[0] - 1, f"{int(stopping_epoch)}", ha="center", va="top", color="red", bbox=dict(facecolor='white', edgecolor='none', pad=5.0))
        if plotting:
            plt.show()
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))
        plt.savefig(f'{save_pth}/accuracy_plot.png')

    # ——— Generalisation gap ———
    if gap_loss is not None and gap_acc is not None:
        plt.figure()
        plt.plot(gap_loss, label='Loss Gap (Val − Train)')
        plt.plot(gap_acc,  label='Acc Gap  (Train − Val)')
        plt.xlabel('Epoch')
        plt.title('Generalisation Gap Over Time')
        plt.legend()
        plt.grid(True)
        if stopping_epoch is not None:
            ymin, ymax = plt.gca().get_ylim()
            plt.vlines(stopping_epoch, ymin=ymin, ymax=ymax, color='red', linestyle='--')
            plt.text(stopping_epoch, plt.ylim()[0] - 1, f"{int(stopping_epoch)}", ha="center", va="top", color="red", bbox=dict(facecolor='white', edgecolor='none', pad=5.0))
        if plotting:
            plt.show()
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))
        plt.savefig(f'{save_pth}/generalisation_plot.png')

def compute_pairwise_accuracy(mat_true, mat_pred, tex_true, tex_pred, mat_classes, tex_classes):
    mat_true = np.array(mat_true)
    mat_pred = np.array(mat_pred)
    tex_true = np.array(tex_true)
    tex_pred = np.array(tex_pred)

    num_mat = len(mat_classes)
    num_tex = len(tex_classes)

    # Create dictionaries to track results per pair
    accuracy_table = np.full((num_tex, num_mat), np.nan)  # texture on rows, material on columns

    # Group by (material, texture)
    for m in range(num_mat):
        for t in range(num_tex):
            indices = np.where((mat_true == m) & (tex_true == t))[0]
            if len(indices) == 0:
                continue

            mat_correct = (mat_pred[indices] == mat_true[indices]).astype(np.float32)
            tex_correct = (tex_pred[indices] == tex_true[indices]).astype(np.float32)

            # Compute joint accuracy (or average of the two)
            joint_accuracy = (mat_correct + tex_correct) / 2.0
            accuracy_table[t, m] = np.mean(joint_accuracy)

    return accuracy_table

def plot_joint_accuracy_table(accuracy_table, mat_classes, tex_classes, save_path):
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        accuracy_table, 
        xticklabels=mat_classes, 
        yticklabels=tex_classes, 
        annot=True, 
        fmt=".2f", 
        cmap='Blues', 
        cbar_kws={'label': 'Avg Accuracy'}
    )
    ax.set_xlabel("Material Classes")
    ax.set_ylabel("Texture Classes")
    ax.set_title("Joint Accuracy Table: Material (X) vs Texture (Y)")

    plt.tight_layout()

    plt.savefig(save_path)

if __name__ == '__main__':
    filepath    = r'data_final/multigrasp_train/ef30_multigrasp/smallberry_ef30/smallberry_ef30m/sensor1_data_20250730_155016.csv'
    signal_df   = data_loader(filepath, cropping=False, filtering=False, calibrated=True)
    cropped_df  = data_loader(filepath, cropping=False, filtering=True, calibrated=True)
    print(signal_df.head())
    sensor_plotter(signal_df, cropped_df, calibrated=True)