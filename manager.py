# General Imports
import json, os, itertools, random
import numpy as np
import pandas as pd

# Model Imports
import model_CNN, model_CAE, model_LSTM
import signal_dataset, utils

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Setting fixed seeds for more equal model comparison
seed = 935248
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
g = torch.Generator().manual_seed(seed)

# ******************************************* Manager Class *******************************************
# 
# > Responsible for all aspects of training/ testing and managing a model
# > Automatically generates new folders, as well as figures and reports
# 
#   Parameters:
#       > model: Pytorch model architecture
#       > file_pth: folder for saving model weights and train/test figures
#       > num_epochs, batch_size, shuffle, lr, weight_decay: standard machine learning hyperparameters
#
#       > multigrasp : filters datasets for specific gripper positions 
#           > none: uses full dataset
#           > True/ False: uses multi-grasp positions or single_grasp only 
#           > 'h1'/'h2'/'l'/'m'/'r': gets specific gripper position (high 1, high 2, left, middle, right)
#
#       > tex_weight            [float]     : weight factor multiplied with texture loss        (default 1.0)
#       > mat_weight            [float]     : weight factor multiplied with material loss       (default 1.0)
#       > reconstruct_weight    [float]     : weight factor multiplied with reconstruction loss (default 1.0)
#       > early_stopping        [int/ None] : specifies after how many epochs without val loss improvement before training stops early
#       > reconstruct           [bool]      : specifies if the reconstruction training program should run
#       > freeze_epoch          [int]       : number op epochs after which CAE decoder will be frozen
#       > tex_classes           [str-list]  : list of all texture classes the model will train and test with
#       > mat_classes           [str-list]  : list of all material classes the model will train and test with
#       > filtering             [bool]      : will apply butterworth filter to all signals
#       > cropping              [bool]      : specifies if heavy signal cropping will be employed (cropping window can be changed in utils)
#       > normalise             [bool]      : normalises dataset
#       > augment               [bool]      : will augment dataset during training loop
#       > data_partition:
#           > True: Training on In-Distribution, tesing on Out-of-distribution (Out-of-distribution Results)
#           > False: Training and testing on combined dataset (In-Distribution Results)
#

class Manager():
    def __init__(self, model, file_pth='experiments/misc',
                 num_epochs=20, batch_size=15, shuffle=True, lr=0.001, weight_decay=0.01, multigrasp=None,
                 tex_weight=1.0, mat_weight=1.0, reconstruct_weight=1.0, early_stopping=None, 
                 reconstruct=True, freeze_epoch=5,
                 tex_classes=['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry'],
                 mat_classes=['ds20', 'ds30', 'ef10', 'ef30', 'ef50'],
                 filtering=False, cropping=True, normalise=False, augment=False, data_partition=True):

        # Hyperparams
        self.num_epochs                 = num_epochs
        self.batch_size                 = batch_size
        self.shuffle                    = shuffle
        self.learning_rate              = lr
        self.weight_decay               = weight_decay
        self.file_pth                   = file_pth
        self.tex_weight                 = tex_weight
        self.mat_weight                 = mat_weight
        self.reconstruct_weight         = reconstruct_weight
        self.early_stopping             = early_stopping if early_stopping is not None else num_epochs
        self.filtering                  = filtering
        self.cropping                   = cropping
        self.normalise                  = normalise
        self.augment                    = augment
        self.reconstruct                = reconstruct
        self.data_partition             = data_partition
        self.freeze_epoch               = freeze_epoch

        # Class Data
        self.tex_classes                = sorted(tex_classes)
        self.mat_classes                = sorted(mat_classes)

        # Loss Functions
        self.mat_loss                   = nn.CrossEntropyLoss()
        self.tex_loss                   = nn.CrossEntropyLoss()
        if reconstruct:
            self.AE_loss                = nn.MSELoss()

        # Model and Optimiser
        self.device                     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model                      = model.to(self.device)
        self.optim                      = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Dataset Management
        if data_partition:
            train_set              = signal_dataset.SignalDataset('data_final/multigrasp_train', multigrasp=multigrasp, filtering=filtering, cropping=cropping, normalise=False, augment=False, tex_classes=self.tex_classes, mat_classes=self.mat_classes)
            test_set               = signal_dataset.SignalDataset('data_final/multigrasp_test', multigrasp=multigrasp, filtering=filtering, cropping=cropping, normalise=False, augment=False, tex_classes=self.tex_classes, mat_classes=self.mat_classes)

            if normalise:
                self.train_mean, self.train_std = utils.compute_dataset_mean_std(train_set, batch_size=self.batch_size)
                train_set           = signal_dataset.SignalDataset('data_final/multigrasp_train', multigrasp=multigrasp, filtering=filtering, cropping=cropping, normalise=True, augment=False, mean=self.train_mean, std=self.train_std, tex_classes=self.tex_classes, mat_classes=self.mat_classes)
                
                test_set           = signal_dataset.SignalDataset('data_final/multigrasp_test', multigrasp=multigrasp, filtering=filtering, cropping=cropping, normalise=True, augment=False, mean=self.train_mean, std=self.train_std, tex_classes=self.tex_classes, mat_classes=self.mat_classes)

            train_set, val_set    = torch.utils.data.random_split(train_set, [0.7, 0.3], generator=g)

        else:
            # All Dataset
            self.full_dataset                   = signal_dataset.SignalDataset('data_final', multigrasp=multigrasp, filtering=filtering, cropping=cropping, normalise=False, augment=False, tex_classes=self.tex_classes, mat_classes=self.mat_classes)
            
            if normalise:
                self.train_mean, self.train_std = utils.compute_dataset_mean_std(self.full_dataset, batch_size=self.batch_size)
                self.full_dataset               = signal_dataset.SignalDataset('data_final', multigrasp=multigrasp, filtering=filtering, cropping=cropping, normalise=True, augment=False, mean=self.train_mean, std=self.train_std, tex_classes=self.tex_classes, mat_classes=self.mat_classes)
            
            train_set, test_set, val_set        = torch.utils.data.random_split(self.full_dataset, [2520, 1500, 1080], generator=g)        

        print(f'Train Dataset Length: {len(train_set)}')
        print(f'Test Dataset Length:  {len(test_set)}')
        print(f'Val Dataset Length:   {len(val_set)}')

        # Dataloader, Loss and Optimiser
        self.train_data                 = DataLoader(train_set, batch_size=self.batch_size, shuffle=shuffle, generator=g)
        self.test_data                  = DataLoader(test_set, batch_size=self.batch_size, shuffle=shuffle, generator=g)
        self.val_data                   = DataLoader(val_set, batch_size=self.batch_size, shuffle=shuffle, generator=g)

        self.save_model_params()

    # ------------------------------------------- Model Training ------------------------------------------

    def run_training(self, train_data, val_data):
        min_val_loss         = float('inf')
        epoch_no_improvement = 0
        saved_model_epoch    = 0

        train_loss_plot_data       = []
        val_loss_plot_data         = []
        train_acc_plot_data        = []
        val_acc_plot_data          = []
        gap_loss_plot_data         = []
        gap_acc_plot_data          = []

        for epoch in range(self.num_epochs):
            # —————————————————————————————————————————— Training pass 
            self.model.train()
            running_loss    = 0.0
            running_correct = 0
            running_total   = 0

            # Freezing CAE decoder after certain number of epochs
            if self.reconstruct and epoch == self.freeze_epoch:
                for p in model.deconv2.parameters(): p.requires_grad = False
                for p in model.dbn2.parameters():   p.requires_grad = False
                for p in model.deconv1.parameters(): p.requires_grad = False
                for p in model.dbn1.parameters():   p.requires_grad = False
                for p in model.deconv0.parameters(): p.requires_grad = False
                self.optim = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=1e-3
                )

            for i, batch in enumerate(train_data):

                # Getting data and targets
                signal, mat_target, tex_target = batch
                
                aug_rand = random.random()
                if self.augment and aug_rand < 0.5:
                    signal = self.augment_signal(signal)

                signal              = signal.to(self.device).float()
                mat_target          = mat_target.to(self.device).long()
                tex_target          = tex_target.to(self.device).long()

                # Forward Pass
                if self.reconstruct and epoch < self.freeze_epoch:
                    signal_out          = self.model(signal, reconstruct=True)
                    loss_rec            = self.AE_loss(signal_out, signal)

                mat_out, tex_out    = self.model(signal)
                mat_loss            = self.mat_loss(mat_out, mat_target)
                tex_loss            = self.tex_loss(tex_out, tex_target)
                loss_cls            = (self.mat_weight * mat_loss) + (self.tex_weight * tex_loss)

                if self.reconstruct and epoch < self.freeze_epoch:
                    loss = loss_cls + (self.reconstruct_weight * loss_rec) # type: ignore
                else:
                    loss = loss_cls

                # Accuracy Calculations
                mat_preds = mat_out.argmax(dim=1)
                tex_preds = tex_out.argmax(dim=1) 
                running_correct += (mat_preds == mat_target).sum().item() + (tex_preds == tex_target).sum().item()
                running_total   += mat_target.size(0) + tex_target.size(0)
                running_loss += loss.item()

                # Backprop
                self.optim.zero_grad()
                loss.backward() 
                self.optim.step()

                if i % 20 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], "
                        f"Step [{i}/{len(train_data)}], "
                        f"Loss: {loss.item():.4f}") 
            
            avg_train_loss = running_loss / len(train_data)
            train_acc      = running_correct / running_total
            train_acc_plot_data.append(train_acc)
            train_loss_plot_data.append(avg_train_loss)
            print(f"Epoch [{epoch+1}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc}") 
                
            # ——————————————————————————————————————— Validation pass
            self.model.eval()
            running_val_loss = 0.0
            val_correct      = 0
            val_total        = 0

            with torch.no_grad():
                for i, batch in enumerate(val_data):

                    # Getting data and targets
                    signal, mat_target, tex_target = batch
                    signal = signal.to(self.device).float()
                    mat_target          = mat_target.to(self.device).long()
                    tex_target          = tex_target.to(self.device).long()

                    # Forward Pass
                    if self.reconstruct and epoch < self.freeze_epoch:
                        signal_out          = self.model(signal, reconstruct=True)
                        loss_rec            = self.AE_loss(signal_out, signal)

                    mat_out, tex_out    = self.model(signal)
                    mat_loss            = self.mat_loss(mat_out, mat_target)
                    tex_loss            = self.tex_loss(tex_out, tex_target)
                    loss_cls            = (self.mat_weight * mat_loss) + (self.tex_weight * tex_loss)

                    if self.reconstruct and epoch < self.freeze_epoch:
                        loss = loss_cls + (self.reconstruct_weight * loss_rec) # type: ignore
                    else:
                        loss = loss_cls

                    # Acc Calculations
                    mat_preds = mat_out.argmax(dim=1)
                    tex_preds = tex_out.argmax(dim=1)
                    val_correct += (mat_preds == mat_target).sum().item() + (tex_preds == tex_target).sum().item()
                    val_total   += mat_target.size(0) + tex_target.size(0)
                    running_val_loss += loss.item()  

            avg_val_loss = running_val_loss / len(val_data)
            val_acc      = val_correct / val_total
            val_loss_plot_data.append(avg_val_loss)
            val_acc_plot_data.append(val_acc)

            # ——— Generalisation gaps ———
            gap_loss = avg_val_loss - avg_train_loss
            gap_acc  = train_acc - val_acc

            gap_loss_plot_data.append(gap_loss)
            gap_acc_plot_data.append(gap_acc)

            print(f"Epoch [{epoch+1}] Val Loss: {avg_val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}")
            print(f"Epoch [{epoch+1}] Gen Gap → Loss: {gap_loss:.4f}, "
                f"Acc: {gap_acc:.4f}\n")

            # ——— Early stopping bookkeeping ———
            if avg_val_loss < min_val_loss:    
                print(f'  → New best validation Loss: {avg_val_loss}')   
                min_val_loss = avg_val_loss        
                epoch_no_improvement = 0
                torch.save(self.model.state_dict(), f'{self.file_pth}/model_weights.pth')
                saved_model_epoch = epoch
                print(f'Model saved under {self.file_pth}/model_weights.pth')
            else:
                epoch_no_improvement += 1
                print(f'  → No improvement for {epoch_no_improvement} epoch(s). Val Loss: {avg_val_loss}')
                if epoch_no_improvement >= self.early_stopping:
                    print('Early stopping activated.')
                    break

        utils.loss_plots(self.file_pth,
            train_loss_plot_data,
            val_loss_plot_data,
            train_acc_plot_data,
            val_acc_plot_data,
            gap_loss_plot_data,
            gap_acc_plot_data,
            plotting=False,
            stopping_epoch=saved_model_epoch
        )

        print('Training complete')

    # ------------------------------------------- Model Testing -------------------------------------------

    def run_testing(self, test_data):

        self.load_model()
        self.model.eval()

        all_mat_targets, all_mat_preds = [], []
        all_tex_targets, all_tex_preds = [], []

        with torch.no_grad():
            for batch in test_data:
                signal, mat_target, tex_target = batch
                signal     = signal.to(self.device).float()
                mat_target = mat_target.to(self.device).long()
                tex_target = tex_target.to(self.device).long()
                
                mat_out, tex_out    = self.model(signal)
                mat_preds = mat_out.argmax(dim=1)
                tex_preds = tex_out.argmax(dim=1)

                all_mat_targets.append(mat_target.cpu())                                   
                all_mat_preds.append(mat_preds.cpu())                                 
                all_tex_targets.append(tex_target.cpu())                                       
                all_tex_preds.append(tex_preds.cpu())       
                    
        mat_true = torch.cat(all_mat_targets).numpy()                                           # type: ignore
        mat_pred = torch.cat(all_mat_preds).numpy()                                             # type: ignore
        tex_true = torch.cat(all_tex_targets).numpy()                                           # type: ignore
        tex_pred = torch.cat(all_tex_preds).numpy()                                             # type: ignore

        # Material
        mat_acc    = accuracy_score(mat_true, mat_pred)
        mat_report = classification_report(mat_true, mat_pred, digits=4)
        mat_cm     = confusion_matrix(mat_true, mat_pred)
        print(f"Material Test Accuracy: {mat_acc:.4f}")
        print("\nMaterial Classification Report:\n", mat_report)

        # Texture
        tex_acc    = accuracy_score(tex_true, tex_pred)
        tex_report = classification_report(tex_true, tex_pred, digits=4)
        tex_cm     = confusion_matrix(tex_true, tex_pred)
        print(f"\nTexture Test Accuracy: {tex_acc:.4f}")
        print("\nTexture Classification Report:\n", tex_report)

        # Combined
        combined_true = list(zip(mat_true, tex_true))
        combined_pred = list(zip(mat_pred, tex_pred))

        combined_true_str = [f"{m}_{t}" for m, t in combined_true]
        combined_pred_str = [f"{m}_{t}" for m, t in combined_pred]

        combined_acc = accuracy_score(combined_true_str, combined_pred_str)
        print(f'Combined ACC: {combined_acc}')
        combined_report = classification_report(combined_true_str, combined_pred_str, digits=4)
        combined_cm = confusion_matrix(combined_true_str, combined_pred_str)

        # Plot both with confusion_plotter_dual
        mat_cm_file = f"{self.file_pth}/confusion_matrix_material.png"
        tex_cm_file = f"{self.file_pth}/confusion_matrix_texture.png"
        comb_file   = f'{self.file_pth}/confusion_matrix_combined.png'

        comb_classes = [f"{m}_{t}" for m, t in itertools.product(self.mat_classes, self.tex_classes)]

        utils.plot_confusion_matrix(mat_cm, self.mat_classes, mat_cm_file)
        utils.plot_confusion_matrix(tex_cm, self.tex_classes, tex_cm_file)
        utils.plot_confusion_matrix(combined_cm, comb_classes, comb_file)

        # Save reports
        report_path = f'{self.file_pth}/test_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Material Classification Report ===\n")
            f.write(mat_report)                                                                 # type: ignore
            f.write("\n\n=== Texture Classification Report ===\n")
            f.write(tex_report)                                                                 # type: ignore
            f.write("\n\n=== Dual Classification Report ===\n")
            f.write(combined_report)                                                            # type: ignore

        if self.reconstruct:
            self.reconstruction_tst(test_data)

    # ------------------------------------------ Reconstruction -------------------------------------------

    def reconstruction_tst(self, test_data):
        self.load_model()
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(test_data):
                signal, mat_target, tex_target = batch
                signal = signal[0].unsqueeze(0).to(self.device).float()  # (1, C, L)
                mat_target = mat_target[0].unsqueeze(0).to(self.device).long()
                tex_target = tex_target[0].unsqueeze(0).to(self.device).long()
                
                signal_out          = self.model(signal, reconstruct=True)
                mat_out, tex_out    = self.model(signal)
                break

        mat_probs = torch.argmax(F.softmax(mat_out, dim=1))          # type: ignore
        tex_probs = torch.argmax(F.softmax(mat_out, dim=1))          # type: ignore

        print(f'Mat Target: {mat_target} | Mat Output: {mat_probs}') # type: ignore
        print(f'Tex Target: {tex_target} | Tex Output: {tex_probs}') # type: ignore

        signal        = utils.zscore_denormalize(signal, self.train_mean, self.train_std)     # type: ignore
        signal_out    = utils.zscore_denormalize(signal_out, self.train_mean, self.train_std) # type: ignore
        signal_in_np  = signal.squeeze(0).detach().cpu().numpy()     # type: ignore
        signal_out_np = signal_out.squeeze(0).detach().cpu().numpy() # type: ignore

        in_df = pd.DataFrame(signal_in_np)
        in_df = in_df.T
        in_df = in_df.iloc[:, :12]

        out_df = pd.DataFrame(signal_out_np)
        out_df = out_df.T
        out_df = out_df.iloc[:, :12]

        col_names = [
            f"{axis}{i}_calib"
            for i in range(1, 5)      # 1 to 4
            for axis in ["x", "y", "z"]
        ]
        in_df.columns  = col_names
        out_df.columns = col_names
        save_pth = rf'{self.file_pth}/reconstructed_signal.png'
        utils.sensor_plotter(in_df, out_df, save_pth=save_pth) # type: ignore

    # ------------------------------------- Loading and saving params -------------------------------------

    def load_model(self):
        ckpt = torch.load(f'{self.file_pth}/model_weights.pth', map_location=self.device, weights_only=True)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)

    def save_model_params(self):

        param_dict = {
            'num_epochs'        : self.num_epochs,
            'batch_size'        : self.batch_size,
            'early_stopping'    : self.early_stopping,
            'shuffle_batches'   : self.shuffle,
            'learning_rate'     : self.learning_rate,
            'weight_decay'      : self.weight_decay,
            'tex_weight'        : self.tex_weight,
            'mat_weight'        : self.mat_weight,
            'filtering'         : self.filtering,
            'cropping'          : self.cropping,
            'normalise'         : self.normalise,
            'augment'           : self.augment,
            'file_pth'          : self.file_pth,
            'Day2-3_partition'  : self.data_partition,
            'tex_classes'       : self.tex_classes,
            'mat_classes'       : self.mat_classes
        }

        if not os.path.exists(param_dict['file_pth']):
            os.makedirs(param_dict['file_pth'])

        with open(f'{self.file_pth}/model_params.json', 'w') as json_file:
            json.dump(param_dict, json_file, indent=4)

    # ------------------------------------- Signal Augmentation -------------------------------------

    def augment_signal(self, x, sigma=0.01, scale_range=(0.95,1.05), max_shift=8):
        # x: (B, C, L)
        if not self.model.training: return x
        B, C, L = x.shape
        noise = sigma * torch.randn_like(x)
        scale = torch.empty(B, 1, 1, device=x.device).uniform_(*scale_range)
        x = x * scale + noise
        shifts = torch.randint(-max_shift, max_shift+1, (B,), device=x.device)
        x = torch.stack([torch.roll(x[i], int(shifts[i].item()), dims=-1) for i in range(B)], dim=0)
        return x

if __name__ == '__main__':

    tex_classes=['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry']
    mat_classes=['ds20', 'ds30', 'ef10', 'ef30', 'ef50']

    model = model_CNN.Model()
    # model = model_CAE.Model()
    # model = model_LSTM.Model()

    manager = Manager(model, file_pth=f'EXPERIMENTS/CNN/run1', 
                    num_epochs=20, batch_size=32, shuffle=True,
                    tex_classes=tex_classes,
                    mat_classes=mat_classes,
                    multigrasp=None,
                    tex_weight=1.0,
                    mat_weight=1.5, 
                    reconstruct_weight=0.2,
                    early_stopping=None,
                    reconstruct=False,
                    filtering=False,
                    cropping=True,
                    normalise=True,
                    augment=False, 
                    data_partition=True)
    
    manager.run_training(manager.train_data, manager.val_data)
    manager.run_testing(manager.test_data)
