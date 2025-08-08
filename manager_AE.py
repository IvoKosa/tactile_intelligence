# General Imports
import signal_dataset, model_AE, utils, json, os, shutil
import matplotlib.pyplot as plt

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import Saliency, IntegratedGradients, Occlusion
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Manager():
    def __init__(self, file_pth='experiments/misc',
                 num_epochs=25, batch_size=15, shuffle=True, lr=0.001, weight_decay=0.01, multigrasp=None,
                 distribution=[0.7, 0.2, 0.1], tex_weight=1.0, mat_weight=1.0, early_stopping=5,
                 filtering=False, cropping=False, normalise=False, augment=False):

        # Hyperparams
        self.num_epochs                 = num_epochs
        self.batch_size                 = batch_size
        self.shuffle                    = shuffle
        self.learning_rate              = lr
        self.weight_decay               = weight_decay
        self.file_pth                   = file_pth
        self.distribution               = distribution
        self.tex_weight                 = tex_weight
        self.mat_weight                 = mat_weight
        self.early_stopping             = early_stopping if early_stopping is not None else num_epochs
        self.filtering                  = filtering
        self.cropping                   = cropping
        self.normalise                  = normalise
        self.augment                    = augment

        # Dataset and Model
        self.device                     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mat_loss                   = nn.CrossEntropyLoss()
        self.tex_loss                   = nn.CrossEntropyLoss()

        self.model                      = model_AE.Tactile_CNN().to(self.device)
        self.AE_loss                    = nn.MSELoss()
    
        # Dataset Functions
        self.full_dataset               = signal_dataset.SignalDataset('data', True, multigrasp=multigrasp, filtering=filtering, cropping=cropping, normalise=False, augment=augment)
        if normalise:
            mean, std                   = utils.compute_dataset_mean_std(self.full_dataset)
            self.full_dataset           = signal_dataset.SignalDataset('data', True, multigrasp=multigrasp, filtering=filtering, cropping=cropping, normalise=True, augment=augment, mean=mean, std=std)
        train_set, test_set, val_set    = torch.utils.data.random_split(self.full_dataset, [0.7, 0.2, 0.1])

        print(f'Train Dataset Length: {len(train_set)}')
        print(f'Test Dataset Length:  {len(test_set)}')
        print(f'Val Dataset Length:   {len(val_set)}')

        # Dataloader, Loss and Optimiser
        self.train_data                 = DataLoader(train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.test_data                  = DataLoader(test_set, batch_size=self.batch_size, shuffle=shuffle)
        self.val_data                   = DataLoader(val_set, batch_size=self.batch_size, shuffle=shuffle)
        self.optim                      = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.save_model_params()

    # ------------------------------------------- Model Training ------------------------------------------

    def run_training(self, train_data, val_data):
        min_val_loss         = float('inf')
        epoch_no_improvement = 0

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

            for i, batch in enumerate(train_data):

                # Getting data and targets
                signal, mat_target, tex_target = batch
                signal              = signal.to(self.device).float()
                mat_target          = mat_target.to(self.device).long()
                tex_target          = tex_target.to(self.device).long()

                # Forward Pass
                signal_out          = self.model(signal)
                loss_rec            = self.AE_loss(signal_out, signal)

                mat_out, tex_out    = self.model(signal, classify=True)
                mat_loss            = self.mat_loss(mat_out, mat_target)
                tex_loss            = self.tex_loss(tex_out, tex_target)
                loss_cls            = (self.mat_weight * mat_loss) + (self.tex_weight * tex_loss)

                loss = loss_cls + loss_rec

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
            print(f"Epoch [{epoch+1}] Train Loss: {avg_train_loss:.4f}") 
                
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
                    signal_out          = self.model(signal)
                    loss_rec            = self.AE_loss(signal_out, signal)

                    mat_out, tex_out    = self.model(signal, classify=True)
                    mat_loss            = self.mat_loss(mat_out, mat_target)
                    tex_loss            = self.tex_loss(tex_out, tex_target)
                    loss_cls            = (self.mat_weight * mat_loss) + (self.tex_weight * tex_loss)

                    loss = loss_cls + loss_rec

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
                print(f'Model saved under {self.file_pth}/model_weights.pth')
            else:
                epoch_no_improvement += 1
                print(f'  → No improvement for {epoch_no_improvement} epoch(s).')
                if epoch_no_improvement > self.early_stopping:
                    print('Early stopping activated.')
                    break

        utils.loss_plots(self.file_pth,
            train_loss_plot_data,
            val_loss_plot_data,
            train_acc_plot_data,
            val_acc_plot_data,
            gap_loss_plot_data,
            gap_acc_plot_data,
            plotting=False
        )

        print('Training complete')

    # ------------------------------------------- Model Testing -------------------------------------------

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
            'shuffle_batches'   : self.shuffle,
            'learning_rate'     : self.learning_rate,
            'weight_decay'      : self.weight_decay,
            'test_train_split'  : self.distribution,
            'filtering'         : self.filtering,
            'cropping'          : self.cropping,
            'normalise'         : self.normalise,
            'augment'           : self.augment,
            'file_pth'          : self.file_pth
        }

        if not os.path.exists(param_dict['file_pth']):
            os.makedirs(param_dict['file_pth'])

        with open(f'{self.file_pth}/model_params.json', 'w') as json_file:
            json.dump(param_dict, json_file, indent=4)

        source_path         = f'model_AE.py'
        destination_path    = f'{self.file_pth}/model.py'
        shutil.copy2(source_path, destination_path)

    # ------------------------------------- Training and Testing -------------------------------------

if __name__ == '__main__':

    # TODO
    # Variations on Test/ Train sets:
    #   - Train on Multi-grasp only, test on normal
    #   - Try removing various classes for training
    #   - Train on only 1 class for either mat or tex

    manager = Manager(file_pth='experiments/autoencoder/run2', 
                      num_epochs=100, batch_size=20, shuffle=True,
                      multigrasp=None,
                      distribution=[0.7, 0.2, 0.1],
                      tex_weight=1.0,
                      mat_weight=1.0,
                      early_stopping=7,
                      filtering=False,
                      cropping=False,
                      normalise=True,
                      augment=False)
    
    manager.run_training(manager.train_data, manager.val_data)
    # manager.run_testing()
