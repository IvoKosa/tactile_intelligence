# General Imports
import signal_dataset, AE_model, utils, json, os, shutil
import matplotlib.pyplot as plt

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import Saliency, IntegratedGradients, Occlusion
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Manager():
    def __init__(self, file_pth='experiments/misc', 
                 num_epochs=20, batch_size=15, shuffle=True, lr=0.001, weight_decay=0.01, 
                 distribution=[0.7, 0.2, 0.1], tex_weight=1.0, mat_weight=1.0, early_stopping=3,
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
        self.model                      = AE_model.Tactile_CNN().to(self.device)
        self.dual_cls                   = self.model.dual_cls

        # Dataset Functions
        full_set                        = signal_dataset.SignalDataset('data', self.dual_cls, multigrasp='Both', filtering=filtering, cropping=cropping, normalise=normalise, augment=augment)
        train_set, test_set, val_set    = torch.utils.data.random_split(full_set, [0.7, 0.2, 0.1])

        print(f'Train Dataset Length: {len(train_set)}')
        print(f'Test Dataset Length:  {len(test_set)}')
        print(f'Val Dataset Length:   {len(val_set)}')

        # Dataloader, Loss and Optimiser
        self.train_data                 = DataLoader(train_set, batch_size=self.batch_size, shuffle=shuffle)
        self.test_data                  = DataLoader(test_set, batch_size=self.batch_size, shuffle=shuffle)
        self.val_data                   = DataLoader(val_set, batch_size=self.batch_size, shuffle=shuffle)
        self.optim                      = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if self.dual_cls:
            self.mat_loss               = nn.CrossEntropyLoss()
            self.tex_loss               = nn.CrossEntropyLoss()
        else:
            self.loss                   = nn.CrossEntropyLoss()

        self.save_model_params()

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

        source_path         = 'model.py'
        destination_path    = f'{self.file_pth}/model.py'
        shutil.copy2(source_path, destination_path)

    # ------------------------------------- Training and Testing -------------------------------------
