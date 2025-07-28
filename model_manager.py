# General Imports
import signal_dataset, model, MLP_Model, utils, json, os, shutil
import matplotlib.pyplot as plt

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import Saliency, IntegratedGradients, Occlusion
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Manager():
    def __init__(self, file_pth='experiments/misc', num_epochs=20, batch_size=15, shuffle=True, lr=0.001, weight_decay=0.01,
                 distribution=[0.7, 0.2, 0.1], filtering=False, cropping=False, normalise=False, augment=False):

        # Hyperparams
        self.num_epochs                 = num_epochs
        self.batch_size                 = batch_size
        self.shuffle                    = shuffle
        self.learning_rate              = lr
        self.weight_decay               = weight_decay
        self.file_pth                   = file_pth
        self.distribution               = distribution
        self.filtering                  = filtering
        self.cropping                   = cropping
        self.normalise                  = normalise
        self.augment                    = augment

        # Dataset and Model
        self.device                     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model                      = model.Tactile_CNN().to(self.device)
        self.dual_cls                   = self.model.dual_cls
        # self.model                      = MLP_Model.SimpleMLP().to(self.device)

        # Old Dataset Functions
        # self.full_dataset               = signal_dataset.SignalDataset('data', filtering=filtering, cropping=cropping, normalise=normalise, augment=augment)
        # gen_seed                        = torch.Generator().manual_seed(42)
        # train_set, test_set, val_set    = random_split(self.full_dataset, [0.7, 0.2, 0.1], generator=gen_seed)

        # New Dataset Functions
        train_set                       = signal_dataset.SignalDataset('data', self.dual_cls, 'train', distribution , filtering=filtering, cropping=cropping, normalise=normalise, augment=augment)
        test_set                        = signal_dataset.SignalDataset('data', self.dual_cls, 'test', distribution , filtering=filtering, cropping=cropping, normalise=normalise, augment=False)
        val_set                         = signal_dataset.SignalDataset('data', self.dual_cls, 'val', distribution , filtering=filtering, cropping=cropping, normalise=normalise, augment=False)
        
        print(f'Train Dataset Length: {len(train_set)}')
        print(f'Test Dataset Length: {len(test_set)}')
        print(f'Val Dataset Length: {len(val_set)}')

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

    def run_training(self, plotting=False):

        min_val_loss         = float('inf')
        epoch_no_improvement = 0

        train_loss_plot_data       = []
        val_loss_plot_data         = []
        train_acc_plot_data        = []
        val_acc_plot_data          = []
        gap_loss_plot_data         = []
        gap_acc_plot_data          = []

        for epoch in range(self.num_epochs):
            # ——— Training pass ———
            self.model.train()
            running_loss = 0.0
            running_correct = 0
            running_total   = 0

            for i, batch in enumerate(self.train_data):
                if self.dual_cls:
                    signal, mat_target, tex_target = batch
                    signal              = signal.to(self.device).float()
                    mat_target          = mat_target.to(self.device).long()
                    tex_target          = tex_target.to(self.device).long()

                    mat_out, tex_out    = self.model(signal)
                    mat_loss            = self.mat_loss(mat_out, mat_target)
                    tex_loss            = self.tex_loss(tex_out, tex_target)
                    loss                = mat_loss + tex_loss
                    running_loss += loss.item()
                    mat_preds = mat_out.argmax(dim=1)
                    tex_preds = tex_out.argmax(dim=1)
                    running_correct += (mat_preds == mat_target).sum().item() + (tex_preds == tex_target).sum().item()
                    running_total   += mat_target.size(0) + tex_target.size(0)

                else:
                    signal, target      = batch
                    signal              = signal.to(self.device).float()
                    target              = target.to(self.device).long()

                    out                 = self.model(signal)
                    loss                = self.loss(out, target)
                    running_loss += loss.item()

                    preds = out.argmax(dim=1)
                    running_correct += (preds == target).sum().item()
                    running_total   += target.size(0)
                

                # forward + backward
                # out   = self.model(signal)
                # loss  = self.loss(out, target)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # accumulate loss
                # running_loss += loss.item()

                # accumulate accuracy
                # preds = out.argmax(dim=1)
                # running_correct += (preds == target).sum().item()
                # running_total   += target.size(0)

                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], "
                        f"Step [{i}/{len(self.train_data)}], "
                        f"Loss: {loss.item():.4f}")

            # compute epoch training metrics
            avg_train_loss = running_loss / len(self.train_data)
            train_acc      = running_correct / running_total

            train_loss_plot_data.append(avg_train_loss)
            train_acc_plot_data.append(train_acc)

            print(f"Epoch [{epoch+1}] Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}")

            # ——— Validation pass ———
            self.model.eval()
            running_val_loss = 0.0
            val_correct      = 0
            val_total        = 0

            with torch.no_grad():
                for i, batch in enumerate(self.val_data):

                    if self.dual_cls:
                        signal, mat_target, tex_target = batch
                        signal              = signal.to(self.device).float()
                        mat_target          = mat_target.to(self.device).long()
                        tex_target          = tex_target.to(self.device).long()

                        mat_out, tex_out    = self.model(signal)
                        mat_loss            = self.mat_loss(mat_out, mat_target)
                        tex_loss            = self.tex_loss(tex_out, tex_target)
                        loss                = mat_loss + tex_loss
                        running_val_loss += loss.item()
                        mat_preds = mat_out.argmax(dim=1)
                        tex_preds = tex_out.argmax(dim=1)
                        val_correct += (mat_preds == mat_target).sum().item() + (tex_preds == tex_target).sum().item()
                        val_total   += mat_target.size(0) + tex_target.size(0)

                    else:
                        signal, target      = batch
                        signal              = signal.to(self.device).float()
                        target              = target.to(self.device).long()

                        out                 = self.model(signal)
                        loss                = self.loss(out, target)
                        running_val_loss += loss.item()

                        preds = out.argmax(dim=1)
                        val_correct += (preds == target).sum().item()
                        val_total   += target.size(0)

            avg_val_loss = running_val_loss / len(self.val_data)
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
                print('  → New best validation Loss.')
                min_val_loss = avg_val_loss
                epoch_no_improvement = 0
                torch.save(self.model.state_dict(), f'{self.file_pth}/model_weights.pth')
                print(f'Model saved under {self.file_pth}/model_weights.pth')
            else:
                epoch_no_improvement += 1
                print(f'  → No improvement for {epoch_no_improvement} epoch(s).')
                if epoch_no_improvement > 2:
                    print('Early stopping activated.')
                    break

        print('Training complete.')

        self.loss_plots(
            train_loss_plot_data,
            val_loss_plot_data,
            train_acc_plot_data,
            val_acc_plot_data,
            gap_loss_plot_data,
            gap_acc_plot_data,
            plotting=plotting
        )

    # def run_testing(self, plotting=False):
    #     self.load_model()
    #     self.model.eval()

    #     all_targets = []
    #     all_preds   = []

    #     with torch.no_grad():
    #         for signal, target in self.test_data:
    #             signal = signal.to(self.device).float()
    #             target = target.to(self.device).long()
                
    #             outputs = self.model(signal)           # [batch, num_classes]
    #             preds   = torch.argmax(outputs, dim=1) # [batch]
                
    #             all_targets.append(target.cpu())
    #             all_preds.append(preds.cpu())


    #     # Concatenate Batches
    #     y_true = torch.cat(all_targets).numpy()
    #     y_pred = torch.cat(all_preds).numpy()

    #     # Compute Metrics
    #     acc    = accuracy_score(y_true, y_pred)
    #     report = classification_report(y_true, y_pred, digits=4)
    #     cm     = confusion_matrix(y_true, y_pred)

    #     # Display / Return
    #     print(f"Test Accuracy: {acc:.4f}")
    #     print("\nClassification Report:\n", report)
    #     cm_filename = f'{self.file_pth}/confusion_matrix.png'
    #     utils.confusion_plotter(cm, cm_filename, plotting)

    #     with open(f'{self.file_pth}/test_reports.txt', 'w' , encoding="utf-8") as file:
    #         file.write(report)  # type: ignore

    def run_testing(self, plotting=False):
        self.load_model()
        self.model.eval()

        # Prepare containers
        if self.dual_cls:
            all_mat_targets, all_mat_preds = [], []
            all_tex_targets, all_tex_preds = [], []
        else:
            all_targets, all_preds = [], []

        with torch.no_grad():
            for batch in self.test_data:
                if self.dual_cls:
                    signal, mat_target, tex_target = batch
                    signal     = signal.to(self.device).float()
                    mat_target = mat_target.to(self.device).long()
                    tex_target = tex_target.to(self.device).long()

                    mat_out, tex_out = self.model(signal)
                    mat_preds = mat_out.argmax(dim=1)
                    tex_preds = tex_out.argmax(dim=1)

                    all_mat_targets.append(mat_target.cpu())                                        # type: ignore
                    all_mat_preds.append(mat_preds.cpu())                                           # type: ignore
                    all_tex_targets.append(tex_target.cpu())                                        # type: ignore
                    all_tex_preds.append(tex_preds.cpu())                                           # type: ignore
                else:
                    signal, target = batch
                    signal = signal.to(self.device).float()
                    target = target.to(self.device).long()

                    out   = self.model(signal)
                    preds = out.argmax(dim=1)

                    all_targets.append(target.cpu())                                                # type: ignore
                    all_preds.append(preds.cpu())                                                   # type: ignore

        # Concatenate & report
        if self.dual_cls:
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

            # Plot both with confusion_plotter_dual
            mat_cm_file = f"{self.file_pth}/confusion_matrix_material.png"
            tex_cm_file = f"{self.file_pth}/confusion_matrix_texture.png"
            utils.confusion_plotter_dual(
                mat_cm, tex_cm,
                mat_cm_file, tex_cm_file,
                plotting=plotting,
                normalize='true'
            )

            # Save reports
            with open(f'{self.file_pth}/test_reports_material.txt', 'w', encoding="utf-8") as f_mat, \
                open(f'{self.file_pth}/test_reports_texture.txt',  'w', encoding="utf-8") as f_tex:
                f_mat.write(mat_report)                                                             # type: ignore
                f_tex.write(tex_report)                                                             # type: ignore

        else:
            y_true = torch.cat(all_targets).numpy()                                                 # type: ignore
            y_pred = torch.cat(all_preds).numpy()                                                   # type: ignore 

            acc    = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, digits=4)
            cm     = confusion_matrix(y_true, y_pred)

            print(f"Test Accuracy: {acc:.4f}")
            print("\nClassification Report:\n", report)

            cm_filename = f'{self.file_pth}/confusion_matrix.png'
            utils.confusion_plotter(cm, cm_filename, plotting)

            with open(f'{self.file_pth}/test_reports.txt', 'w', encoding="utf-8") as file:
                file.write(report)                                                                  # type: ignore

    def inspect_model(self):
        self.load_model()
        self.model.eval()
        dataset = self.test_data.dataset

        with torch.no_grad():
            signal, target = dataset[0]
            x      = signal.unsqueeze(0).to(self.device).float()
            target = target.unsqueeze(0).to(self.device).long()

        x.requires_grad_()

        # saliency = Saliency(self.model)
        ig = IntegratedGradients(self.model)
        
        # attr = saliency.attribute(x, target=target)
        attr = ig.attribute(x, baselines=torch.zeros_like(x),
                       target=target,
                       n_steps=50)

        attr_np = attr.detach().cpu().squeeze().numpy()   # shape: (C, T)
        time = range(attr_np.shape[1])

        # fig, axes = plt.subplots(nrows=attr_np.shape[0], figsize=(10, 2*attr_np.shape[0]))
        fig, axes = plt.subplots(nrows=3, figsize=(10, 2*attr_np.shape[0]))
        for c, ax in enumerate(axes):
            ax.plot(time, x.detach().cpu().squeeze()[c].numpy(), label=f'Channel {c}')
            ax_t = ax.twinx()
            ax_t.plot(time, attr_np[c], color='C1', alpha=0.5, label='Saliency')
            ax.set_title(f'Channel {c}')
        plt.tight_layout()
        plt.show()

    def load_model(self):
        ckpt = torch.load(f'{self.file_pth}/model_weights.pth', map_location=self.device, weights_only=True)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)

    def loss_plots(self,
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
        plt.savefig(f'{self.file_pth}/loss_plot.png')

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
        plt.savefig(f'{self.file_pth}/accuracy_plot.png')

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
        plt.savefig(f'{self.file_pth}/generalisation_plot.png')

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

if __name__ == '__main__':

    manager = Manager(file_pth='experiments/dual_loss/train1', 
                      num_epochs=15, batch_size=15,
                      distribution=[0.7, 0.2, 0.1],
                      filtering=False,
                      cropping=False,
                      normalise=False,
                      augment=False)
    
    # manager.run_training()
    manager.run_testing()

    texture_list    = ['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry']
    material_list   = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50']

    for i in material_list:
        for j in texture_list:
            print(f'{i} {j} = {utils.get_class(i, j)}')

    # TODO
    # Modify Manager class to accept dual parameter classification for tex and mat individually