import utils, torch
from torch.utils.data import Dataset

class SignalDataset(Dataset):
    def __init__(self, root_dir, dual_cls, split='train', distribution=[0.7, 0.2, 0.1], filtering=False, cropping=False, normalise=False, augment=False):
        super().__init__()
        self.root_dir       = root_dir
        self.dual_cls       = dual_cls
        self.filtering      = filtering
        self.cropping       = cropping
        self.normalise      = normalise
        self.augment        = augment
        self.dirs_list      = utils.collect_files(root_dir, split, distribution)
        # self.dirs_list      = utils.collect_files_old(root_dir)

    def __len__(self):
        return len(self.dirs_list)

    def __getitem__(self, index):
        sensor_0 = utils.data_loader(self.dirs_list[index][0], cropping=self.cropping, filtering=self.filtering)
        sensor_1 = utils.data_loader(self.dirs_list[index][1], cropping=self.cropping, filtering=self.filtering)
        x        = utils.dfs_to_tensor_nearest(sensor_0, sensor_1)

        # [Optional]: Per Sample Normalisation 
        if self.normalise:
            mu   = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True)
            x = (x - mu) / (sigma + 1e-6)  

        # [Optional]: Data Augmentation
        if self.augment:
            x = self.augment_data(x)

        # Dual/ Single classification handling
        if self.dual_cls:
            mat_cls, tex_cls = utils.get_class_dual(self.dirs_list[index][2], self.dirs_list[index][3])
            return (x, torch.tensor(mat_cls, dtype=torch.int64), torch.tensor(tex_cls, dtype=torch.int64))
        else:
            target = utils.get_class(self.dirs_list[index][2], self.dirs_list[index][3])
            return (x, torch.tensor(target, dtype=torch.int64))
    
    def augment_data(self, x):
        x = self.time_shift(x)
        x = self.add_gaussian_noise(x)
        return x

    def time_shift(self, x, max_shift=50):
        shift = torch.randint(-max_shift, max_shift, (1,)).item()
        return x.roll(shifts=shift, dims=-1)
    
    def add_gaussian_noise(self, x, std=0.01):
        noise = torch.randn_like(x) * std
        return x + noise

# if __name__ == '__main__':
#     train_dat = SignalDataset('data')
#     test_dat = SignalDataset('data', split='test')
#     val_dat = SignalDataset('data', split='val')
    # entries = utils.collect_files('data')
    # print(len(dat))
    # print(entries[243])
    # print(dat[0][0].shape)
