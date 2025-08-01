import utils, torch
from torch.utils.data import Dataset

class SignalDataset(Dataset):
    def __init__(self, root_dir, dual_cls, multigrasp, filtering, cropping, normalise, augment):
        super().__init__()
        self.root_dir       = root_dir
        self.dual_cls       = dual_cls
        self.filtering      = filtering
        self.cropping       = cropping
        self.normalise      = normalise
        self.augment        = augment
        self.mat_classes    = ['ds20', 'ds30', 'ef10', 'ef30', 'ef50', 'rigid']
        self.tex_classes    = ['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry']
        self.dict_list      = utils.collect_file_info(root_dir, self.tex_classes, self.mat_classes)

        match multigrasp:
            case 'Only':
                self.dict_list = [item for item in self.dict_list if item.get('multigrasp') is True]
            case 'None':
                self.dict_list = [item for item in self.dict_list if item.get('multigrasp') is False]
            case 'Both':
                pass
            case _:
                raise ValueError('Multigrasp should specify one of the following options: Only/None/Both')

        # self.dict_list = utils.split_dataset(self.dict_list, distribution, split)

    def __len__(self):
        return len(self.dict_list)

    def __getitem__(self, index):
        sensor_0 = utils.data_loader(self.dict_list[index]['s0_file_pth'], cropping=self.cropping, filtering=self.filtering)
        sensor_1 = utils.data_loader(self.dict_list[index]['s1_file_pth'], cropping=self.cropping, filtering=self.filtering)
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
        mat_cls = self.dict_list[index]['mat_cls_int']
        tex_cls = self.dict_list[index]['tex_cls_int']

        if self.dual_cls:
            return (x, torch.tensor(mat_cls, dtype=torch.int64), torch.tensor(tex_cls, dtype=torch.int64))
        else:
            target = mat_cls * len(self.tex_classes) + tex_cls
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
