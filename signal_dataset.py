import utils, torch
from torch.utils.data import Dataset

class SignalDataset(Dataset):
    def __init__(self, root_dir, dual_cls, multigrasp, filtering, cropping, normalise, augment, 
                 mat_classes=None, tex_classes=None, mean=None, std=None):
        super().__init__()
        self.root_dir       = root_dir
        self.dual_cls       = dual_cls
        self.filtering      = filtering
        self.cropping       = cropping
        self.normalise      = normalise
        self.augment        = augment
        self.mean           = mean
        self.std            = std
        self.mat_classes    = sorted(['ds20', 'ds30', 'ef10', 'ef30', 'ef50', 'rigid'])
        self.tex_classes    = sorted(['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry'])
        self.dict_list      = utils.collect_file_info(root_dir, self.tex_classes, self.mat_classes)

        if multigrasp is True or multigrasp is False:
            self.dict_list = [item for item in self.dict_list if item.get('multigrasp') is multigrasp]
        elif multigrasp in ['h1', 'h2', 'l', 'm', 'r']:
            self.dict_list = [item for item in self.dict_list if item.get('grasp_pos') is multigrasp]

        if mat_classes is not None:
            self.dict_list = [item for item in self.dict_list if item.get("mat_cls_str") in set(mat_classes)]
        if tex_classes is not None:
            self.dict_list = [item for item in self.dict_list if item.get("tex_cls_str") in set(tex_classes)]

    def __len__(self):
        return len(self.dict_list)

    def __getitem__(self, index):
        sensor_0 = utils.data_loader(self.dict_list[index]['s0_file_pth'], cropping=self.cropping, filtering=self.filtering)
        sensor_1 = utils.data_loader(self.dict_list[index]['s1_file_pth'], cropping=self.cropping, filtering=self.filtering)
        x        = utils.dfs_to_tensor_nearest(sensor_0, sensor_1)

        # [Optional]: Normalisation 
        if self.normalise:
            mean = self.mean.view(-1, *([1] * (x.ndim-1)))  # type: ignore
            std  = self.std .view(-1, *([1] * (x.ndim-1)))  # type: ignore
            x = (x - mean) / std

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

if __name__ =='__main__':
    dat = SignalDataset('data', True, None, False, False, False, False)
    print(len(dat))
