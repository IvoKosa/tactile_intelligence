import utils, torch
from torch.utils.data import Dataset
"""
Signal Dataset Class

"""

class SignalDataset(Dataset):
    def __init__(self, root_dir, multigrasp, filtering, cropping, normalise, augment, dual_cls=True,
                 mat_classes=None, tex_classes=None, mean=None, std=None, s0_only=False):
        super().__init__()
        self.root_dir        = root_dir
        self.dual_cls        = dual_cls
        self.filtering       = filtering
        self.cropping        = cropping
        self.normalise       = normalise
        self.augment         = augment
        self.mean            = mean
        self.std             = std
        self.s0_only         = s0_only
        self.mat_classes     = mat_classes if mat_classes is not None else sorted(['ds20', 'ds30', 'ef10', 'ef30', 'ef50'])
        self.tex_classes     = tex_classes if tex_classes is not None else sorted(['bigberry', 'citrus', 'rough', 'smallberry', 'smooth', 'strawberry'])
        self.dict_list       = utils.collect_file_info(root_dir, tex_classes=self.tex_classes, mat_classes=self.mat_classes)
        
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
        sensor_0 = utils.data_loader(self.dict_list[index]['s0_file_pth'], cropping=self.cropping, filtering=self.filtering, augment=self.augment)
        if self.s0_only:
            if 't_seconds' in sensor_0.columns:
                sensor_0 = sensor_0.drop(columns='t_seconds')
            if sensor_0.index.name == 't_seconds':
                sensor_0 = sensor_0.reset_index(drop=True)
            x = torch.from_numpy(sensor_0.to_numpy().T)
        else:
            sensor_1 = utils.data_loader(self.dict_list[index]['s1_file_pth'], cropping=self.cropping, filtering=self.filtering, augment=self.augment)
            x        = utils.dfs_to_tensor_nearest(sensor_0, sensor_1)

        # [Optional]: Normalisation 
        if self.normalise:
            mean = self.mean.view(-1, *([1] * (x.ndim-1)))  # type: ignore
            std  = self.std .view(-1, *([1] * (x.ndim-1)))  # type: ignore
            x = (x - mean) / std

        # Dual/ Single classification handling
        mat_cls = self.dict_list[index]['mat_cls_int']
        tex_cls = self.dict_list[index]['tex_cls_int']

        if self.dual_cls:
            return (x, torch.tensor(mat_cls, dtype=torch.int64), torch.tensor(tex_cls, dtype=torch.int64))
        else:
            target = mat_cls * len(self.tex_classes) + tex_cls
            return (x, torch.tensor(target, dtype=torch.int64))
