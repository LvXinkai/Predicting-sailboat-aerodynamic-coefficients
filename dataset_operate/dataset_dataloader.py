import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd

col_n = ['store_path', 'boatAngle', 'jibAngle', 'mainsailAngle', 'x_force_c', 'y_force_c']
sheet_n = ['test', 'train']


class FlowFieldCharacteristicsDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 path_prefix: str = 'D:\\TecplotData\\Flow_Field_Characteristics',
                 phase: str = "test",
                 transform=None
                 ):
        self.df = df
        self.path_prefix = path_prefix
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        pt = self.df.iloc[index].replace("\\", "/")
        dataset = np.load(os.path.join(self.path_prefix, pt))

        if self.transform:
            dataset = self.transform(dataset)
        dataset = dataset.flatten()
        dat, data_min, data_max = data_maxmin_normalize(dataset)
        dat = torch.tensor(dat, dtype=torch.float32)

        dat = dat.unsqueeze(dim=0)

        ar = pt.split('/')
        arr = ar[-1].split('_')

        return {
            'id': index,
            'data': dat,
            # 'mask': mask,
            'boatAngle': ar[-3],
            'jibAngle': ar[-2],
            'mainsailAngle': arr[-1][:-4],
            'data_min': data_min,
            'data_max': data_max,
            'path': pt
        }


class FlowFieldDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 path_prefix: str = 'D:\\TecplotData\\FlowField_64_V_and_P_IDW_mask',
                 phase: str = "test", transform=None
                 ):
        self.df = df
        self.path_prefix = path_prefix
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        pt = self.df.iloc[index].replace("\\", "/")
        # 加载数据
        dataset = np.load(os.path.join(self.path_prefix, pt))
        if self.transform:
            dataset = self.transform(dataset)

        dat, data_min, data_max = data_maxmin_normalize(dataset[-2])
        dat = torch.tensor(dat, dtype=torch.float32)
        mask = torch.tensor(dataset[-1], dtype=torch.float32)

        dat = dat.unsqueeze(dim=0)
        mask = mask.unsqueeze(dim=0)
        ar = pt.split('/')
        arr = ar[-1].split('_')

        return {
            'id': index,
            'data': dat,
            'mask': mask,
            'boatAngle': ar[-3],
            'jibAngle': ar[-2],
            'mainsailAngle': arr[-1][:-4],
            'data_min': data_min,
            'data_max': data_max,
            'path': pt
        }


def data_maxmin_normalize(data: np.ndarray):
    """Normilize image value between 0 and 1."""
    data_min = np.min(data)
    data_max = np.max(data)
    return ((data - data_min) / (data_max - data_min)), data_min, data_max


def data_channal_maxmin_normalize(data: np.ndarray):
    """Normilize image value between 0 and 1."""
    data_min = 0
    data_max = 0
    for i in range(data.shape[0]-1):
        data_min = np.min(data[i])
        data_max = np.max(data[i])
        data[i] = ((data[i] - data_min) / (data_max - data_min))
    return data[:-1], data_min, data_max


def restore_maxmin_normalized_data(data, data_min, data_max):
    return (data * (data_max - data_min)) + data_min


def get_dataloader(
        path_to_excel: str,
        data_path: str,
        phase: str,
        batch_size: int = 1,
        class_dataset=FlowFieldDataset,
        shuffle=True
):
    df = pd.read_excel(path_to_excel, usecols=col_n, sheet_name=sheet_n, header=0)
    if phase == sheet_n[1]:
        xlsx_data = df[sheet_n[1]].loc[:, col_n[0]]
    else:
        xlsx_data = df[sheet_n[0]].loc[:, col_n[0]]
    dataset = class_dataset(xlsx_data, data_path, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader
