import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd


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
        pt = self.df.iloc[index][0].replace("\\", "/")
        dataset = np.load(os.path.join(self.path_prefix, pt))
        dataset = dataset[::16, :, :, :]
        if self.transform:
            dataset = self.transform(dataset)
        dataset = dataset.flatten()

        dat, data_min, data_max = data_maxmin_normalize(dataset)
        dat = torch.tensor(dataset, dtype=torch.float32)

        boatAngle = self.df.iloc[index][1] / 180.0
        jibAngle = (self.df.iloc[index][2] + 30) / 60.0
        mainsailAngle = (self.df.iloc[index][3] + 90) / 180

        return {
            'id': index,
            'data': dat,
            'bjm': torch.tensor([
                    boatAngle,
                    jibAngle,
                    mainsailAngle
                ],
                dtype=torch.float32),
            'bjm_original': torch.tensor([
                self.df.iloc[index][1],
                self.df.iloc[index][2],
                self.df.iloc[index][3]
            ],
                dtype=torch.float32),
            'x_and_y_force_c': torch.tensor([
                    self.df.iloc[index][4],
                    self.df.iloc[index][5]
                ],
                dtype=torch.float32),
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
        class_dataset=FlowFieldCharacteristicsDataset,
        shuffle=True
):
    col_n = ['store_path', 'boatAngle', 'jibAngle', 'mainsailAngle', 'x_force_c', 'y_force_c']
    sheet_n = ['test', 'train', 'draw_pictures']

    df = pd.read_excel(path_to_excel, usecols=col_n, sheet_name=sheet_n, header=0)
    if phase == sheet_n[1]:
        xlsx_data = df[sheet_n[1]].loc[:, [col_n[0], col_n[1], col_n[2], col_n[3], col_n[4], col_n[5]]]
    elif phase == sheet_n[2]:
        xlsx_data = df[sheet_n[2]].loc[:, [col_n[0], col_n[1], col_n[2], col_n[3], col_n[4], col_n[5]]]
    else:
        xlsx_data = df[sheet_n[0]].loc[:, [col_n[0], col_n[1], col_n[2], col_n[3], col_n[4], col_n[5]]]
    dataset = class_dataset(xlsx_data, data_path, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle

    )
    return dataloader
