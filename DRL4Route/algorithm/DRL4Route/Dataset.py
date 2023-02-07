# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import json
from torch import Tensor, dtype
from torch.utils import data
from typing import Tuple, List, Dict
from torch.utils.data import Dataset
from random import shuffle, random, seed


class DRL4RouteDataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict, #parameters dict
    )->None:
        super().__init__()
        if mode not in ["train", "val", "test"]:  # "validate"
            raise ValueError
        path_key = {'train':'train_path', 'val':'val_path','test':'test_path'}[mode]
        path = params[path_key]
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):
        return len(self.data['label_len'])

    def __getitem__(self, index):

        V = self.data['V'][index]
        V_reach_mask = self.data['constraint_mask'][index]
        label = self.data['label'][index]
        label_len = self.data['label_len'][index]

        return V, V_reach_mask, label, label_len

if __name__ == '__main__':
    pass
