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


class DeepRouteDataset(Dataset):
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
        return len(self.data['V_len'])

    def __getitem__(self, index):

        E_static_fea = self.data['E_static_fea'][index]

        V = self.data['V'][index]
        V_reach_mask = self.data['V_reach_mask'][index]
        V_dispatch_mask = self.data['V_dispatch_mask'][index]

        E_mask = self.data['E_mask'][index]
        label = self.data['label'][index]
        label_len = self.data['label_len'][index]
        V_len = self.data['V_len'][index]
        start_fea = self.data['start_fea'][index]
        start_idx = self.data['start_idx'][index]

        return E_static_fea, V, V_reach_mask, V_dispatch_mask, \
               E_mask, label, label_len, V_len, start_fea, start_idx


if __name__ == '__main__':
    pass
