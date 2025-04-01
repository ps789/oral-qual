import numpy as np
import torch
import os
import random


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) # gpu
    torch.cuda.manual_seed_all(seed)
    
def load_model(model, path_dyn, path_rec, device):
    model.dyn.load_state_dict(torch.load(path_dyn, map_location=device, weights_only=True))
    model.rec.load_state_dict(torch.load(path_rec, map_location=device, weights_only=True))
    return model

def l2_relative_error(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2)) / np.sqrt(np.sum(y_true ** 2))

def time_slice_l2_error(y_true, y_pred):
    if y_true.ndim == 2:
        return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)) / np.sqrt(np.sum(y_true ** 2, axis=1))
    elif y_true.ndim == 3:
        return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=(1,2))) / np.sqrt(np.sum(y_true ** 2, axis=(1,2))) 
    else:
        return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=(0,2,3))) / np.sqrt(np.sum(y_true ** 2, axis=(0,2,3))) 